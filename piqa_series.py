import torch, math, json
from types import SimpleNamespace as NS
import sys; sys.path.insert(0, '/root/phi')
from pretrain_tb_fast import LM
from transformers import AutoTokenizer
from datasets import load_dataset
import os
DEV="cuda:0"; B=int(os.environ.get("EB", "2"))
tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
cfg = NS(mode='tied', d=2048, ffn=7168, layers=16, block=32, rank=64, stages=2,
         vocab=100352, seq=4096, rope_base=10000.0, chunked_ce=0)
m = LM(cfg)
import os
CK = os.environ['CKPT']
sd = torch.load(CK, map_location='cpu', weights_only=True)
m.load_state_dict(sd.get('model', sd), strict=False); m.to(DEV).eval()
print(f"loaded step {sd.get('step','?')}", flush=True)

def ll_rank(pairs):
    """pairs: (ctx, [cont1..contN]) -> chosen index by sum-LL and by norm-LL."""
    jobs = []
    for ci, (ctx, conts) in enumerate(pairs):
        ce = tok(ctx).input_ids
        for ri, cont in enumerate(conts):
            ids = ce + tok(cont).input_ids
            jobs.append((ci, ri, ids, len(ce)))
    jobs.sort(key=lambda j: len(j[2]))
    ssum, snorm = {}, {}
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i+B]
        L = max(len(j[2]) for j in chunk)
        inp = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
        for k, (_, _, ids, _) in enumerate(chunk): inp[k,:len(ids)] = torch.tensor(ids)
        with torch.no_grad():
            lg, _ = m(inp.to(DEV))
        lsm = torch.log_softmax(lg.float(), -1)
        for k, (ci, ri, ids, nctx) in enumerate(chunk):
            pos = torch.arange(nctx-1, len(ids)-1, device=DEV)
            tg = torch.tensor(ids[nctx:], device=DEV)
            ll = lsm[k, pos, tg].sum().item()
            ssum[(ci,ri)] = ll; snorm[(ci,ri)] = ll/max(1, len(ids)-nctx)
    acc = accn = n = 0
    for ci, (ctx, conts) in enumerate(pairs):
        a = max(range(len(conts)), key=lambda r: ssum[(ci,r)])
        an = max(range(len(conts)), key=lambda r: snorm[(ci,r)])
        n += 1
        acc += a == 0; accn += an == 0
    return acc/n, accn/n, n

import os
# --- PIQA ---
ds = load_dataset("ybisk/piqa", split="test", revision="refs/convert/parquet")
pairs = []
for d in ds:
    ctx = f"Question: {d['goal']}\nAnswer:"
    pairs.append((ctx, [" "+d['sol1'], " "+d['sol2']]))
pq, pqn, n = ll_rank(pairs)
print(f"PIQA          acc {pq:.4f}  (n={n})", flush=True)

