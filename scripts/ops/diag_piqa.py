import torch
from types import SimpleNamespace as NS
import sys; sys.path.insert(0, '/root/phi')
from pretrain_tb_fast import LM
from transformers import AutoTokenizer
from datasets import load_dataset
DEV='cuda:3'
tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
cfg = NS(mode='tied', d=2048, ffn=7168, layers=16, block=32, rank=64, stages=2,
         vocab=100352, seq=4096, rope_base=10000.0, chunked_ce=0)
m = LM(cfg)
sd = torch.load('/root/phi/ckpt_tied-16L2048d-b32-r64-sota1B_mile_14000.pt', map_location='cpu', weights_only=True)
m.load_state_dict(sd.get('model', sd), strict=False); m.to(DEV).eval()
ds = load_dataset("ybisk/piqa", split="test", revision="refs/convert/parquet")
def ll(ctx, cont):
    ids = tok(ctx).input_ids + tok(cont).input_ids
    nctx = len(tok(ctx).input_ids)
    with torch.no_grad():
        lg, _ = m(torch.tensor(ids).view(1,-1).to(DEV))
    lsm = torch.log_softmax(lg.float(), -1)[0]
    pos = torch.arange(nctx-1, len(ids)-1, device=DEV)
    tg = torch.tensor(ids[nctx:], device=DEV)
    return lsm[pos, tg].sum().item()
for i in range(6):
    d = ds[i]
    ctx = f"Question: {d['goal']}\nAnswer:"
    l1, l2 = ll(ctx, " "+d['sol1']), ll(ctx, " "+d['sol2'])
    n1, n2 = len(tok(d['sol1']).input_ids), len(tok(d['sol2']).input_ids)
    print(f"[{d['label']}] LL1 {l1:8.2f} ({n1} tok) LL2 {l2:8.2f} ({n2} tok) | goal: {d['goal'][:50]!r}")
# also raw-format (no Q/A wrapper) on 200 examples
acc_raw = n = 0
for i in range(200):
    d = ds[i]
    l1, l2 = ll(d['goal'], " "+d['sol1']), ll(d['goal'], " "+d['sol2'])
    n += 1; acc_raw += (l1 > l2) == (d['label'] == 0)
print(f"PIQA raw-format (no wrapper) acc on 200: {acc_raw/n:.4f}")
acc_w = 0
for i in range(200):
    d = ds[i]
    l1, l2 = ll(f"Question: {d['goal']}\nAnswer:", " "+d['sol1']), ll(f"Question: {d['goal']}\nAnswer:", " "+d['sol2'])
    acc_w += (l1 > l2) == (d['label'] == 0)
print(f"PIQA Q/A-wrapper acc on 200: {acc_w/n:.4f}")
