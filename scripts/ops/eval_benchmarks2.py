import torch, re
from types import SimpleNamespace as NS
import sys; sys.path.insert(0, '/root/phi')
from pretrain_tb_fast import LM
from transformers import AutoTokenizer
from datasets import load_dataset
DEV='cuda:3'; B=32
tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
cfg = NS(mode='tied', d=2048, ffn=7168, layers=16, block=32, rank=64, stages=2,
         vocab=100352, seq=4096, rope_base=10000.0, chunked_ce=0)
m = LM(cfg)
sd = torch.load('/root/phi/ckpt_tied-16L2048d-b32-r64-sota1B_mile_14000.pt', map_location='cpu', weights_only=True)
m.load_state_dict(sd.get('model', sd), strict=False); m.to(DEV).eval()
print(f"loaded step {sd.get('step','?')}", flush=True)

def ll_rank(pairs):
    """pairs: (ctx, [conts], gold_idx) -> acc, acc_norm"""
    jobs = []
    for ci, (ctx, conts, gold) in enumerate(pairs):
        ce = tok(ctx).input_ids
        for ri, cont in enumerate(conts):
            ids = ce + tok(cont).input_ids
            jobs.append((ci, ri, ids, len(ce)))
    jobs.sort(key=lambda j: len(j[2]))
    ssum, snorm = {}, {}
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i+B]
        L = max(len(j[2]) for j in chunk)
        inp = torch.zeros((len(chunk), L), dtype=torch.long)
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
    for ci, (ctx, conts, gold) in enumerate(pairs):
        a = max(range(len(conts)), key=lambda r: ssum[(ci,r)])
        an = max(range(len(conts)), key=lambda r: snorm[(ci,r)])
        n += 1; acc += a == gold; accn += an == gold
    return acc/n, accn/n, n

ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
pairs = []
for d in ds:
    if not d['choices']['text'] or d['answerKey'] not in d['choices']['label']: continue
    gold = d['choices']['label'].index(d['answerKey'])
    pairs.append((f"Question: {d['question']}\nAnswer:", [" "+c for c in d['choices']['text']], gold))
arc, arcn, n = ll_rank(pairs)
print(f"ARC-Easy      acc {arc:.4f}  acc_norm {arcn:.4f}  (n={n})", flush=True)

ds = load_dataset("Rowan/hellaswag", split="validation").select(range(3000))
pairs = []
for d in ds:
    ctx = f"{d['activity_label']}: {d['ctx_a']} {d['ctx_b'].capitalize()}"
    pairs.append((ctx, [" "+e for e in d['endings']], int(d['label'])))
hs, hsn, n = ll_rank(pairs)
print(f"HellaSwag     acc {hs:.4f}  acc_norm {hsn:.4f}  (n={n})", flush=True)

ds = load_dataset('EleutherAI/lambada_openai', split='test', revision='refs/convert/parquet')['text']
nonascii = re.compile(r'[^\x20-\x7e\u2018\u2019\u201c\u201d\u00e9\u00e8\u00e0\u00e7]')
en = [t for t in ds if len(nonascii.findall(t)) <= 2 and ' the ' in t.lower()]
sel = en[:2000]
acc = n = 0
for i in range(0, len(sel), B):
    docs = sel[i:i+B]
    ctxs = [d.rsplit(' ', 1)[0] for d in docs]
    tgts = [" "+d.rsplit(' ', 1)[1] for d in docs]
    enc = [tok(c).input_ids for c in ctxs]
    order = sorted(range(len(docs)), key=lambda k: len(enc[k]))
    for bi in range(0, len(order), B):
        chunk = order[bi:bi+B]
        L = max(len(enc[k]) for k in chunk)
        inp = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, ci in enumerate(chunk): inp[k,:len(enc[ci])] = torch.tensor(enc[ci])
        with torch.no_grad():
            lg, _ = m(inp.to(DEV))
        am = lg.argmax(-1)
        for k, ci in enumerate(chunk):
            t = tok(tgts[ci]).input_ids[0]
            n += 1; acc += am[k, len(enc[ci])-1].item() == t
print(f"LAMBADA       acc {acc/n:.4f}  (n={n})", flush=True)
print("DONE", flush=True)
