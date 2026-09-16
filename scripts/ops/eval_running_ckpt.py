import torch, math
from types import SimpleNamespace as NS
import sys; sys.path.insert(0, '/root/phi')
from pretrain_tb_fast import LM
CKPT='/root/phi/ckpt_tied-16L2048d-b32-r64-sota1B_mile_14000.pt'
DEV='cuda:3'
cfg = NS(mode='tied', d=2048, ffn=7168, layers=16, block=32, rank=64, stages=2,
         vocab=100352, seq=4096, rope_base=10000.0, chunked_ce=0)
m = LM(cfg)
sd = torch.load(CKPT, map_location='cpu', weights_only=True)
msd = sd.get('model', sd)
res = m.load_state_dict(msd, strict=False)
print(f"missing {len(res.missing_keys)} unexpected {len(res.unexpected_keys)} step {sd.get('step','?')}")
m.to(DEV).eval()
# 1. perplexity on the held-out val
val = torch.load('/root/phi/val_generic_500M.pt', weights_only=True)
flat = val.view(-1).long()
n = (len(flat)-1)//8192*8192
x = flat[:n].view(-1,4096); y = flat[1:n+1].view(-1,4096)
tot, cnt = 0.0, 0
with torch.no_grad():
    for i in range(0, 200, 10):
        xs, ys = x[i:i+10, :4095].to(DEV), y[i:i+10, :4095].to(DEV)
        _, loss = m(xs, labels=xs)
        tot += loss.item()*xs.numel(); cnt += xs.numel()
ppl = tot/cnt
print(f"VAL generic: {ppl:.4f} nats/token ({math.exp(ppl):.1f} ppl)")
# 2. greedy-temperature sample generations + degenerate-loop check
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
torch.manual_seed(7)
prompts = ["The capital of France is", "Once upon a time in a distant land,",
           "def fibonacci(n):", "The president of the United States said",
           "Water boils at a temperature of"]
with torch.no_grad():
    for p in prompts:
        ids = tok(p, return_tensors='pt').input_ids.to(DEV)
        cur = ids
        for _ in range(48):
            lg, _ = m(cur[:, -1024:])
            nxt = lg[0,-1].argmax().view(1,1)
            cur = torch.cat([cur, nxt], 1)
        text = tok.decode(cur[0, ids.shape[1]:])
        words = text.split(); tris = [tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
        rep = len(tris)-len(set(tris))
        print(f"[{p[:28]!r}] -> {text[:110]!r} (rep {rep})")
