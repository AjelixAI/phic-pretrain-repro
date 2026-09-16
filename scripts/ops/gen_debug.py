import torch
from types import SimpleNamespace as NS
import sys; sys.path.insert(0, '/root/phi')
from pretrain_tb_fast import LM
from transformers import AutoTokenizer
cfg = NS(mode='tied', d=2048, ffn=7168, layers=16, block=32, rank=64, stages=2,
         vocab=100352, seq=4096, rope_base=10000.0, chunked_ce=0)
m = LM(cfg); import sys; CKPT = sys.argv[1] if len(sys.argv) > 1 else '/root/phi/ckpt_tied-16L2048d-b32-r64-sota1B_mile_14000.pt'
sd = torch.load(CKPT, map_location='cpu', weights_only=True)
m.load_state_dict(sd.get('model', sd), strict=False); m.to('cuda:3').eval()
tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
# 1. round-trip sanity
p = "The capital of France is"
ids = tok(p).input_ids
print("roundtrip:", repr(tok.decode(ids)))
# 2. val-slice prompt: argmax vs actual next
val = torch.load('/root/phi/val_generic_500M.pt', weights_only=True).view(-1).long()
ctx = val[:20].view(1,-1).to('cuda:3')
with torch.no_grad():
    lg, _ = m(ctx)
    nxt = lg[0,-1].argmax().item()
print("val-prompt argmax:", nxt, "actual next:", val[20].item(), "match:", nxt==val[20].item())
print("ctx decoded:", repr(tok.decode(val[:20].tolist()))[:80])
print("pred token:", repr(tok.decode([nxt])), "actual:", repr(tok.decode([val[20].item()])))
# 3. same ctx as prompt, generate 10 tokens greedy, compare to val continuation
with torch.no_grad():
    cur = ctx.clone()
    for i in range(10):
        lg, _ = m(cur[:, -1024:])
        cur = torch.cat([cur, lg[0,-1].argmax().view(1,1)], 1)
print("gen from val ctx:", repr(tok.decode(cur[0,20:].tolist())))
print("val actual      :", repr(tok.decode(val[20:30].tolist())))
# 4. natural-prompt logits top-5
with torch.no_grad():
    lg, _ = m(torch.tensor(ids).view(1,-1).to('cuda:3'))
    top = lg[0,-1].topk(5).indices.tolist()
print("natural-prompt top5:", [repr(tok.decode([t]))[:12] for t in top])
