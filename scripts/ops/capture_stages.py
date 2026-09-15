import sys, torch, argparse
sys.path.insert(0, "/root/phi")
import pretrain_tb_fast as pt
from ptied_train import enable_ptied
cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32, rank=64, stages=2)
torch.manual_seed(0)
m = pt.LM(cfg).cuda(); enable_ptied(m)
opt = torch.optim.AdamW(m.parameters(), lr=torch.tensor(3e-4, device='cuda'),
                        weight_decay=0.1, betas=(0.9, 0.95), fused=True, capturable=True)
xb = torch.randint(0, 50304, (8, 1024), device='cuda')

def fwd_bwd():
    logits, _ = m(xb)
    l = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), xb.reshape(-1))
    l.backward()
    return l

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        l = fwd_bwd(); opt.step(); opt.zero_grad(set_to_none=False)
torch.cuda.current_stream().wait_stream(s)

for stage, fn in [('forward', lambda: m(xb)),
                  ('fwd+bwd', lambda: fwd_bwd()),
                  ('fwd+bwd+opt', lambda: (fwd_bwd(), opt.step(), opt.zero_grad(set_to_none=False)))]:
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay(); torch.cuda.synchronize()
        print(f'{stage}: capture OK', flush=True)
    except Exception as e:
        print(f'{stage}: FAILED {type(e).__name__}: {str(e)[:130]}', flush=True)
        break
