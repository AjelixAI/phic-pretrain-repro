import os, sys, torch, time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
bs, no_ckpt, ckpt_every = int(sys.argv[1]), sys.argv[2] == '1', int(sys.argv[3]) if len(sys.argv)>3 else 1
from types import SimpleNamespace as NS
sys.path.insert(0, '/root/phi')
import pretrain_tb_fast as P
from pretrain_tb_fast import LM
from ptied_train import enable_ptied
DEV='cuda:3'
cfg = NS(mode='tied', d=2048, ffn=7168, layers=16, block=32, rank=64, stages=2,
         vocab=100352, seq=4096, rope_base=10000.0, chunked_ce=8192, no_ckpt=no_ckpt, ckpt_every=ckpt_every)
P.A = cfg
m = LM(cfg).to(DEV); m.train(); enable_ptied(m)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=True)
x = torch.randint(0, 100352, (bs, 4096), device=DEV)
def step():
    opt.zero_grad(set_to_none=True)
    _, loss = m(x, labels=x); loss.backward(); opt.step()
step(); torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(2): step()
torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/2
peak = torch.cuda.max_memory_allocated()/1e9
print(f"RESULT bs={bs} no_ckpt={no_ckpt} every={ckpt_every}: {dt*1000:.0f} ms/step {bs*4096/dt:,.0f} tok/s peak {peak:.1f} GB")
