import os, torch, time, gc
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from types import SimpleNamespace as NS
import sys; sys.path.insert(0, '/root/phi')
import pretrain_tb_fast as P
from pretrain_tb_fast import LM
from ptied_train import enable_ptied
DEV='cuda:3'
print("lever sweep: activation checkpointing x batch size (contended w/ live run)")
for bs, no_ckpt in [(12, False), (12, True), (24, True), (36, True)]:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = NS(mode='tied', d=2048, ffn=7168, layers=16, block=32, rank=64, stages=2,
             vocab=100352, seq=4096, rope_base=10000.0, chunked_ce=8192, no_ckpt=no_ckpt)
    P.A = cfg  # Block reads global A for no_ckpt
    m = LM(cfg).to(DEV); m.train()
    enable_ptied(m)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=True)
    x = torch.randint(0, 100352, (bs, 4096), device=DEV)
    def step():
        opt.zero_grad(set_to_none=True)
        _, loss = m(x, labels=x)
        loss.backward()
        opt.step()
    for _ in range(1): step(); torch.cuda.synchronize()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(2): step()
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/2
    peak = max(torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated())/1e9
    toks = bs*4096/dt
    label = f"bs={bs:>2} {'no-ckpt' if no_ckpt else 'ckpt   '}"
    print(f"{label}: {dt*1000:>7.0f} ms/step  {toks:>8,.0f} tok/s  peak {peak:5.1f} GB")
    del m, opt, x
    try:
        gc.collect(); torch.cuda.empty_cache()
    except Exception: pass
