import os, torch, time
os.environ['FP8_BUILD'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from types import SimpleNamespace as NS
import sys; sys.path.insert(0, '/root/phi')
import pretrain_tb_fast as P
from pretrain_tb_fast import LM
from ptied_train import enable_ptied, _FP8Build
DEV='cuda:3'
cfg = NS(mode='tied', d=2048, ffn=7168, layers=16, block=32, rank=64, stages=2,
         vocab=100352, seq=4096, rope_base=10000.0, chunked_ce=8192)
m = LM(cfg).to(DEV); m.train()
n_swapped = enable_ptied(m)
print(f"ptied forwards swapped: {n_swapped} (matches running run)")
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=True)
x = torch.randint(0, 100352, (12, 4096), device=DEV)
def sync(): torch.cuda.synchronize()
def t(fn, n=3, warm=1):
    for _ in range(warm): fn(); sync()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    sync()
    return (time.perf_counter()-t0)/n*1000

def full_step():
    opt.zero_grad(set_to_none=True)
    _, loss = m(x, labels=x)
    loss.backward()
    opt.step()
ms_total = t(full_step)
print(f"FULL step (fwd+bwd+opt, FP8-build ON): {ms_total:.0f} ms -> {12*4096/(ms_total/1000):,.0f} tok/s 1GPU (contended)")

def fwd_only():
    with torch.no_grad(): m(x)
ms_fwd = t(fwd_only)
print(f"forward incl. FP8 P-rebuild:           {ms_fwd:.0f} ms ({ms_fwd/ms_total*100:.0f}%)")

# P-rebuild alone, correct signature
mods = [mm for b in m.blocks for mm in [b.attn.q,b.attn.k,b.attn.v,b.attn.o,b.gate,b.up,b.down]]
def rebuild_all():
    for mod in mods:
        _FP8Build.apply(mod.Mf, mod.blocks, mod.w_idx, mod.n2, mod.d_out, mod.nb_out)
ms_reb = t(rebuild_all)
print(f"P-rebuild {len(mods)} instances (FP8):  {ms_reb:.0f} ms ({ms_reb/ms_total*100:.0f}% of full step)")

def fwd_norebuild():
    with torch.no_grad():
        for mod in mods:
            h = (12*4096, mod.d_in)
            pass
        m(x)
os.environ['FP8_BUILD']='0'
ms_fwd0 = t(fwd_only)
print(f"forward with FP8 OFF:                  {ms_fwd0:.0f} ms -> rebuild overhead ~{ms_fwd-ms_fwd0:.0f} ms")
os.environ['FP8_BUILD']='1'

def fwd_bwd():
    opt.zero_grad(set_to_none=True)
    _, loss = m(x, labels=x); loss.backward()
ms_fb = t(fwd_bwd)
print(f"forward+backward (no opt):             {ms_fb:.0f} ms -> optimizer = {ms_total-ms_fb:.0f} ms ({(ms_total-ms_fb)/ms_total*100:.0f}%)")

# GEMM share inside forward: time a pure x@P GEMM of the same shapes
h = torch.randn(49152, 2048, device=DEV, dtype=torch.bfloat16)
Pq = torch.randn(2048, 2048, device=DEV, dtype=torch.bfloat16)
def gemm():
    h @ Pq
ms_gemm = t(gemm, n=20)
print(f"one x@P GEMM [49152x2048]@[2048x2048]: {ms_gemm:.2f} ms; 28 such GEMMs/step ~= {28*ms_gemm:.0f} ms")
print("NOTE: contended with live training on GPU3; relative shares indicative.")
