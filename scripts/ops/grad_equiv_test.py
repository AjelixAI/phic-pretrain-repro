import os, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from types import SimpleNamespace as NS
import sys; sys.path.insert(0, '/root/phi')
import pretrain_tb_fast as P
from pretrain_tb_fast import LM
from ptied_train import enable_ptied
DEV='cuda:3'
torch.manual_seed(123)
def grads(ckpt_every):
    torch.manual_seed(42)
    cfg = NS(mode='tied', d=2048, ffn=7168, layers=16, block=32, rank=64, stages=2,
             vocab=100352, seq=4096, rope_base=10000.0, chunked_ce=8192, ckpt_every=ckpt_every)
    P.A = cfg
    m = LM(cfg).to(DEV); m.train(); enable_ptied(m)
    x = torch.randint(0, 100352, (2, 512), device=DEV)
    _, loss = m(x, labels=x)
    loss.backward()
    gs = [p.grad.detach().clone() for p in m.parameters() if p.grad is not None]
    del m; torch.cuda.empty_cache()
    return gs, loss.item()
g1, l1 = grads(1)     # checkpoint every layer (current run's math)
g4, l4 = grads(4)     # checkpoint every 4th layer
gN, lN = grads(64)    # effectively none (16 layers -> none checkpointed)
g1b, l1b = grads(1)   # SAME config twice: pure run-to-run nondeterminism floor
print(f"loss: ckpt-all {l1:.6f} | every-4 {l4:.6f} | none {lN:.6f}")
import itertools
for name, (ga, gb) in [("all vs all (rerun)", (g1b, g1)), ("every-4 vs all", (g4, g1)), ("none vs all", (gN, g1))]:
    diffs = [(a-b).abs().max().item() for a, b in zip(ga, gb)]
    rel = [(a-b).abs().max().item() / (a.abs().max().item()+1e-12) for a, b in zip(ga, gb)]
    print(f"{name}: max-abs grad diff {max(diffs):.3e}, max-relative {max(rel):.3e}, n={len(ga)} tensors")
