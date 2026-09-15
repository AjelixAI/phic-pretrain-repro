import sys, time, torch, argparse, os
import numpy as np
sys.path.insert(0, "/root/phi")
import pretrain_tb_fast as pt
from ptied_train import enable_ptied

# The CUDA-graph training step for the SOTA run config: the identical kernels,
# one launch. Gate: the loss curve matches eager over N steps on the same data.
cfg = argparse.Namespace(mode='tied', d=2048, layers=16, ffn=7168, block=32,
                         rank=64, stages=2, vocab=100352)
TPB = 8 * 1024
torch.manual_seed(0)

def fresh():
    m = pt.LM(cfg).cuda()
    enable_ptied(m)
    opt = torch.optim.AdamW(m.parameters(), lr=torch.tensor(3e-4, device='cuda'),
                            weight_decay=0.1, fused=True, capturable=True)
    return m, opt

data = torch.load('/root/phi/data_cache_gatedata.pt', weights_only=True, mmap=True)
nb = data.numel() // TPB
N = 30

def loss_of(m, xb):
    logits, _ = m(xb)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), xb.reshape(-1))

def run_eager():
    torch.manual_seed(0)
    m, opt = fresh()
    losses = []
    t0 = time.time()
    for i in range(N):
        off = (i % nb) * TPB
        xb = torch.as_tensor(np.asarray(data[off:off+TPB])).long().cuda().view(8, 1024)
        l = loss_of(m, xb)
        l.backward(); opt.step(); opt.zero_grad(set_to_none=False)
        losses.append(l.item())
    torch.cuda.synchronize()
    return losses, (time.time() - t0) / N

def run_graph():
    torch.manual_seed(0)
    m, opt = fresh()
    # warmup on the side stream
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    xb_s = torch.zeros(8, 1024, dtype=torch.long, device='cuda')
    with torch.cuda.stream(s):
        for _ in range(3):
            off = 0
            xb_s.copy_(torch.as_tensor(np.asarray(data[off:off+TPB])).long().cuda().view(8, 1024))
            l = loss_of(m, xb_s)
            l.backward(); opt.step(); opt.zero_grad(set_to_none=False)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        l = loss_of(m, xb_s)
        l.backward(); opt.step(); opt.zero_grad(set_to_none=False)
    losses = []
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(N):
        off = (i % nb) * TPB
        xb_s.copy_(torch.as_tensor(np.asarray(data[off:off+TPB])).long().cuda().view(8, 1024))
        g.replay()
        losses.append(l.item())
    torch.cuda.synchronize()
    return losses, (time.time() - t0) / N

# GRAPH FIRST: the capture needs fresh memory (the eager-first order aliases
# the pool - the failure mode caught at the 1B config; the standing rule)
l_g, t_g = run_graph()
torch.cuda.empty_cache()
l_e, t_e = run_eager()
diffs = [abs(a-b)/max(abs(a), 1e-6) for a, b in zip(l_e, l_g)]
print(f'eager: {t_e*1000:.0f} ms/step ({TPB/t_e/1000:.0f} tok/s) | '
      f'graph: {t_g*1000:.0f} ms/step ({TPB/t_g/1000:.0f} tok/s) | '
      f'{t_e/t_g:.2f}x', flush=True)
print(f'max relative loss diff over {N} steps: {max(diffs):.5f}', flush=True)
print('GATE:', 'PASS' if max(diffs) < 0.01 else 'FAIL', flush=True)
