import sys, time, torch, argparse
sys.path.insert(0, "/root/phi")
import pretrain_tb_fast as pt
from pretrain_tb_fast import TiedLinear
from ptied_train import enable_ptied

cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32,
                         rank=64, stages=2)
m = pt.LM(cfg).cuda()
enable_ptied(m)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.1,
                        betas=(0.9, 0.95), fused=True)
xb = torch.randint(0, 50304, (32, 2048), device="cuda")

def step():
    logits, _ = m(xb)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), xb.reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

for _ in range(6):
    step()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    step()
torch.cuda.synchronize()
wall = (time.time() - t0) / 10
print(f"wall per step: {wall*1000:.1f} ms | {8192/wall/1000:.0f} tok/s (single GPU)", flush=True)

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(6):
        step()
    torch.cuda.synchronize()
evs = prof.key_averages()
gpu_tot = sum(e.self_device_time_total for e in evs) / 6
cpu_tot = sum(e.self_cpu_time_total for e in evs) / 6
n_launch = sum(e.count for e in evs if 'LaunchKernel' in e.key) / 6
print(f"GPU busy per step: {gpu_tot/1000:.1f} ms | CPU busy: {cpu_tot/1000:.1f} ms | launches/step: {n_launch:.0f}", flush=True)
print(f"=> dispatch/other overhead: {wall - gpu_tot/1000:.1f} ms ({(wall - gpu_tot/1000)/wall*100:.0f}% of wall)", flush=True)
rows = [(e.key, e.self_device_time_total / 6, e.count / 6)
        for e in evs if e.self_device_time_total > 0]
rows.sort(key=lambda r: -r[1])
for k, ms, c in rows[:10]:
    print(f"  {k[:58]:58s} {ms/1000:7.3f} ms x{c:6.1f}", flush=True)
