import sys, torch, argparse, os
sys.path.insert(0, "/root/phi")
import pretrain_tb_fast as pt
from ptied_train import enable_ptied
import numpy as np

cfg = argparse.Namespace(mode='tied', d=2048, layers=16, ffn=7168, block=32,
                         rank=64, stages=2, vocab=100352)
torch.manual_seed(0)
m = pt.LM(cfg).cuda(); enable_ptied(m)
opt = torch.optim.AdamW(m.parameters(), lr=torch.tensor(3e-4, device='cuda'),
                        weight_decay=0.1, fused=True, capturable=True)
data = torch.load('/root/phi/data_cache_gatedata.pt', weights_only=True, mmap=True)
TPB = 8 * 1024
nb = data.numel() // TPB
xb = torch.as_tensor(np.asarray(data[:TPB])).long().cuda().view(8, 1024)

def loss_fp8(on):
    os.environ['FP8_GEMM'] = '1' if on else '0'
    logits, _ = m(xb)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), xb.reshape(-1)).item()

# train 400 steps bf16 (the model state where we compare)
for i in range(400):
    off = (i % nb) * TPB
    b = torch.as_tensor(np.asarray(data[off:off + TPB])).long().cuda().view(8, 1024)
    l = torch.nn.functional.cross_entropy(
        m(b)[0].reshape(-1, 100352).float(), b.reshape(-1))
    l.backward(); opt.step(); opt.zero_grad(set_to_none=False)
    if i % 200 == 0: print(f'train step {i} loss {l.item():.4f}', flush=True)

# the paired forward-only comparison: the same batches, both precisions
print('=== the paired forward-only comparison (the same batches, the same weights) ===', flush=True)
diffs, refs = [], []
for k in range(6):
    off = ((400 + k * 97) % nb) * TPB
    xb = torch.as_tensor(np.asarray(data[off:off + TPB])).long().cuda().view(8, 1024)
    lb = loss_fp8(False); lf = loss_fp8(True)
    diffs.append(lf - lb); refs.append(lb)
    print(f'  batch {k}: bf16 {lb:.4f} | fp8 {lf:.4f} | diff {lf-lb:+.4f} ({(lf-lb)/lb*100:+.2f}%)', flush=True)
import statistics
mean_ref = statistics.mean(refs); mean_d = statistics.mean(diffs)
print(f'=== mean loss: bf16 {mean_ref:.4f} | fp8 {mean_ref+mean_d:.4f} ===', flush=True)
print(f'=== FP8 quantization effect on loss: {mean_d:+.4f} nats ({mean_d/mean_ref*100:+.2f}%) ===', flush=True)
print('GATE:', 'PASS' if abs(mean_d) < 0.05 else 'MARGINAL' if abs(mean_d) < 0.15 else 'FAIL', flush=True)
