import sys, torch, argparse
sys.path.insert(0, "/root/phi")
import pretrain_tb_fast as pt
from ptied_train import enable_ptied
cfg = argparse.Namespace(mode='tied', d=2048, layers=16, ffn=7168, block=32, rank=64, stages=2)
torch.manual_seed(0)
m = pt.LM(cfg).cuda(); enable_ptied(m)
print('actual params:', sum(p.numel() for p in m.parameters())/1e6, 'M', flush=True)
xb = torch.randint(0, 50304, (4, 1024), device='cuda')
logits, _ = m(xb)
loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), xb.reshape(-1))
loss.backward()
print('FP8 forward+backward OK, loss:', loss.item(), flush=True)
gn = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
print('grad norm proxy:', gn, flush=True)
