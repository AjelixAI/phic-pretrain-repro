import sys, torch, argparse, os
sys.path.insert(0, '/root/phi')
import pretrain_tb_fast as pt
from pretrain_tb_fast import save_full
from ptied_train import enable_ptied
cfg = argparse.Namespace(mode='tied', d=2048, layers=16, ffn=7168, block=32, rank=64, stages=2, vocab=100352)
torch.manual_seed(0)
m = pt.LM(cfg).cuda(); enable_ptied(m)
opt = torch.optim.AdamW(m.parameters(), lr=torch.tensor(3e-4, device='cuda'), weight_decay=0.1, fused=True, capturable=True)
xb = torch.randint(0, 100352, (4, 1024), device='cuda')
l = torch.nn.functional.cross_entropy(m(xb)[0].reshape(-1, 100352).float(), xb.reshape(-1))
l.backward(); opt.step()
before_loss = l.item(); before_w = m.emb.weight[:2, :4].clone()
save_full('/root/phi/test_ckpt.pt', m, opt, 123, argparse.Namespace(force_save=True))
m2 = pt.LM(cfg).cuda(); enable_ptied(m2)
ck = torch.load('/root/phi/test_ckpt.pt', map_location='cpu', weights_only=False)
m2.load_state_dict(ck['model'], strict=True)
opt2 = torch.optim.AdamW(m2.parameters(), lr=torch.tensor(3e-4, device='cuda'), weight_decay=0.1, fused=True, capturable=True)
opt2.load_state_dict(ck['opt'])
sw = sum(torch.equal(p.cpu(), q.cpu()) for p, q in zip(m.parameters(), m2.parameters()))
print(f'round-trip: step {ck["step"]} | full state-dict bitwise equal: {sw}/{sum(1 for _ in m.parameters())} tensors')
with torch.no_grad():
    l2 = torch.nn.functional.cross_entropy(m2(xb)[0].reshape(-1, 100352).float(), xb.reshape(-1))
print('loss identical:', abs(l2.item() - before_loss) < 1e-4, f'({l2.item():.6f} vs {before_loss:.6f})')
print('opt state entries:', len(ck['opt']['state']), '| step:', ck['step'])
os.remove('/root/phi/test_ckpt.pt')
print('ROUND-TRIP TEST PASSED' if torch.equal(m2.emb.weight[:2, :4].cpu(), before_w.cpu()) else 'ROUND-TRIP FAILED')
