import sys, torch, argparse
sys.path.insert(0, "/root/phi")
import pretrain_tb as pt
cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32,
                         rank=64, stages=2)
m = pt.LM(cfg).cuda().eval()
b0 = m.blocks[0].attn
S = 16
q_pre = (torch.randn(1, 8, S, 64) * 0.5).bfloat16().cuda()
# OUR rope
q_ours = b0.rot(q_pre)                     # [1, 8, S, 64]
# LLAMA-style rope on the permuted q (as the export assumes)
per_head = list(range(0, 64, 2)) + list(range(1, 64, 2))
q_perm = q_pre[:, :, :, per_head]
inv = 1.0 / (64 ** (torch.arange(0, 64, 2).float() / 64))
t = torch.arange(S).float()
f = torch.outer(t, inv)
cos = f.cos().bfloat16().cuda()            # [S, 32]
sin = f.sin().bfloat16().cuda()
x1 = q_perm[..., :32]; x2 = q_perm[..., 32:]
out_ll = torch.cat([x1 * cos - x2 * sin,
                    x1 * sin + x2 * cos], dim=-1)   # [1, 8, S, 64]
# unpermute out_ll back to our row order and compare
inv_perm = torch.argsort(torch.tensor(per_head))
out_back = out_ll[..., inv_perm]
d = (q_ours - out_back).abs().max().item()
print("rotary transform: max|d| =", d,
      "(rel", d / q_ours.abs().max().item(), ")", flush=True)
