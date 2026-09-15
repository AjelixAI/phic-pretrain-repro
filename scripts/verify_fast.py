import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/tmp/bench")
import importlib.util
spec = importlib.util.spec_from_file_location("pt_old", "/tmp/bench/pretrain_tb.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
spec2 = importlib.util.spec_from_file_location("pt_new", "/tmp/bench/pretrain_tb_fast.py")
new = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(new)

class C:  # minimal cfg
    block, stages, mode, rank = 32, 2, "tied", 64

torch.manual_seed(0)
SHAPES = [(512, 512), (512, 1792), (1792, 512), (512, 128), (128, 512)]
B, S = 2, 96
for (di, doo) in SHAPES:
    o = old.TiedLinear(di, doo, C, 7).cuda()
    f = new.TiedLinear(di, doo, C, 7).cuda()
    # randomize blocks: identity init is symmetric and hides transposition bugs
    torch.manual_seed(123)
    o.blocks.data = torch.randn_like(o.blocks.data) * 0.3
    f.blocks.data.copy_(o.blocks.data)
    f.U.data.copy_(o.U.data); f.V.data.copy_(o.V.data)
    x = (torch.randn(B, S, di) * 0.5).bfloat16().cuda().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    yo = o(x); yf = f(x2)
    d_fwd = (yo - yf).abs().max().item()
    rel = d_fwd / yo.abs().max().item()
    go = yo.sum().backward(retain_graph=True) or None
    gf = yf.sum().backward() or None
    gblocks = (o.blocks.grad - f.blocks.grad).abs().max().item()
    gU = (o.U.grad - f.U.grad).abs().max().item()
    gV = (o.V.grad - f.V.grad).abs().max().item()
    gx = (x.grad - x2.grad).abs().max().item()
    scale = max(yo.abs().max().item(), 1e-6)
    nb_n = o.blocks.grad.abs().max().item() + 1e-9
    xn = x.grad.abs().max().item() + 1e-9
    print(f"({di}->{doo}): fwd rel {rel:.1e} | rel grads: blocks "
          f"{gblocks/nb_n:.1e} U {gU:.1e} V {gV:.1e} x {gx/xn:.1e}", flush=True)
print("VERIFY-DONE")
