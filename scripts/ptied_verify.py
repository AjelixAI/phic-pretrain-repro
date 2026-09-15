import sys, time, types, torch, argparse
sys.path.insert(0, "/root/phi")
sys.path.insert(0, "/tmp")
import pretrain_tb_fast as pt
from pretrain_tb_fast import TiedLinear
from ptied_train import enable_ptied, disable_ptied

torch.manual_seed(3)
cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32,
                         rank=64, stages=2)
m = pt.LM(cfg).cuda()
sd = torch.load("/root/phi/ckpt_pretrain_tied-22L512d-b32-r64.pt",
                map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=True)

# --------------------------------------------------- gradient equivalence ---
print("== gradient check: P-form vs eager fast ==", flush=True)
fams = [("q", "attn"), ("k", "attn"), ("gate", None), ("down", None)]
worst = 0.0
for name, owner in fams:
    blk = m.blocks[0]
    tl = getattr(blk.attn if owner else blk, name)
    x = (torch.randn(2, 64, tl.d_in, dtype=torch.bfloat16, device="cuda")
         * 0.5)
    # eager
    for p in m.parameters():
        p.grad = None
    with torch.enable_grad():
        y_e = tl(x)
        g = torch.randn_like(y_e)
        y_e.backward(g)
    ge = {n: p.grad.clone() for n, p in
          [("blocks", tl.blocks), ("U", tl.U), ("V", tl.V)]}
    # P-form
    for p in m.parameters():
        p.grad = None
    from ptied_train import ptied_forward
    ptied = types.MethodType(ptied_forward, tl)
    with torch.enable_grad():
        y_p = ptied(x)
        y_p.backward(g)
    gp = {n: p.grad.clone() for n, p in
          [("blocks", tl.blocks), ("U", tl.U), ("V", tl.V)]}
    for n in ge:
        d = (ge[n].float() - gp[n].float()).abs().max().item()
        s = ge[n].float().abs().max().item()
        rel = d / max(s, 1e-6)
        worst = max(worst, rel)
        print(f"  blk0.{name:5s} dL/d{n:6s}: max|d|={d:.4f} "
              f"|g|={s:.4f} rel={rel:.4f}", flush=True)
print(f"  WORST grad rel: {worst:.4f}", flush=True)

# ------------------------------------------------------- loss-curve check ---
print("== 100-step loss curves (fixed seed/data) ==", flush=True)
torch.manual_seed(42)
data = torch.randint(0, 50304, (64, 256), device="cuda")
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)

def run_curve(tag, steps=50):
    opt.zero_grad(set_to_none=True)
    losses = []
    for i in range(steps):
        xb = data[(i * 8) % 56 : (i * 8) % 56 + 8]
        logits, _ = m(xb[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            xb[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i % 10 == 0 or i == steps - 1:
            losses.append(loss.item())
    print(f"  {tag:8s}: " + " ".join(f"{l:.3f}" for l in losses), flush=True)
    return losses

# reseed params + optimizer for both runs
import copy
sd0 = copy.deepcopy(m.state_dict())
l_eager = run_curve("eager")
m.load_state_dict(sd0)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
enable_ptied(m)
l_p = run_curve("P-form")
dfinal = abs(l_eager[-1] - l_p[-1])
print(f"  final-loss |d|: {dfinal:.4f} (eager {l_eager[-1]:.3f} vs "
      f"P {l_p[-1]:.3f})", flush=True)

# ------------------------------------------------------- step-time bench ----
print("== training step throughput (8192 tok/step) ==", flush=True)
m.load_state_dict(sd0)
disable_ptied(m)
xb = torch.randint(0, 50304, (8, 1024), device="cuda")

def step_time(tag, n=20):
    for _ in range(5):
        logits, _ = m(xb)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), xb.reshape(-1))
        loss.backward()
        m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        logits, _ = m(xb)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), xb.reshape(-1))
        loss.backward()
        m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"  {tag:8s}: {dt*1000:6.1f} ms/step = {8192/dt:8.0f} tok/s",
          flush=True)
    return dt

step_time("eager")
enable_ptied(m)
step_time("P-form")
print(f"  GPU mem (P-form): {torch.cuda.memory_allocated()/2**30:.2f} GB",
      flush=True)
