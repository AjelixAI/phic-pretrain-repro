import sys, time, torch, argparse
sys.path.insert(0, "/root/phi")
sys.path.insert(0, "/tmp")
import pretrain_tb_fast as pt
from pretrain_tb_fast import TiedLinear
from triton_kernels import TritonTiedLinear
from transformers import AutoTokenizer

cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32,
                         rank=64, stages=2)
m = pt.LM(cfg).cuda().eval()
sd = torch.load("/root/phi/ckpt_pretrain_tied-22L512d-b32-r64.pt",
                map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=True)

insts = []
for mod in list(m.modules()):
    for attr in list(mod._modules.keys()):
        if isinstance(mod._modules[attr], TiedLinear):
            insts.append((mod, attr))
for mod, attr in insts:
    t = mod._modules[attr]
    del mod._modules[attr]
    setattr(mod, attr, TritonTiedLinear(t))
del insts
torch.cuda.empty_cache()
torch.cuda.synchronize()
print("wrapped 154 instances with TritonTiedLinear", flush=True)
params_mb = sum(p.numel() * p.element_size() for p in m.parameters()) / 2**20
buf_mb = sum(b.numel() * b.element_size() for b in m.buffers()) / 2**20
print(f"model params: {params_mb:.1f} MB | persistent buffers: {buf_mb:.2f} MB",
      flush=True)

L = cfg.layers
ctx_len = 1024
k_cache = torch.zeros(L, 2, ctx_len, 64, dtype=torch.bfloat16, device="cuda")
v_cache = torch.zeros(L, 2, ctx_len, 64, dtype=torch.bfloat16, device="cuda")
pos = torch.zeros(1, dtype=torch.long, device="cuda")
token_buf = torch.zeros(1, 1, dtype=torch.long, device="cuda")
ar = torch.arange(ctx_len, device="cuda")
cos_t = m.blocks[0].attn.rot.cos[:ctx_len]
sin_t = m.blocks[0].attn.rot.sin[:ctx_len]

def rope_one(x, p):
    ct = cos_t.index_select(0, p).view(1, 1, 32)
    st = sin_t.index_select(0, p).view(1, 1, 32)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * ct - x2 * st, x1 * st + x2 * ct], -1).flatten(-2)

def decode_step(tok_b):
    h = m.emb(tok_b)
    mask = torch.where(ar <= pos, 0.0, -3.0e38).view(1, 1, 1, ctx_len)
    for l, blk in enumerate(m.blocks):
        x_in = h
        xn = blk.n1(x_in)
        q = blk.attn.q(xn).view(1, 8, 1, 64)
        k = blk.attn.k(xn).view(1, 2, 1, 64)
        v = blk.attn.v(xn).view(1, 2, 1, 64)
        q = rope_one(q, pos); k = rope_one(k, pos)
        k_cache[l] = k_cache[l].index_copy(1, pos, k[0])
        v_cache[l] = v_cache[l].index_copy(1, pos, v[0])
        kk = k_cache[l].unsqueeze(0).repeat_interleave(4, dim=1)
        vv = v_cache[l].unsqueeze(0).repeat_interleave(4, dim=1)
        scores = (q @ kk.transpose(-1, -2)) / (64 ** 0.5)
        scores = scores + mask
        a = torch.softmax(scores, -1).to(vv.dtype) @ vv
        a = a.transpose(1, 2).reshape(1, 1, 512)
        h = x_in + blk.attn.o(a)
        h = h + blk.down(torch.nn.functional.silu(blk.gate(blk.n2(h)))
                         * blk.up(blk.n2(h)))
    return m.head(m.nf(h))

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
prompt_ids = tok("The history of artificial intelligence began",
                 add_special_tokens=False).input_ids
prompt = torch.tensor([prompt_ids], device="cuda")
ids = list(prompt_ids)
for step in range(len(prompt_ids) - 1):
    pos.fill_(step)
    token_buf[0, 0] = prompt[0, step]
    logits = decode_step(token_buf)
tok_b = prompt[:, -1:]
pos.fill_(len(prompt_ids) - 1)
for step in range(31):
    token_buf[0, 0] = tok_b[0, -1]
    logits = decode_step(token_buf)
    tok_b = logits[:, -1:].argmax(-1)
    ids.append(tok_b.item())
    pos.fill_(len(prompt_ids) + step)
text = tok.decode(ids)
ref = ("The history of artificial intelligence began"
       " to evolve.\n\nThe first time I saw the first time I saw the first")
ok = sum(int(a == b) for a, b in zip(text, ref)) / len(ref)
print(f"butterfly text: {repr(text[:80])}", flush=True)
print(f"char-match vs known-good prefix: {ok*100:.1f}%", flush=True)

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        pos.fill_(0)
        decode_step(token_buf)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.stream(s):
    g.capture_begin(); decode_step(token_buf); g.capture_end()
torch.cuda.current_stream().wait_stream(s)
for _ in range(10):
    pos.fill_(5); g.replay()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    g.replay()
torch.cuda.synchronize()
dt = time.time() - t0
print(f"butterfly graph decode: {100/dt:.1f} tok/s ({dt/100*1000:.2f} ms/step)",
      flush=True)
print(f"GPU allocated (graph pool incl.): "
      f"{torch.cuda.memory_allocated()/2**20:.1f} MB", flush=True)

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
evs = prof.key_averages()
rows = [(e.key, e.self_device_time_total / 20, e.count / 20)
        for e in evs if e.self_device_time_total > 0]
rows.sort(key=lambda r: -r[1])
tot = sum(r[1] for r in rows)
print(f"GPU time/step: {tot/1000:.2f} ms | launches: "
      f"{sum(c for _, _, c in rows):.0f}", flush=True)
for k, ms, c in rows[:8]:
    print(f"  {k[:52]:52s} {ms/1000:7.3f} ms x{c:6.1f}", flush=True)
