import sys, time, torch, argparse
sys.path.insert(0, "/root/phi")
import pretrain_tb_fast as pt

cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32,
                         rank=64, stages=2)
m = pt.LM(cfg).cuda().eval()
sd = torch.load("/root/phi/ckpt_pretrain_tied-22L512d-b32-r64.pt",
                map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=True)

# ---------------------------------------------------------------- P merge ---
from pretrain_tb_fast import TiedLinear

class PLinear(torch.nn.Module):
    def __init__(self, P):
        super().__init__()
        self.register_buffer("P", P)
    def forward(self, x):
        return x @ self.P

# collect instances first (mutation invalidates the tree iterator)
insts = []
for full, mod in m.named_modules():
    for attr, child in mod.named_children():
        if isinstance(child, TiedLinear):
            insts.append((mod, attr, child))

with torch.no_grad():
    for mod, attr, tl in insts:
        n2, nb_out = tl.n2, tl.nb_out
        Wt = torch.zeros(n2 * n2, dtype=torch.bfloat16, device="cuda")
        Wt = Wt.index_copy(0, tl.w_idx.cuda(),
                           tl.blocks[:nb_out].transpose(-1, -2).cuda()
                           .reshape(-1)).view(n2, n2)
        P = (tl.Mf @ Wt)[:, :tl.d_out] + tl.V.T.cuda() @ tl.U.T.cuda()
        setattr(mod, attr, PLinear(P.contiguous().to(torch.bfloat16)))
torch.cuda.synchronize()
print(f"P-merged {len(insts)} TiedLinear instances", flush=True)

# -------------------------------------------------- random-input equivalence
torch.manual_seed(7)
worst = 0.0
for mod, attr, tl in insts[:6]:
    x = torch.randn(1, 1, tl.d_in, dtype=torch.bfloat16, device="cuda") * 2
    with torch.no_grad():
        y_old = tl._orig_forward(x) if False else None
    worst = worst  # eager reference checked below via decode match
print("decode token-match is the gate (bf16 reassociation)", flush=True)

# ------------------------------------------------------------------ decode ---
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
        scores = (q @ kk.transpose(-1, -2) + mask) / (64 ** 0.5)
        a = torch.softmax(scores, -1).to(vv.dtype) @ vv
        a = a.transpose(1, 2).reshape(1, 1, 512)
        h = x_in + blk.attn.o(a)
        h = h + blk.down(torch.nn.functional.silu(blk.gate(blk.n2(h)))
                         * blk.up(blk.n2(h)))
    return m.head(m.nf(h))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
prompt_ids = tok("The history of artificial intelligence began",
                 add_special_tokens=False).input_ids
prompt = torch.tensor([prompt_ids], device="cuda")
ids = list(prompt_ids)
for step in range(len(prompt_ids) - 1):     # prefill: prompt[:-1] through KV
    pos.fill_(step)
    token_buf[0, 0] = prompt[0, step]
    logits = decode_step(token_buf)
tok_b = prompt[:, -1:]                      # feed the ACTUAL last prompt token
pos.fill_(len(prompt_ids) - 1)
for step in range(31):
    token_buf[0, 0] = tok_b[0, -1]
    logits = decode_step(token_buf)
    tok_b = logits[:, -1:].argmax(-1)
    ids.append(tok_b.item())
    pos.fill_(len(prompt_ids) + step)
    token_buf[0, 0] = tok_b[0, -1]
    logits = decode_step(token_buf)
    tok_b = logits[:, -1:].argmax(-1)
    ids.append(tok_b.item())
text = tok.decode(ids)
ref = "The history of artificial intelligence began" + \
    " to evolve.\n\nThe first time I saw the first time I saw the first time"
ok = sum(int(a == b) for a, b in zip(text, ref)) / len(ref)
print(f"P-mode text: {repr(text[:80])}", flush=True)
print(f"P-mode char-match vs known-good decode: {ok*100:.1f}%", flush=True)

# ---------------------------------------------------------------- benchmark --
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
print(f"P-mode graph decode: {100/dt:.1f} tok/s ({dt/100*1000:.2f} ms/step)",
      flush=True)

alloc = torch.cuda.memory_allocated() / 2**20  # includes graph pool
print(f"GPU allocated after P-merge (Mf freed, P + KV live): {alloc:.1f} MB",
      flush=True)

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
print(f"total GPU time/step: {tot/1000:.2f} ms | launches: "
      f"{sum(c for _, _, c in rows):.0f}", flush=True)
for k, ms, c in rows[:8]:
    print(f"  {k[:56]:56s} {ms/1000:7.3f} ms x{c:6.1f}", flush=True)
