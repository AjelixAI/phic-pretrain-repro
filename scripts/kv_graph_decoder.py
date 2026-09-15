"""KV-cached CUDA-graph decoder for Ajelix-Fiber (tied-blocks, 12MB).
Single-token decode steps captured as ONE CUDA graph (~2 launches/token).
Includes the intelligence-preservation check: graph-generated sequence vs
the eager full-context decoder, token by token.
"""
import sys, time, torch, argparse
sys.path.insert(0, "/root/phi")
import pretrain_tb_fast as pt
from transformers import AutoTokenizer

DEV = "cuda"
ctx_len = 1024
cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32,
                         rank=64, stages=2)
m = pt.LM(cfg).to(DEV).eval()
sd = torch.load("/root/phi/ckpt_pretrain_tied-22L512d-b32-r64.pt",
                map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=True)
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
L = cfg.layers

# ---- static state --------------------------------------------------------
k_cache = torch.zeros(L, 2, ctx_len, 64, dtype=torch.bfloat16, device=DEV)
v_cache = torch.zeros(L, 2, ctx_len, 64, dtype=torch.bfloat16, device=DEV)
pos = torch.zeros(1, dtype=torch.long, device=DEV)
token_buf = torch.zeros(1, 1, dtype=torch.long, device=DEV)
ar = torch.arange(ctx_len, device=DEV)
cos_t = m.blocks[0].attn.rot.cos[:ctx_len]     # [ctx, 32] bf16
sin_t = m.blocks[0].attn.rot.sin[:ctx_len]

def rope_one(x, p):                            # x [1, H, 1, 64]
    ct = cos_t.index_select(0, p).view(1, 1, 32)
    st = sin_t.index_select(0, p).view(1, 1, 32)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * ct - x2 * st, x1 * st + x2 * ct], -1).flatten(-2)

def decode_step(tok_id_b):                     # tok_id_b: [1,1] long
    h = m.emb(tok_id_b)                        # [1,1,512]
    mask = torch.where(ar <= pos, 0.0, -3.0e38).view(1, 1, 1, ctx_len)
    new_k = [None] * L
    new_v = [None] * L
    attn_out = [None] * L
    for l, blk in enumerate(m.blocks):
        x_in = h
        xn = blk.n1(x_in)                      # pre-norm (the model's math!)
        q = blk.attn.q(xn).view(1, 8, 1, 64)
        k = blk.attn.k(xn).view(1, 2, 1, 64)
        v = blk.attn.v(xn).view(1, 2, 1, 64)
        q = rope_one(q, pos)
        k = rope_one(k, pos)
        k_cache[l] = k_cache[l].index_copy(1, pos, k[0])
        v_cache[l] = v_cache[l].index_copy(1, pos, v[0])
        kk = k_cache[l]                        # [2, ctx, 64]
        vv = v_cache[l]
        kk = kk.unsqueeze(0).repeat_interleave(4, dim=1)   # [1, 8, ctx, 64]
        vv = vv.unsqueeze(0).repeat_interleave(4, dim=1)
        scores = (q @ kk.transpose(-1, -2)) / (64 ** 0.5)  # [1,8,1,ctx]
        scores = scores + mask
        a = torch.softmax(scores, -1).to(vv.dtype) @ vv    # [1,8,1,64]
        a = a.transpose(1, 2).reshape(1, 1, 512)
        h = x_in + blk.attn.o(a)
        h = h + blk.down(torch.nn.functional.silu(blk.gate(blk.n2(h)))
                         * blk.up(blk.n2(h)))
    logits = m.head(m.nf(h))
    return logits                            # [1,1,50304]

# ---- capture --------------------------------------------------------------
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        pos.fill_(0)
        out_static = decode_step(token_buf)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.stream(s):
    g.capture_begin()
    out_static = decode_step(token_buf)
    g.capture_end()
torch.cuda.current_stream().wait_stream(s)
print("graph captured (KV-cached decode step)", flush=True)

# ---- generation with the graph -------------------------------------------
@torch.no_grad()
def gen_graph(prompt_ids, n, use_graph=True):
    torch.cuda.synchronize()
    t0 = time.time()
    out = []
    # PREFILL: run every prompt token through the graph to fill the KV cache
    for t, tid in enumerate(prompt_ids[:-1]):
        pos.fill_(t)
        token_buf.fill_(tid)
        g.replay() if use_graph else decode_step(token_buf)
    cur = prompt_ids[-1]
    for t in range(len(prompt_ids) - 1, len(prompt_ids) - 1 + n):
        pos.fill_(t)
        token_buf.fill_(cur)
        if use_graph:
            g.replay()
            logits_out = out_static
        else:
            logits_out = decode_step(token_buf)
        cur = logits_out[0, 0].argmax().item()
        out.append(cur)
    dt = time.time() - t0
    return out, n / dt

@torch.no_grad()
def gen_eager(prompt_ids, n):
    x = torch.tensor([prompt_ids]).cuda()
    out = []
    t0 = time.time()
    for _ in range(n):
        logits, _ = m(x)
        cur = logits[0, -1].argmax().item()
        out.append(cur)
        x = torch.cat([x, torch.tensor([[cur]]).cuda()], 1)
        if x.shape[1] > 2048:
            x = x[:, -2048:]
    dt = time.time() - t0
    return out, n / dt

prompt = tok("The history of artificial intelligence began",
             add_special_tokens=False).input_ids
N = 160
e_tokens, rate_e = gen_eager(prompt, N)
g_tokens_nc, rate_nc = gen_graph(prompt, N, use_graph=False)
tok_match_nc = sum(int(a == b) for a, b in zip(g_tokens_nc, e_tokens)) / N
print(f"eager full-context baseline: {rate_e:.1f} tok/s", flush=True)
print(f"eager KV-decode: {rate_nc:.1f} tok/s | match vs full-context: "
      f"{tok_match_nc:.1%}", flush=True)
print("eager-KV text:", repr(tok.decode(g_tokens_nc)[:160]), flush=True)
g_tokens, rate_g = gen_graph(prompt, N)
tok_match_g = sum(int(a == b) for a, b in zip(g_tokens, e_tokens)) / N
print(f"graph decode: {rate_g:.1f} tok/s | match vs full-context: "
      f"{tok_match_g:.1%}", flush=True)
print("graph text:", repr(tok.decode(g_tokens)[:160]), flush=True)
print("eager text:", repr(tok.decode(e_tokens)[:160]), flush=True)
ram = (sum(p.numel() * p.element_size() for p in m.parameters()) +
       k_cache.numel() * 2 + v_cache.numel() * 2) / 1e6
print(f"weights+KV RAM: {ram:.1f} MB", flush=True)
