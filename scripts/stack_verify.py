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
L, ctx_len = cfg.layers, 1024

def make_caches():
    return (torch.zeros(L, 2, ctx_len, 64, dtype=torch.bfloat16,
                        device="cuda"),
            torch.zeros(L, 2, ctx_len, 64, dtype=torch.bfloat16,
                        device="cuda"))

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

def decode_step(tok_b, k_cache, v_cache, linear_of):
    h = m.emb(tok_b)
    mask = torch.where(ar <= pos, 0.0, -3.0e38).view(1, 1, 1, ctx_len)
    for l, blk in enumerate(m.blocks):
        x_in = h
        xn = blk.n1(x_in)
        q = linear_of(blk.attn.q, xn).view(1, 8, 1, 64)
        k = linear_of(blk.attn.k, xn).view(1, 2, 1, 64)
        v = linear_of(blk.attn.v, xn).view(1, 2, 1, 64)
        q = rope_one(q, pos); k = rope_one(k, pos)
        k_cache[l] = k_cache[l].index_copy(1, pos, k[0])
        v_cache[l] = v_cache[l].index_copy(1, pos, v[0])
        kk = k_cache[l].unsqueeze(0).repeat_interleave(4, dim=1)
        vv = v_cache[l].unsqueeze(0).repeat_interleave(4, dim=1)
        scores = (q @ kk.transpose(-1, -2)) / (64 ** 0.5)
        scores = scores + mask
        a = torch.softmax(scores, -1).to(vv.dtype) @ vv
        a = a.transpose(1, 2).reshape(1, 1, 512)
        h = x_in + linear_of(blk.attn.o, a)
        ml = torch.nn.functional.silu(linear_of(blk.gate, blk.n2(h))) \
            * linear_of(blk.up, blk.n2(h))
        h = h + linear_of(blk.down, ml)
    return m.head(m.nf(h))

def run(mode, n_steps=30):
    if mode == "eager":
        lo = lambda tl, x: tl(x)
    elif mode == "P":
        # merge into P once, cache on the instance
        cache = {}
        def lo(tl, x):
            key = id(tl)
            if key not in cache:
                Wt = torch.zeros(tl.n2 * tl.n2, dtype=torch.bfloat16,
                                 device="cuda")
                Wt = Wt.index_copy(0, tl.w_idx.cuda(),
                                   tl.blocks[:tl.nb_out].transpose(-1, -2)
                                   .cuda().reshape(-1)).view(tl.n2, tl.n2)
                cache[key] = ((tl.Mf @ Wt)[:, :tl.d_out]
                              + tl.V.T.cuda() @ tl.U.T.cuda()).contiguous()
            return x @ cache[key]
    else:  # butterfly
        wrap = {}
        def lo(tl, x):
            key = id(tl)
            if key not in wrap:
                wrap[key] = TritonTiedLinear(tl)
            return wrap[key](x)
    kc, vc = make_caches()
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    pids = tok("The history of artificial intelligence began",
               add_special_tokens=False).input_ids
    logits_log = []
    ids = list(pids)
    for step in range(len(pids) - 1):
        pos.fill_(step); token_buf[0, 0] = pids[step]
        logits_log.append(decode_step(token_buf, kc, vc, lo).clone())
    cur = torch.tensor([[pids[-1]]], device="cuda")
    for step in range(n_steps):
        pos.fill_(len(pids) - 1 + step); token_buf[0, 0] = cur.item()
        lg = decode_step(token_buf, kc, vc, lo)
        logits_log.append(lg.clone())
        cur = lg[:, -1:].argmax(-1)
        ids.append(cur.item())
    return ids, torch.cat(logits_log, 1)     # [1, T, vocab]

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
ids_e, lg_e = run("eager")
ids_p, lg_p = run("P")
ids_b, lg_b = run("butterfly")

for name, ids, lg in [("P", ids_p, lg_p), ("butterfly", ids_b, lg_b)]:
    tm = sum(int(a == b) for a, b in zip(ids_e, ids)) / len(ids_e) * 100
    d = (lg.float() - lg_e.float()).abs()
    rel = d / lg_e.float().abs().clamp_min(1e-3)
    # top-1 agreement per position
    agree = (lg.argmax(-1) == lg_e.argmax(-1)).float().mean().item() * 100
    print(f"{name:9s}: token-match {tm:5.1f}% | top-1 agree {agree:5.1f}% | "
          f"logit max|d| {d.max().item():.3f} | med rel {rel.median().item():.4f}",
          flush=True)
print("eager text:", repr(tok.decode(ids_e)[:70]), flush=True)
print("P text:     ", repr(tok.decode(ids_p)[:70]), flush=True)
print("butterfly:  ", repr(tok.decode(ids_b)[:70]), flush=True)
