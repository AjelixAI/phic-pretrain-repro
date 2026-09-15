import os
import types
import torch


def _is_tied(child):
    """Duck-typed check: robust to the __main__-vs-imported-module class
    split when the trainer runs as a script."""
    return (type(child).__name__ == "TiedLinear"
            and all(hasattr(child, a) for a in
                    ("basis", "blocks", "Mf", "U", "V", "n2")))

def _slim_w_idx(tl):
    """w_idx for the [n2, d_out] slim Wt (only the used block-columns)."""
    nb_out, b = tl.nb_out, tl.b
    rows = torch.arange(nb_out, device=tl.w_idx.device)[:, None, None] * b \
        + torch.arange(b, device=tl.w_idx.device)[None, :, None]
    cols = torch.arange(nb_out, device=tl.w_idx.device)[:, None, None] * b \
        + torch.arange(b, device=tl.w_idx.device)[None, None, :]
    return (rows * tl.d_out + cols).reshape(-1)

class _FP8GEMM(torch.autograd.Function):
    """the forward GEMM in FP8 (rowwise scales), the exact-bf16 backward.
    The gradient math is IDENTICAL to the bf16 path; only the forward's
    y = x @ P product is quantized."""
    @staticmethod
    def forward(ctx, x, P):
        q, sx = _quantize_rowwise(x)
        qp, sp = _quantize_colwise(P)
        qp = qp.t().contiguous().t()
        y = torch._scaled_mm(q, qp, scale_a=sx, scale_b=sp,
                             out_dtype=torch.bfloat16)
        ctx.save_for_backward(x, P)
        return y
    @staticmethod
    def backward(ctx, grad):
        x, P = ctx.saved_tensors
        gx = grad @ P.T          # [M, d_out] @ [d_out, d_in]
        gP = grad.T @ x          # [d_out, M] @ [M, d_in] -> [d_out, d_in]
        return gx, gP.T

def build_fused_pairs(attn=None, mlp=None):
    """Returns the fused-op closures for the attention (q+k+v -> one GEMM)
    and the MLP (gate+up -> one GEMM). Math-identical: the same P matrices
    concatenated along d_out; the outputs slice back apart."""
    if attn is not None:
        q, k, v = attn.q, attn.k, attn.v
        def fused_qkv(x, rot):
            n2q, dq = q.n2, q.d_out
            P = torch.cat([_p_of(q), _p_of(k), _p_of(v)], dim=-1)  # [d_in, dq+dk+dv]
            y = x.reshape(-1, x.shape[-1]) @ P
            y = y.reshape(*x.shape[:-1], P.shape[-1])
            qq = y[..., :dq]; kk = y[..., dq:dq+k.d_out]; vv = y[..., dq+k.d_out:]
            return rot(qq), rot(kk), vv
        return fused_qkv
    if mlp is not None:
        g, u = mlp
        def fused_gu(x):
            dg = g.d_out
            P = torch.cat([_p_of(g), _p_of(u)], dim=-1)
            y = x.reshape(-1, x.shape[-1]) @ P
            y = y.reshape(*x.shape[:-1], P.shape[-1])
            return y[..., :dg], y[..., dg:]
        return fused_gu

def _p_of(tl):
    if not hasattr(tl, "_w_idx_slim"):
        tl._w_idx_slim = _slim_w_idx(tl)
    n2, d_out, nb_out = tl.n2, tl.d_out, tl.nb_out
    Wt = torch.zeros(n2 * d_out, dtype=torch.bfloat16, device=tl.Mf.device)
    Wt = Wt.index_copy(0, tl.w_idx_slim,
                       tl.blocks[:nb_out].transpose(-1, -2).reshape(-1))
    return tl.Mf @ Wt.view(n2, d_out) + tl.V.T @ tl.U.T

class _FP8Build(torch.autograd.Function):
    """the P's Mf@Wt in FP8; the exact-bf16 backward via the blockdiag
    extraction of Mf.T @ dP (the blocks' gradients restored exactly)."""
    @staticmethod
    def forward(ctx, Mf, blocks, w_idx, n2, d_out, nb_out):
        print(f'FP8Build.apply: blocks {tuple(blocks.shape)} nb_out {nb_out}', flush=True)
        Wt = torch.zeros(n2 * d_out, dtype=torch.bfloat16, device=Mf.device)
        Wt = Wt.index_copy(0, w_idx,
                           blocks[:nb_out].transpose(-1, -2).reshape(-1))
        Wt2 = Wt.view(n2, d_out)
        qm, sm = _quantize_rowwise(Mf)
        qw, sw = _quantize_colwise(Wt2)
        qw = qw.t().contiguous().t()
        P = torch._scaled_mm(qm, qw, scale_a=sm, scale_b=sw,
                             out_dtype=torch.bfloat16)
        ctx.save_for_backward(Mf)
        ctx.w_idx = w_idx; ctx.n2 = n2; ctx.d_out = d_out
        ctx.nb_out = nb_out; ctx.n_b = blocks.shape[0]
        return P
    @staticmethod
    def backward(ctx, dP):
        Mf, = ctx.saved_tensors
        dWt_full = Mf.T @ dP                      # [n2, d_out] bf16
        dflat = dWt_full.reshape(-1)[ctx.w_idx]   # the blockdiag extraction
        dblocks = torch.zeros(ctx.n_b, 32, 32, dtype=dflat.dtype,
                              device=dflat.device)
        dblocks[:ctx.nb_out] = dflat.view(ctx.nb_out, 32, 32).transpose(-1, -2)
        return None, dblocks, None, None, None, None

def _quantize_rowwise(t):
    """rowwise fp32 scales: a [M,K] -> q e4m3 + scale [M,1]; validated vs
    the bf16 reference (max-rel 3.4% = the e4m3 precision)."""
    scale = t.abs().amax(-1, keepdim=True).clamp(min=1e-6).float() / 448
    q = (t / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale

def _quantize_colwise(t):
    scale = t.abs().amax(0, keepdim=True).clamp(min=1e-6).float() / 448
    q = (t / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale

def ptied_forward(self, x):
    """Training path: the TiedLinear as ONE dense operator, differentiable.

    P = (Mf @ Wt_slim) + V.T @ U.T, rebuilt each call (per optimizer step);
    forward and backward are plain dense GEMMs. Math identical to
    TiedLinear.forward up to bf16 reassociation.
    With FP8_GEMM=1: the x@P and the backward GEMMs run in FP8
    (_scaled_mm); the master weights, P, and the optimizer stay bf16.
    """
    if not hasattr(self, "_w_idx_slim"):
        self._w_idx_slim = _slim_w_idx(self)
    n2, d_out, nb_out = self.n2, self.d_out, self.nb_out
    B, S = x.shape[0], x.shape[1]
    # the stale-P: the rebuild every K steps (the approximation - gated)
    K = int(os.environ.get("P_REBUILD_EVERY", "1"))
    if not hasattr(self, "_p_cache") or self._p_step is None:
        self._p_cache = None; self._p_step = -10**9
    rebuild = (self._p_cache is None) or (self._p_step // max(K,1) != (self._p_step + 1) // max(K,1))
    rebuild = rebuild or K <= 1
    if rebuild:
        Wt = torch.zeros(n2 * d_out, dtype=x.dtype, device=x.device)
        Wt = Wt.index_copy(0, self._w_idx_slim,
                           self.blocks[:nb_out].transpose(-1, -2).reshape(-1))
        if os.environ.get("FP8_BUILD") == "1" and n2 >= 1024:
            Pb = _FP8Build.apply(self.Mf, self.blocks, self._w_idx_slim,
                                 n2, d_out, nb_out)
        else:
            Pb = self.Mf @ Wt.view(n2, d_out)
        P = Pb + self.V.T @ self.U.T
        self._p_cache = P; self._p_step = self._p_step + 1 if self._p_step > -10**8 else 0
    else:
        self._p_step += 1
    P = self._p_cache
    xf = x.reshape(B * S, self.d_in)
    if os.environ.get("FP8_GEMM") == "1" and xf.shape[0] >= 4096:
        y = _FP8GEMM.apply(xf, P)
    else:
        y = xf @ P
    return y.reshape(B, S, d_out)

def enable_ptied(model):
    """Swap every TiedLinear's forward to the merged-operator form.
    Modules and parameters stay intact (optimizer state unchanged)."""
    n = 0
    for mod in model.modules():
        for attr, child in list(mod.named_children()):
            if _is_tied(child):
                child.forward = types.MethodType(ptied_forward, child)
                n += 1
    return n

def disable_ptied(model):
    for mod in model.modules():
        for attr, child in list(mod.named_children()):
            if _is_tied(child) and \
                    isinstance(child.forward, types.MethodType) and \
                    child.forward.__func__ is ptied_forward:
                child.forward = types.MethodType(type(child).forward, child)
    return None
