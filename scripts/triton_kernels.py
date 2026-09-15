import torch, triton, triton.language as tl

# ------------------------------------------- B1: butterfly stage 1, grid=NB --
@triton.jit
def bfly_s1(x_ptr, w_ptr, y_ptr, v_ptr, t_ptr, NB: tl.constexpr,
            B: tl.constexpr, D_IN: tl.constexpr, R: tl.constexpr,
            BLOCK_DIN: tl.constexpr):
    i = tl.program_id(0)                       # block index; i==NB does t
    offs_r = tl.arange(0, B)
    offs_o = tl.arange(0, B)
    if i < NB:
        xb = tl.load(x_ptr + i * B + offs_r,
                     mask=(i * B + offs_r) < D_IN, other=0.0).to(tl.float32)
        W = tl.load(w_ptr + i * B * B + offs_o[:, None] * B
                    + offs_r[None, :])
        yb = tl.sum(W.to(tl.float32) * xb[None, :], axis=1)
        tl.store(y_ptr + i * B + offs_o, yb.to(tl.bfloat16))
    else:                                      # residual intermediate, ONCE
        offs_rr = tl.arange(0, R)
        offs_din = tl.arange(0, BLOCK_DIN)
        mask_din = offs_din < D_IN
        x = tl.load(x_ptr + offs_din, mask=mask_din, other=0.0).to(tl.float32)
        Vm = tl.load(v_ptr + offs_rr[:, None] * D_IN + offs_din[None, :],
                     mask=mask_din[None, :], other=0.0)
        t = tl.sum(Vm.to(tl.float32) * x[None, :], axis=1)         # [R]
        tl.store(t_ptr + offs_rr, t.to(tl.bfloat16))

# --------------------- B2: stage2 + blockdiag + rank-r residual, grid=NB_OUT --
@triton.jit
def bfly_s2_bd_res(y1_ptr, w2_ptr, blk_ptr, t_ptr, u_ptr, y_ptr,
                   NB: tl.constexpr, B: tl.constexpr,
                   R: tl.constexpr, D_OUT: tl.constexpr):
    i = tl.program_id(0)                       # output block index
    offs_r = tl.arange(0, B)
    offs_o = tl.arange(0, B)
    offs_rr = tl.arange(0, R)
    t = tl.load(t_ptr + offs_rr).to(tl.float32)                    # [R]
    # permuted gather: v'[L] = v[(L%NB)*B + L//NB]
    offs_lin = i * B + offs_r
    src = (offs_lin % NB) * B + offs_lin // NB
    h2 = tl.load(y1_ptr + src).to(tl.float32)                      # [B]
    W2 = tl.load(w2_ptr + i * B * B + offs_o[:, None] * B
                 + offs_r[None, :])
    s2 = tl.sum(W2.to(tl.float32) * h2[None, :], axis=1)           # [o]
    Wb = tl.load(blk_ptr + i * B * B + offs_o[:, None] * B
                 + offs_r[None, :])
    y2 = tl.sum(Wb.to(tl.float32) * s2[None, :], axis=1)           # [o]
    d_idx = i * B + offs_o
    m = d_idx < D_OUT
    Ub = tl.load(u_ptr + d_idx[:, None] * R + offs_rr[None, :],
                 mask=m[:, None], other=0.0)
    yr = tl.sum(Ub.to(tl.float32) * t[None, :], axis=1)            # [o]
    tl.store(y_ptr + d_idx, (y2 + yr).to(tl.bfloat16), mask=m)

class TritonTiedLinear:
    """12MB-RAM decode path: butterfly stages straight from the seed params
    (no Mf buffer), fused blockdiag + residual. 2 launches per instance."""
    def __init__(self, tl_mod):
        self.d_in, self.d_out = tl_mod.d_in, tl_mod.d_out
        self.n, self.b, self.nb = tl_mod.n, tl_mod.b, tl_mod.nb
        self.nb_out, self.r = tl_mod.nb_out, tl_mod.U.shape[1]
        self.w1 = tl_mod.basis.stages[0].cuda().contiguous()
        self.w2 = tl_mod.basis.stages[1].cuda().contiguous()
        self.blocks = tl_mod.blocks.cuda().contiguous()
        self.V = tl_mod.V.cuda().contiguous()
        self.t = torch.empty(self.r, dtype=torch.bfloat16, device="cuda")
        self.U = tl_mod.U.cuda().contiguous()
        self.y1 = torch.empty(self.n, dtype=torch.bfloat16, device="cuda")

    def __call__(self, x):
        B, S = x.shape[0], x.shape[1]
        xf = x.reshape(-1).contiguous()
        bfly_s1[(self.nb + 1,)](xf, self.w1, self.y1, self.V, self.t,
                                NB=self.nb, B=self.b, D_IN=self.d_in,
                                R=self.r,
                                BLOCK_DIN=triton.next_power_of_2(self.d_in),
                                num_warps=8)
        y = torch.empty(B * S * self.d_out, dtype=torch.bfloat16,
                        device="cuda")
        bfly_s2_bd_res[(B * S * self.nb_out,)](
            self.y1, self.w2, self.blocks, self.t, self.U, y,
            NB=self.nb, B=self.b, R=self.r, D_OUT=self.d_out,
            num_warps=4)
        return y.view(B, S, self.d_out)

# --------------------------------------------------------- A: P-mode GEMV ---
@triton.jit
def gemv_p(x_ptr, p_ptr, y_ptr, D_IN, D_OUT,
           BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < D_OUT
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_IN
    x = tl.load(x_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    p = tl.load(p_ptr + offs_d[:, None] * D_OUT + offs_n[None, :],
                mask=mask_d[:, None] & mask_n[None, :], other=0.0)
    acc = tl.sum(x[:, None] * p.to(tl.float32), axis=0)
    tl.store(y_ptr + offs_n, acc.to(tl.bfloat16), mask=mask_n)

def gemv_p_launch(x, P, y):
    d_in, d_out = P.shape
    grid = (triton.cdiv(d_out, 64),)
    gemv_p[grid](x, P, y, d_in, d_out,
                 BLOCK_D=triton.next_power_of_2(d_in), BLOCK_N=64,
                 num_warps=4)
    return y
