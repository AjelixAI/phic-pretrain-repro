import torch
torch.manual_seed(0)
dev = 'cuda'
M, K, N = 4096, 512, 512
a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
ref = a @ b

# attempt 1: rowwise fp32 scales (a: [M,1], b: [1,N])
try:
    a8 = (a / sa).clamp(-448, 448).to(torch.float8_e4m3fn)
    b8 = (b / sb).clamp(-448, 448).to(torch.float8_e4m3fn).t().contiguous().t()
    sa = a.abs().amax(1, keepdim=True).clamp(min=1e-6).float() / 448
    # the quantize divides by the scale; scaled_mm multiplies it back
    sb = b.abs().amax(0, keepdim=True).clamp(min=1e-6).float() / 448
    y = torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    err = (y.float() - ref.float()).abs().max().item() / ref.abs().max().item()
    print(f'rowwise fp32: OK, rel err {err:.4f}', flush=True)
except Exception as e:
    print(f'rowwise fp32: FAILED {str(e)[:100]}', flush=True)

# attempt 2: MXFP8 1x32 blockwise, e8m0 scales
try:
    def mx_quant(t):
        M_, K_ = t.shape
        tb = t.view(M_, K_ // 32, 32).float()
        amax = tb.abs().amax(-1, keepdim=True).clamp(min=1e-6)
        e = torch.ceil(torch.log2(amax)) - 8
        e = e.clamp(-127, 127)
        scale = torch.pow(2.0, e)
        q = (tb / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        q = q.view(M_, K_)
        s = scale.squeeze(-1).to(torch.float8_e8m0fnu)
        return q, s
    a8, sa = mx_quant(a)
    b8, sb = mx_quant(b)
    b8 = b8.t().contiguous().t()
    y = torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    err = (y.float() - ref.float()).abs().max().item() / ref.abs().max().item()
    print(f'MXFP8 1x32: OK, rel err {err:.4f}', flush=True)
except Exception as e:
    print(f'MXFP8 1x32: FAILED {str(e)[:130]}', flush=True)
