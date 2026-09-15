import torch
torch.manual_seed(0)
dev = 'cuda'
M, K, N = 4096, 512, 512
a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
ref = a @ b

# the rowwise fp32 recipe: the quantize divides by the scale, scaled_mm multiplies back
sa = a.abs().amax(1, keepdim=True).clamp(min=1e-6).float() / 448
sb = b.abs().amax(0, keepdim=True).clamp(min=1e-6).float() / 448
a8 = (a / sa).clamp(-448, 448).to(torch.float8_e4m3fn)
b8 = (b / sb).clamp(-448, 448).to(torch.float8_e4m3fn).t().contiguous().t()
y = torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
err = (y.float() - ref.float()).abs().max().item() / ref.abs().max().item()
rel = (y.float() - ref.float()).abs().mean().item() / ref.float().abs().mean().item()
print(f'rowwise fp32: OK max-rel {err:.4f} mean-rel {rel:.5f}', flush=True)

# the MXFP8 1x32: the quantize-only round-trip first (isolate)
def mx_quant(t):
    M_, K_ = t.shape
    tb = t.view(M_, K_ // 32, 32).float()
    amax = tb.abs().amax(-1, keepdim=True).clamp(min=1e-6)
    e = torch.ceil(torch.log2(amax)) - 8
    e = e.clamp(-127, 127)
    scale = torch.pow(2.0, e)
    q = (tb / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(M_, K_)
    s = scale.squeeze(-1).to(torch.float8_e8m0fnu)
    return q, s

a8m, sam = mx_quant(a)
sam_f = sam.float().view(M, K // 32, 1)
rt = (a8m.view(M, K // 32, 32).float() * sam_f).view(M, K)
qrt = (rt - a.float()).abs().max().item() / a.abs().max().item()
print(f'MX quantize round-trip err: {qrt:.4f}', flush=True)
b8m, sbm = mx_quant(b)
b8m = b8m.t().contiguous().t()
y = torch._scaled_mm(a8m, b8m, scale_a=sam, scale_b=sbm, out_dtype=torch.bfloat16)
err = (y.float() - ref.float()).abs().max().item() / ref.abs().max().item()
rel = (y.float() - ref.float()).abs().mean().item() / ref.float().abs().mean().item()
print(f'MXFP8 1x32: OK max-rel {err:.4f} mean-rel {rel:.5f}', flush=True)
