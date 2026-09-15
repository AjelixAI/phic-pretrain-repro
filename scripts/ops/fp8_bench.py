import torch, time
torch.manual_seed(0)
dev = 'cuda'
M, K, N = 65536, 512, 512          # the exact training shape: x @ P

a_bf = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
b_bf = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
# FP8: e4m3 activations/weights with per-tensor scales
a8 = (torch.randn(M, K, device=dev) * 0.1).clamp(-448, 448).to(torch.float8_e4m3fn)
b8 = (torch.randn(K, N, device=dev) * 0.1).clamp(-448, 448).to(torch.float8_e4m3fn).t().contiguous().t()
sa = torch.ones((), device=dev); sb = torch.ones((), device=dev)

def bench(fn, n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n

t_bf = bench(lambda: a_bf @ b_bf)
t_fp8 = bench(lambda: torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
fl = 2 * M * K * N
print(f'bs65536: bf16 {t_bf*1e3:.2f} ms -> {fl/t_bf/1e12:.0f} TFLOPS')
print(f'bs65536: FP8  {t_fp8*1e3:.2f} ms -> {fl/t_fp8/1e12:.0f} TFLOPS')
print(f'FP8/bf16 speedup on our shape: {t_bf/t_fp8:.2f}x')
# the per-instance size the P-form actually uses at bs 8x1024
M2 = 8192
a2 = torch.randn(M2, K, device=dev, dtype=torch.bfloat16)
a28 = (torch.randn(M2, K, device=dev) * 0.1).to(torch.float8_e4m3fn)
t_bf2 = bench(lambda: a2 @ b_bf)
t_fp82 = bench(lambda: torch._scaled_mm(a28, b8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
print(f'bs8192: bf16 {t_bf2*1e3:.2f} ms -> {fl/8/t_bf2/1e12:.0f} TFLOPS')
print(f'bs8192: FP8  {t_fp82*1e3:.2f} ms -> {fl/8/t_fp82/1e12:.0f} TFLOPS')
print(f'FP8/bf16 speedup: {t_bf2/t_fp82:.2f}x')
