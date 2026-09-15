import sys, torch, argparse
sys.path.insert(0, "/root/phi")
import pretrain_tb_fast as pt

# candidate SOTA-run config: dense-equiv ~1B
cfg = argparse.Namespace(mode="tied", d=2048, layers=16, ffn=7168, block=32,
                         rank=64, stages=2)
torch.manual_seed(0)
m = pt.LM(cfg)
n_actual = sum(p.numel() for p in m.parameters())
# dense equivalent: the same shapes with each TiedLinear as a dense matrix
n_dense = 0
n_dense += m.emb.weight.numel()          # the tied emb/head counted once
n_dense += sum(p.numel() for p in [m.nf.weight])
for blk in m.blocks:
    for mod in [blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.o,
                blk.gate, blk.up, blk.down]:
        n_dense += mod.d_in * mod.d_out  # the dense weight
        # the q/k/o biases? none in our arch
n_dense += sum(p.numel() for p in sum([list(b.attn.rot.parameters()) for b in m.blocks], [])) if False else 0
flops_tok = 6 * n_dense / 1e9
print(f"actual params: {n_actual/1e6:.1f}M ({n_actual*2/2**20:.0f} MB bf16 file est.)")
print(f"dense-equiv params: {n_dense/1e6:.1f}M")
print(f"compression ratio (dense/actual): {n_dense/n_actual:.2f}x")
print(f"FLOPs/token (training, 6x dense): {flops_tok:.2f} GFLOP")
print(f"100B tokens total: {100e9*flops_tok/1e9:.0f} EFLOP")
for rate_tf, name in [(0.75e15, '4xRTX FP8 @0.75PF eff'),
                      (1.0e15, '4xRTX FP8 @1.0PF eff'),
                      (0.59e15, '8xH100 bf16 (measured 580K tok/s scale)')]:
    days = 100e9 * flops_tok * 1e9 / (rate_tf * 1e12) / 86400
    tok_s = rate_tf * 1e12 / (flops_tok * 1e9)
    print(f"  {name}: {tok_s/1000:.0f}K tok/s -> {days:.1f} days")
