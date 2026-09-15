# Speed engineering record — tied-blocks trainer (RTX PRO 6000 + H100)

All changes verified with `scripts/verify_fast.py` (forward + all gradient
groups, randomized blocks, five shape families) — the standing rule is that
training results must not change, only speed.

## Final ladder (per card, bs=32, tied-22L512d-b32-r64)

| config | tok/s | gain |
|---|---|---|
| reference implementation (staged butterfly, ckpt on, materialized CE) | 52,770 | 1.0× |
| + composed frozen operator (basis → one dense buffer) | 81,446 | 1.54× |
| + no activation checkpointing (`--no-ckpt`) | 105,625 | 2.00× |
| + scatter-GEMM blockdiag | **106,659** | **2.02×** |

8-card aggregates: RTX station ~853k tok/s; H100 node expected similar
(same per-card math) → 1B-token anneal ≈ 20 min local / 40 min node;
100B-token Phase-C ≈ 32 h.

## What was done (all math-preserving, bf16 reassociation only)

1. **Composed frozen operator**: the 2-stage frozen butterfly basis + fixed
   permutations equals ONE dense linear operator. Precomputed at init by
   running the original code path on the identity matrix
   (`basis(I)[0]`), stored as a **non-persistent buffer** (regenerated from
   seed → the 12 MB checkpoint file story is unchanged). Verified: forward
   and all gradients at bf16 rounding scale (3.7–9.7e-3 rel), U/V grads
   bit-exact.
2. **Scatter-GEMM blockdiag**: the trained block-diagonal multiply
   (`blocks[j]`, layout `[output_row, input_col]`) as one dense GEMM with
   `W_total` rebuilt per step via a precomputed-index scatter
   (`index_copy`). **The parameter layout is `[out, in]` — the scatter
   needs `blocks.transpose(-1, -2)`**; the harness caught the transposed
   version because the init state (identity blocks) is symmetric — the
   harness now randomizes blocks to keep this bug class detectable.
3. **`--no-ckpt`**: activation checkpointing removed — pure recompute
   strategy, zero math change; fits at bs≤48 (MLP activations ~66 GB at
   bs=64 + 26 GB fp32 logits overflow 95 GB).
4. **`--chunked-ce N`**: chunked cross-entropy (never materializes the full
   `[bs, seq, 50304]` logits; 26 GB at bs=32 fp32). Required for big-batch
   runs; costs ~6% at bs=32 (checkpoint recompute).

## Measured dead ends (do not retry without new evidence)

| attempt | result |
|---|---|
| torch.compile (default) | parity with uncompiled fast path (~+8% instantaneous, minus compile tax) |
| torch.compile max-autotune | inductor crash: illegal memory access in backward |
| CUDA graphs (reduce-overhead) | incompatible with manual `.backward()` + cudagraph-tree pooling at our graph scale; 3 mitigations attempted (mark_step_begin, dropping logits output, loss.clone) — all fail |
| compiled autograd | same cudagraph-tree error |
| residual fold (`W_res = V.T @ U.T`) | **corrupts V/U gradient routing** (1.0 rel error) — caught by harness, reverted; correct version needs a custom autograd Function |
| big-batch scaling (bs=128) | old implementation: 31,244 (inverted!); with the scatter-GEMM fix: 70,718 — still below bs=32; memory wall: no-ckpt fits only at bs≤48 |

## The bottleneck story (profiler-verified)

- original: ~3,800 CUDA launches/step; >70% of GPU time in copies, pads,
  reshapes, launches — not matmul. MFU ~4%.
- fast path: 794 launches/step; MFU ~24%. Remaining: CPU dispatch
  ("Command Buffer Full" 42% of CPU time — the +1% null result after
  removing the 40%-of-step bmm proves the step is dispatch-bound), the
  blockdiag dense GEMM's 16× wasted FLOPs (acceptable — 6.6× faster
  wall-clock than the bmm), memory-bound elementwise chains.

## Files

| file | role |
|---|---|
| `scripts/pretrain_tb_fast.py` | the verified fast trainer (adds `--compile`, `--chunked-ce`, `--no-ckpt`, `--init-from` pending) |
| `scripts/verify_fast.py` | the equivalence harness (run before ANY kernel change) |
| `scripts/shuffle_cache.py` | document-level shuffle (the Phase-B lesson) |
| `scripts/sample_olmomix.py` | proportional mix sampler (pattern reused for Dolmino) |
