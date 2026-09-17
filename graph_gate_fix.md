# CUDA-graph gate: SOLVED (2026-09-17)

Two independent bugs, both found today:

1. **Capture crash (illegal memory access)** = launch race. With pinned launches
   (CUDA_LAUNCH_BLOCKING=1, or a torch.cuda.synchronize() before capture_begin)
   the identical capture succeeds. The AccumulateGrad/stream-mismatch warning was
   the visible symptom.

2. **The 27.8% "loss divergence" = gate-harness accounting bug, not graph numerics.**
   The graph path's 3 capture-warmup steps on batch 0 are REAL optimizer steps;
   comparing graph replay i against eager step i compared graph-step (i+4) vs
   eager-step (i+1). Evidence: step-0 losses already differed (11.57 vs 10.34)
   with identical weights and identical batch.

Fixed harness (both paths take the same 3 warmup steps):
  max rel loss diff over 30 steps: 0.00026 (bf16 noise floor) -> GATE: PASS

Speed reality: on a contended GPU the tiny-config ratio is 1.01x (compute-bound);
the real-config gain = amortizing the eager FP8-quantize overhead (~34 ms/step,
which is what made FP8_GEMM=1 net-negative on this rig) + launch gaps.
Expected ~3-4% on the current 4x RTX rig; the big payoff is on H200/B200
continuation configs, where native FP8 tensor cores + graphs flip FP8 GEMMs
to a real ~2x win. The current SOTA run is NOT being restarted for ~3-4%.
