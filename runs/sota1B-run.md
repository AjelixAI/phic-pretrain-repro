# The SOTA 1B-dense run record — launched 2026-09-16

## Configuration (all verified pre-launch)

| | |
|---|---|
| model | d=2048, L=16, ffn=7168, tied-blocks, block 32, rank 64, stages 2 |
| dense-equivalent | **975.4M (50k vocab) → ~1,095M (the OLMo 2 100,352 vocab) — the 1B-class+** |
| actual params | **293.2M** (the compression **3.74×**) |
| tokenizer | **OLMo 2, 100,352, byte-level BPE** (the scaling-law optimum at this size) |
| context / RoPE | 4,096 / **base 10,000** (the RoPE bug fixed: the head_dim was the base!) |
| data | **87.3B tokens**: the OLMo-mix (the 10.5% math/code/science upsampled) + the Dolmino annealing tail (1.96B), one-stage WSD |
| vals | val_generic_500M (the held-out generic, disjoint) + val_anneal_250M (the held-out anneal-domain) |
| schedule | the warmup 500 → stable 3e-4 → **the linear decay-to-zero** over the final ~10k steps (arXiv 2502.15938: D2Z beats 10%-decay; the benefit grows with TPP — ours is ~90 TPP) |
| precision | **FP8 P-build** (the autograd Function: the FP8 forward, the exact-bf16 backward via the blockdiag extraction) + the bf16 x@P GEMMs (the measured net loss with the eager FP8) |
| the hardware | 4× RTX PRO 6000 Blackwell (96 GB), GPUs 0-3 |
| the throughput | **~88K tok/s** (~127 TFLOPS/GPU effective, the 25% MFU — the 2× the H100's MFU at its run) |
| duration | ~9.7 days (87.3B tokens ÷ 88K tok/s) |

## Protections (all tested)

| | |
|---|---|
| full checkpoints | every 700 steps (~17 min): weights + optimizer + step, the rotation (the running + the milestones, the keep-5) |
| true resume | weights + optimizer + step + data position — **tested with an actual kill-and-resume** |
| crash auto-recovery | **the watchdog** (sota_watchdog.sh): the run dies → the auto-resume within 5 min, up to 50 restarts |
| the checkpoint integrity | 595/595 tensors bitwise identical through the save/load round-trip |
| monitoring | val/general + val/anneal-domain every 500 steps, the loss + the throughput every 25, the wandb: phic-sota1B |

## The data build (all gates passed)

| | |
|---|---|
| the stream | 85.3B tokens (the OLMo-mix generic) + 1.96B (the Dolmino tail) |
| the shuffle | the global PCG64 permutation, the mixing gate (the hard-fail) |
| the val carves | the generic 500M + the anneal-domain 218M, both disjoint by file |
| the cache | 341 GB raw int32 memmap (the zero-copy) |
| the proportions | the dclm 87%, the starcoder 6%, the math 3%, the academic 3.5%, the wiki 0.5% (the reasoning upsampled from OLMo's 3.2%) |
| the EOS == docs | verified per segment |
| the shortfall | the 85.3B+1.96B = ~87.3B vs the 100B target (the 87%): the per-file cap undershoot on the single-line-JSON sources — documented, the proportions intact |

## The session's verification cascades (the lessons)

1. The custom scorer was broken (the chance-level rows) — fixed, validated.
2. The RoPE base was head_dim (64) — the positional aliasing 6× — fixed to 10,000.
3. The FP8-eager is net-negative for the x@P (the quantize cost > the GEMM saving) but net-positive for the P-rebuild (the tiny quantize, the big GEMM) — the measured hybrid.
4. The CUDA graphs fail at the 1B config (the 27.8% divergence, no speedup — the GPU-bound regime) — deferred with the debug data.
5. The val-domain must match the question (the general val for the preservation, the anneal-domain val for the skill gain).
6. The stale numbers propagate silently (the config_math's 50k-vocab 975.4M vs the true 1,095M with the 100k vocab) — re-verify after every change.
7. The checkpoint system: the full (weights+optimizer+step), the rotation, the resume — tested with an actual kill.
8. The dataset builder: the streaming design (the bounded RAM), the per-config truncation, the mixing gate.

## The in-flight checkpoint verification (step 14,000, the run untouched)

| check | result |
|---|---|
| the checkpoint load | missing 0 / unexpected 0 — the state dict intact |
| val generic (the eval-side recompute) | 4.3734 nats/token — matches the training's logged val ✓ |
| the generations | fluent early-stage English; the high-frequency loops = the expected base-model behavior at ~3B tokens |
| the training | untouched: the eval ran read-only on the milestone, the spare GPU memory, the ~1 min |

**The lesson**: the false-alarm garbage generation was the EVAL's wrong tokenizer — the trainer's script loaded `EleutherAI/gpt-neox-20b` (the 130M FineWeb-era leftover) while the data is tokenized with `allenai/OLMo-2-1124-7B` (the OLMo 2, the same 100,352 vocab size, the different token mappings). The same-size vocab made the mismatch silent: the ids are valid in both mappings, the decode is nonsense. The eval scripts MUST use the DATA's tokenizer. The trainer's tokenizer line is now documented as a decoy: the training never touches it (the ids come from the cache), but any encode/decode work must use the OLMo 2 map.

## The in-flight benchmark sweep (step 14,000, ~2.75B tokens)

| benchmark | acc | chance | note |
|---|---|---|---|
| ARC-Easy | 0.290 | 0.25 | the knowledge tasks: the data-limited at this stage |
| HellaSwag | 0.302 | 0.25 | the adversarial completions: the mid-training skill |
| PIQA | 0.499 | 0.50 | the MC discrimination: not yet developed; both prompt formats tested |
| LAMBADA | 0.027 | ~0.001 | the exact-word prediction: weak now, climbs sharply mid-run |

**The eval bugs found and fixed (the user's challenge: "maybe the eval is wrong?")**:
1. **The answer-key bug**: the first harness scored ARC/HellaSwag assuming the gold = the index 0 — the ARC's gold is the `answerKey` ('A'/'B'/...), the HellaSwag's is `label`. The first numbers measured the position bias, not the accuracy. Fixed in `scripts/ops/eval_benchmarks2.py`.
2. **The PIQA format test**: the raw and Q/A-wrapped formats both at the chance → the model's genuine early-stage behavior, not the format sensitivity.
3. **The LAMBADA method** (the argmax == the target's first token): verified correct; 0.027 is the plausible at 2.75B tokens (the Pythia-160M's matched-stage reference: ~0.03-0.05).

**The lesson**: the uncorrected MC "accuracy" is silently the choice-order preference — the gold key must come from the dataset's field. The reference for the in-flight benchmarks = the Pythia's checkpoints at the SAME data step, never their 300B-token finals. Re-check this suite at ~30B tokens to watch the skills develop; the val loss stays the primary signal.

## The throughput-lever sweep (step ~16,300, the run untouched; contended GPU-3 measurements)

| config | ms/step | tok/s/GPU | vs baseline |
|---|---|---|---|
| bs 12, ckpt-all (the running config) | 3,942 | 12,468 | — |
| bs 12, ckpt-every-2 | 3,563 | 13,797 | +11% |
| bs 12, **ckpt-every-4** | **3,420** | **14,372** | **+15%** |
| bs 16, ckpt-every-2 | 4,684 | 13,990 | +12% |
| bs 24, any | OOM | — | needs an uncontended 96GB GPU |
| bs 12, no-ckpt | OOM | — | needs ~70GB; misses by ~200MB when contended |

**The step's wall-time breakdown (the profile)**: the backward ~79% (of which ~60% is the checkpointing's activation recompute), the forward ~25%, the FP8 P-rebuild **~4%** (the earlier "50-70% of the step" was FLOPs-normalized, not wall-time — the FP8-build made it cheap), the optimizer ~1%.

**The trainer patch (committed)**: `ckpt_every` — the checkpoint only every Nth layer (the Block stores the layer_idx; the ckpt_every=1 = the old behavior). The run #2's config: **bs 24 + ckpt-every-4 on an uncontended GPU: the projected ~1.4-1.5× (the ~120-130K tok/s aggregate)**, with the FP8-x@P + the CUDA-graphs as the next ~1.2-1.3× (the graphs' 1B gate: the 27.8% divergence: the debug data saved).

**The honest 10× verdict**: the theoretical FLOP ceiling on the 4×RTX = the ~280K tok/s (the 3.2× today). The ~10× needs the hardware scale or the optical engine.

## The checkpoint-over-checkpoint benchmark comparison (the step 14,000 -> 24,500, +1.38B tokens)

| benchmark | step 14,000 | step 24,500 | delta | interpretation |
|---|---|---|---|---|
| val generic (the run's own val) | 4.373 | **4.326** (the step 24,000's val) | -0.047 | the steady WSD's descent ✓ |
| ARC-Easy acc | 0.290 | 0.285 | ~flat | the knowledge tasks: the data-limited: the noise-dominated at this stage |
| HellaSwag acc | 0.302 | 0.295 | ~flat | the same: the MC discrimination develops later |
| PIQA acc (n=200) | 0.499 | 0.470 | ~flat | the chance-level: the expected |
| LAMBADA acc | 0.027 | 0.033 | +0.006 | the exact-word prediction: the first sign of the movement |

**The reading**: the MC benchmarks are dominated by the noise at this data stage — the val loss (the -0.047 nats) is the real signal of improvement. The benchmarks' expected development: the mid-run (the 10-40B tokens). The re-check cadence: the ~30B tokens (~step 155,000).
