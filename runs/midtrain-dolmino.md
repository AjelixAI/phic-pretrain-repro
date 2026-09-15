# Mid-training run record — Ajelix-Fiber-130M, Dolmino anneal

Date: 2026-09-15 · Node: scaleway 8×H100 (scw-focused-meninsky) ·
W&B: wandb.ai/excelpro-excelpro/phic-midtrain/runs/sxrhk51e

## Preflight (all gates verified by command output, per PREFLIGHT.md)

| gate | result |
|---|---|
| sample: 963,395,417 tokens, 1,210,499 docs, EOS == doc count | PASS |
| id range: max 50,276 < 50,304 (GPT-NeoX-20B vocab) | PASS |
| proportions: dclm 67.1%, flan 8.9%, wiki 8.1%, pes2o 7.3%, stackexchange 4.7%, math 4.0% (top-up applied: +11.7M math) | PASS (recipe §4) |
| global shuffle: longest similar-length doc run 239 → 21 (raw → shuffled); first 20 docs 37–19,906 tok — no source blocks | PASS |
| trainer resume: weights loaded from ckpt_tied-22L512d-b32-r64_step40000.pt (strict) | PASS |
| resume gate: step-0 loss 4.406 vs the recorded Phase-B stable plateau ~4.125 (+0.28 = the new domain mix) | PASS |
| val on unseen dolmino tail (dry-run): 4.0625 | sane |
| optimizer: fresh AdamW (reset_trainer_state pattern); LR 1e-4 peak, warmup 50, linear decay to 2% floor over the run | per plan §5 |
| epochs = 1 (steps 1837 × 524,288 = 963.2M ≈ cache 963.4M — no wrap) | PASS |
| W&B: logged in via WANDB_API_KEY, run syncing | PASS |

## Launch command (the reusable mid-training command)

```bash
source /root/.wandb_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=8 /root/phi/pretrain_tb_fast.py \
  --mode tied \
  --init-from /root/phi/ckpt_tied-22L512d-b32-r64_step40000.pt \
  --steps 1837 --bs 32 --seq 2048 --d 512 --layers 22 --ffn 1792 \
  --block 32 --rank 64 --lr 1e-4 --warmup 50 --decay-start 1837 \
  --project phic-midtrain --cache /root/phi/data_cache_dolmino_shuf.pt \
  --no-ckpt --chunked-ce 8192 --tag-suffix=-mid-dolmino-963M
```

## Throughput (cross-checked, not just the banner)

Banner: ~1.32M tok/s. Sustained cross-check: steps 275→450 = 175 steps ×
524,288 tok = 91.75M tokens in ~90 s wall = **~1.02M tok/s sustained**
(8×H100 ≈ 130 TFLOPS/GPU effective ≈ 25% of peak for this model's matmul
sizes, with the P-form merged-operator path ON, 154 instances).
~3× the 410K tok/s the plan estimated for this node.

## Trainer changes (committed: scripts/pretrain_tb_fast.py)

- `--init-from` (weights only; fresh optimizer — OLMo reset_trainer_state)
- resume-aware sanity gate (resume band [2.8, 5.4] vs the scratch ln(V) gate)
- `--tag-suffix` (protects prior checkpoints from overwrite)
- P-form merged-operator training path (`ptied_train.enable_ptied`, ON by
  default, `--no-ptied` opt-out); duck-typed class matching (the __main__
  trap); `import torch._dynamo` scoping fix (bound as `_dynamo`)

## Honesty notes

- The plan referenced `scripts/verify_fast.py` (the Phase-A/B harness on the
  local station, not reachable in this session). Its function for the P-form
  path is superseded by `scripts/ptied_verify.py` (gradient + loss-curve
  equivalence vs the reference; worst grad rel 0.75%, 50-step loss curves
  identical to 3 decimals). Recovery pending.
- eval_ours.py numbers land here after the run (§ below, to be filled).

## Results (filled after the run + eval)

- final val loss: TBD
- eval suite vs the raw Phase-B base and the references: TBD

## Eval protocol reconciliation (2026-09-15, same-session)

lm-eval 0.4.13, 0-shot, FULL validation sets, both models as verified
Llama exports, same node, same day — this is now the authoritative
protocol (internally apples-to-apples):

| task | base (Phase-B, 27.8B) | mid 963M-Dolmino (22.0B total) |
|---|---|---|
| arc_easy acc | 39.4 | 39.7 |
| hellaswag acc_norm | 26.5 | 26.5 |
| piqa acc | 58.2 | 57.8 |
| lambada_openai acc | 15.6 | **11.0 (−4.6)** |

**Protocol discrepancy found**: the earlier recorded rows (arc 41.0,
hswag acc_norm 34.6) do NOT reproduce under 0-shot (checked) or 10-shot
(probed: hswag 26.5, arc 37.6) on this node/version. Their original
invocation is not recoverable from the records; they are marked
UNREPRODUCIBLE for cross-session comparison. All sweep arms are compared
under the same-session protocol above.

**Budget confound found and being fixed**: the 963M arm's total is 22.0B
vs the base's 27.8B — a matched-budget staged sweep is running: same
27.8B total, replaying the original generic tail from the step40000
checkpoint (byte-identical trajectory) and switching to the Dolmino mix
for the final window (1.81B / 3B / 6.81B arms).

**First finding**: the pure-Dolmino tail regresses lambada (web-text
prediction) by 4.6 points while leaving the MC tasks flat — the staged
arms (generic tail preserved) are designed to test whether the staged
mix fixes this.
