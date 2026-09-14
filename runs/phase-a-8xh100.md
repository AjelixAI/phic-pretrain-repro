# Phase-A run record — 8×H100, tied-blocks, 3.0B tokens

Node: scaleway H100 node (51.159.86.99), 8×H100 80GB, `/root/phi/`.
Date: 2026-09-14. W&B project: `phic-pretrain`.

## Exact launch

```bash
source /root/.wandb_env
export WANDB_API_KEY=$WANDB_API_KEY PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup /root/venv/bin/python -m torch.distributed.run --nproc_per_node=8 \
  /root/phi/pretrain_tb.py --mode tied --gpu 0 --steps 5722 --bs 32 \
  --seq 2048 --d 512 --layers 22 --ffn 1792 --block 32 --rank 64 \
  --lr 3e-4 --warmup 115 --decay-start 4580 --project phic-pretrain \
  > /root/phi/pretrain.log 2>&1 &
```

Data cache: `/root/phi/data_cache.pt` (3.2B int32 tokens, FineWeb-Edu
sample-10BT via `scripts/pretokenize.py`).

## Bugs found and fixed during bring-up (all now gated)

1. **Label-shift bug**: model saw the current token's label (loss → 0.0003
   "too good") — labels must be shifted: `model(xb)` vs `labels=xb`; the
   loss compares logits[:, :-1] with y. Fixed + a sanity gate.
2. **Embedding init scale**: tied head init N(0,1) made step-0 logits ~±60 →
   loss 96.5 at d=512. Fixed to N(0, 0.02) (standard practice).
3. **Val indexing bug** (crashed a run at its step-1000 eval): val batch was
   still 2-D indexed on the 1-D cache. Fixed; val = unseen tail slice.

## Sanity gates (in pretrain_tb.py)

- step-0 loss must be within tolerance of `ln(vocab) ≈ 10.83` — catches both
  the copy-shortcut and the init-scale pathologies.
- abort if step-0 loss < 5 (leak) or > 50 (scale pathology).

## Training curve (live record)

| step | train loss | val loss |
|---|---|---|
| 0 | 10.94 | — (SANITY OK) |
| 50 | 8.94 | |
| 500 | 6.56 | 6.56 |
| 1000 | 6.22 | 6.09 |
| 1500 | 5.78 | 5.75 |
| 2000 | 5.56 | 5.50 |
| 3000–4500 | 4.8–5.0 grind phase | 4.84 @ ~4500 |
| 4580 | **anneal starts** (LR 3e-4 → 6e-6) | |
| 5722 | final | TBD |

Throughput: ~410k tok/s steady across 8 ranks (524,288 tokens/step × 5722
steps = 3.0B tokens).

## Results (final)

Final val loss: **4.53 nats** (ppl 92.7). Anneal contribution: −0.31 nats
(4.84 stable-phase floor → 4.53 annealed).

### Downstream (500-doc slices; ours = mean-logprob scoring ≈ lm-eval acc_norm; refs = lm-eval 0.4.13)

| model | tokens | arc_easy | hellaswag | piqa | lambada |
|---|---|---|---|---|---|
| **Ajelix-Fiber-130M** | 3.0B | 20.4 | 26.8 | 46.2 | 6.8 |
| Pythia-160m@step2000 (matched) | ≈2–4B | 35.4 | 36.8 | 58.0 | 14.6 |
| Pythia-160m final | 300B | 42.6 | 39.0 | 62.2 | 12.6 |
| SmolLM-135M | 1.1T | 59.6 | 46.6 | 69.4 | 35.6 |

OOD: WikiText-103-raw test ppl 447.58 (loss 6.1038).

### Honest caveats

1. **Capacity mismatch**: 55.9M actual params vs Pythia's 160M at matched
   tokens. A dense-56M control at the same budget is needed to separate the
   architecture penalty from raw capacity.
2. **Missing EOS separators** (found in review): pretokenize.py glued
   documents without EOS between them — nonstandard vs GPT-2/Pythia/OLMo.
   Fixed in pretokenize.py (eos between docs); corrected rerun Phase-A2.
3. All reference rows are raw base models; budgets disclosed per row.
