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

## Results

*(to be appended on completion: final val, downstream suite, OOD ppl,
reference comparisons — Pythia-160m@step2000 matched-budget, Pythia final,
SmolLM-135M.)*
