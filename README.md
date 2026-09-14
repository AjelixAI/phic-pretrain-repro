# phic-pretrain-repro — Phase-A from-scratch pretrain (tied-blocks architecture)

Complete reproduction bundle for the Phase-A pretraining run: a from-scratch
55.9M-actual-parameter / 130M-dense-equivalent **tied-blocks** transformer,
trained for exactly **3.0B tokens** on open FineWeb-Edu data with the
OLMo-style recipe (WSD schedule, W&B-logged), on 8×H100.

Part of the **ΦIC (Phase-A)** project — see the thesis in the parent repo
(`fiber-phi-engine/THESIS.md`). This repo exists so the run is bit-level
reproducible: data, tokenizer, architecture, schedule, launch commands, eval.

## The architecture in one paragraph

A standard pre-norm transformer decoder, except the 22 attention blocks are
**tied**: each layer is `emb + U·(σ(B·emb))·Vᵀ` where `B` is a **shared
butterfly-structured low-rank core** (block-diagonal width-32 blocks, 2
stages), and each layer gets only a small **residual correction** `U·Vᵀ`
(rank 64). Layers share the big core; the per-layer parameters are tiny. The
weight tensors are **materialized** from these tied/factorized forms at
runtime, so the forward pass is a standard matmul sequence — the memory
saving is in *storage* (the files: 112MB bf16 dense-equivalent → 12MB
tied), which is the whole point for on-device use.

| param | value |
|---|---|
| layers / d_model / ffn | 22 / 512 / 1792 (SwiGLU) |
| tied block width / rank / stages | 32 / 64 / 2 |
| vocab | 50304 (GPT-NeoX-20B BPE, padded from 50281), tied embeddings |
| actual / dense-equiv parameters | **55.9M / 130M** |
| context | 2048 |

## Data

- **FineWeb-Edu** `HuggingFaceFW/fineweb-edu`, config `sample-10BT`,
  train split, byte-level BPE (EleutherAI/gpt-neox-20b), no special tokens.
- 3.2B tokens tokenized to a flat int32 cache (`data_cache.pt`), consumed as
  contiguous 65536-token slices per step (8 ranks × 32 seq × 2048 tokens).
- Val = the unseen tail slice of the same cache (guaranteed no overlap with
  training batches).

## Recipe (OLMo-style)

| phase | steps | LR |
|---|---|---|
| warmup | 0 → 115 | 0 → 3e-4 linear |
| stable | 115 → 4580 | 3e-4 |
| **anneal (WSD decay)** | 4580 → 5722 | 3e-4 → 6e-6 linear |

- Optimizer: AdamW, bf16 compute, activation checkpointing per block,
  grad-clip 1.0, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Sanity gates** (abort the run if violated): step-0 loss must be
  `ln(50304) ≈ 10.8` (certifies no copy-shortcut/label-shift bug and correct
  init scale; embedding init N(0, 0.02)). These gates exist because the
  project previously hit exactly those two failure modes — see
  `runs/phase-a-8xh100.md`.
- Logging: W&B project `phic-pretrain` (loss, tok/s, LR, val every 500 steps).

## Reproduce

```bash
# 0. env: torch + transformers + datasets + wandb, 8×H100
pip install torch transformers datasets wandb

# 1. tokenize (single node, ~32 procs) -> data_cache.pt (3.2B int32)
python scripts/pretokenize.py

# 2. pretrain (the exact Phase-A launch)
torchrun --nproc_per_node=8 scripts/pretrain_tb.py \
  --mode tied --gpu 0 --steps 5722 --bs 32 --seq 2048 \
  --d 512 --layers 22 --ffn 1792 --block 32 --rank 64 \
  --lr 3e-4 --warmup 115 --decay-start 4580 --project phic-pretrain

# 3. downstream + OOD eval on the annealed checkpoint
python scripts/eval_ours.py --ckpt ckpt_pretrain_tied-22L512d-b32-r64.pt
```

Reference evals for comparison (same methodology; lm-eval):
`EleutherAI/pythia-160m` @ revision `step2000` (≈2–4B tokens, the
matched-budget checkpoint) and final (300B), plus `HuggingFaceTB/SmolLM-135M`
(1.1T tokens). Tasks: lambada_openai, piqa, hellaswag, arc_easy (500-doc
validation slices) + WikiText-103-raw-test perplexity per model with its own
tokenizer.

## Files

| file | role |
|---|---|
| `scripts/pretokenize.py` | FineWeb-Edu → flat int32 token cache |
| `scripts/pretrain_tb.py` | the pretrainer (dense/tied modes, DDP, WSD, gates) |
| `scripts/eval_ours.py` | 4 downstream tasks + OOD ppl for OUR model |
| `runs/phase-a-8xh100.md` | the actual run record: commands, curve, fixes, results |
