# Mid-training / anneal plan — Ajelix-Fiber-130M

Complete plan, rationale, artifacts, and instructions for running
mid-training ("Dolmino-style" annealing) on the Phase-B base model.
Branch: `anneal-plan`. Status: READY — pending Phase-B completion
(checkpoint `stable-step40000` already saved on the training node).

---

## 1. What mid-training is and why it works

Mid-training is a small, high-leverage stage between large-scale pretraining
and post-training (SFT/DPO). OLMo 2's published recipe:

| stage | tokens | data | LR |
|---|---|---|---|
| pretraining | 4T (99%) | OLMo-mix-1124 (95% web + code/math/science) | WSD, stable 3e-4 |
| **mid-training** | **50B (1.25%)** | **Dolmino-mix-1124** — curated HQ web, StackExchange Q&A, math (TuluMath/TinyGSM/synthetic), wiki, papers, decontaminated FLAN | **annealed to 0** |
| post-training | — | SFT + DPO + GRPO | — |

Why it works (three compounding reasons):

1. **The annealing window is the most valuable real estate in a WSD run.**
   The model improves most sharply as LR decays (settles into its final
   basin). Feeding that window curated, task-relevant data maximizes the
   value of the final steps. Same mechanism as BitNet's "cooldown".
2. **The data is benchmark-adjacent by design** — math word problems, Q&A
   formats, instruction-like text. This is the open version of the industry
   practice of aligning training mixes with evaluation targets.
3. **It is nearly free**: ~1.25% of the training budget for a published,
   measurable downstream jump.

OLMo 2 detail: the 7B/13B models ran three anneals on differently-weighted
Dolmino subsets and merged them (model soup); the 1B ran one anneal.

## 2. Our starting point (Phase-B base model)

| item | value |
|---|---|
| base run | `phic-pretrain-b`, tied-22L512d-b32-r64 (55.9M actual / 130M dense-equiv) |
| data | OLMo-mix-1124 proportional sample, 27.78B tokens, EOS-separated, document-shuffled |
| schedule | WSD: warmup 300 → stable 3e-4 → anneal 42,400→52,991 (LR → 6e-6) |
| **stable checkpoint** | **`ckpt_tied-22L512d-b32-r64_step40000.pt`** — saved by periodic checkpointing, ~2,400 steps before the Phase-B anneal began (still at stable LR — the ideal mid-training start) |
| final annealed model | `ckpt_pretrain_tied-22L512d-b32-r64.pt` (end of Phase-B anneal, on the generic mix) |
| node | scaleway 8×H100 (410k tok/s); local RTX PRO 6000 station (853k tok/s aggregate with the verified fast trainer) |

## 3. Three paths (decision)

| path | mechanics | verdict |
|---|---|---|
| **A. Fresh anneal from the stable checkpoint** | resume `step40000` weights → anneal to zero over 0.5–1B tokens on a Dolmino-style HQ slice | **RECOMMENDED** — the clean OLMo replication; the Phase-B generic anneal becomes the pilot/control |
| B. Second anneal from the final (annealed) model | brief re-warmup to ~1e-4 → anneal again on HQ data | fallback; precedent = BitNet's two cooldowns; slightly less clean |
| C. Extend stable phase first | more generic tokens, then anneal | only if Phase-C also wants a bigger budget |

## 4. Data pipeline (Dolmino-style slice)

`allenai/dolmino-mix-1124` is public on HF. Composition targets for OUR
mid-train slice (0.5–1B tokens), scaled from the paper's Dolmino table
(High-Quality subset 832.6B + Math 10.7B):

| source | share |
|---|---|
| DCLM-Baseline (fasttext top 7%) | ~50% |
| FineWeb ≥2 score | ~18% |
| FLAN (decontaminated) | ~8% |
| peS2o (academic) | ~8% |
| Wikipedia + Wikibooks | ~8% |
| StackExchange Q&A | ~4% |
| Math mix (TuluMath, SynthMath, TinyGSM-MIND) | ~4% |

Sampling procedure — reuse the proven pattern from
`scripts/sample_olmomix.py` (per-config proportional targets, stride across
shards, our GPT-NeoX-20B tokenizer, **EOS between documents**):

```bash
# on the training node
python scripts/sample_dolmino.py --target 750_000_000 \
    --out /root/phi/data_cache_dolmino.pt
```

(`sample_dolmino.py` = `sample_olmomix.py` with the repo id and per-config
targets swapped; to be committed on this branch before the run.)

**Gates (from PREFLIGHT.md, all mandatory):**
- EOS count == document count
- document-level shuffle (source-blocked cache = invalid; the Phase-B
  shuffle bug is the standing warning)
- proportions printed and checked against the table above
- cache ids within tokenizer range (max < 50,277)
- sanity gate: step-0 loss ≈ ln(50304) + resume offset (see §5)

## 5. Trainer change required: `--init-from`

The trainer currently starts from scratch. Mid-training needs:

```python
p.add_argument("--init-from", default=None)
# in main(), after model construction:
if A.init_from:
    sd = torch.load(A.init_from, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    if RANK == 0:
        print(f"[{tag}] RESUMED weights from {A.init_from}", flush=True)
```

Semantics (the OLMo `reset_trainer_state` pattern):
- weights: loaded from the checkpoint
- optimizer: **fresh** (AdamW state reset — correct for an anneal, which
  restarts the LR schedule anyway)
- LR schedule: the anneal — `--warmup` short (~50–100 steps to peak
  ~1e-4–3e-4) then linear decay to ~0 over the anneal length
- one process loads at startup → no DDP desync (all ranks load the same
  file before training begins)

**Verification gate for the resume path**: after loading, run one batch
and compare val loss against the source checkpoint's recorded val
(± bf16 noise). A mismatch = wrong checkpoint or wrong architecture tag.

## 6. Exact run commands

### Path A (recommended) — from the stable checkpoint

```bash
# node, 8×H100 (~410k tok/s → 1B tokens ≈ 40 min)
source /root/.wandb_env
export WANDB_API_KEY=$WANDB_API_KEY PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=8 scripts/pretrain_tb.py \
  --mode tied --gpu 0 --init-from /root/phi/ckpt_tied-22L512d-b32-r64_step40000.pt \
  --steps 1907 --bs 32 --seq 2048 --d 512 --layers 22 --ffn 1792 \
  --block 32 --rank 64 --lr 1e-4 --warmup 50 --decay-start 1907 \
  --project phic-midtrain --cache /root/phi/data_cache_dolmino.pt
```

Notes:
- `--steps 1907` = 1B tokens (1907 × 524,288). For a 500M-token anneal:
  953 steps.
- peak LR for the anneal: **1e-4** (a third of the pretraining peak — the
  model must settle, not re-explore; matches the BitNet cooldown practice).
  `--decay-start` = `--steps` means "decay from step 0" after the brief
  warmup — i.e. a pure anneal.
- the fast trainer (`scripts/pretrain_tb_fast.py`, local station) runs the
  same command with `--no-ckpt` at ~853k tok/s aggregate → 1B tokens in
  ~20 min. Math verified identical (see §7).

### Path B (fallback) — second anneal from the final model

Same command with `--init-from ckpt_pretrain_tied-....pt` (the Phase-B
final), `--lr 5e-5 --warmup 30` (gentler: the model is already settled).

## 7. The fast trainer and its equivalence guarantee

`scripts/pretrain_tb_fast.py` = the speed-optimized implementation
(2.02× per card: 106,659 tok/s on an RTX PRO 6000 vs 52,770 baseline):

| change | what |
|---|---|
| composed frozen operator | the 2-stage frozen butterfly + fixed permutations collapse to one dense matrix per TiedLinear, precomputed at init from the seed (non-persistent buffer — **the 12 MB file story is unchanged**) |
| scatter-GEMM blockdiag | the trained block-diagonal multiply runs as one dense GEMM (`W_total` rebuilt per step by a precomputed-index scatter) |
| `--no-ckpt` | activation checkpointing off (fits at bs≤48; the recompute was pure overhead) |
| `--chunked-ce N` | chunked cross-entropy for big-batch runs (removes the 26 GB logits wall) |

**Equivalence harness: `scripts/verify_fast.py`.** Compares forward AND all
four gradient groups (blocks, U, V, input) against the reference
implementation on five shape families, with **randomized (non-symmetric)
blocks** — the harness caught two real bugs during development (a
transposed scatter invisible at identity-init, and a gradient-corrupting
residual fold). Current status: all shapes at 3.7–9.7e-3 relative = bf16
rounding scale; U/V gradients bit-exact.

**RULE: any future change to the architecture kernels must pass
`verify_fast.py` (randomized blocks) before any run starts** — recorded in
PREFLIGHT.md.

## 8. Eval protocol (identical to Phase-B)

```bash
python scripts/eval_ours.py --ckpt <mid-trained ckpt>
# references: lm-eval on EleutherAI/pythia-160m (step2000 = matched budget,
# final), HuggingFaceTB/SmolLM-135M — lambada_openai, piqa, hellaswag,
# arc_easy (500-doc slices) + WikiText-103-raw test ppl
```

The mid-trained model gets its own rows; the raw Phase-B base rows stay
intact. Success criteria for the mid-train:
1. downstream suite improves vs the raw Phase-B base (the OLMo-published
   effect replicating on our architecture), AND/OR
2. WikiText-103 OOD ppl improves materially (Phase-B: ~447 at 3B tokens;
   the 27.8B base should already land well below; the mid-train should
   improve further).

## 9. Model versioning and honesty

| artifact | naming |
|---|---|
| raw base (Phase-B final) | `Ajelix-Fiber-130M` (the current HF repo, unchanged) |
| mid-trained variant | `Ajelix-Fiber-130M-mid` (or v1.1) — card discloses: "pretrained on OLMo-mix-1124 (27.8B tokens), mid-trained on a Dolmino-style curated anneal (X tokens)" |

HF checkpoint layout (OLMo-style, one branch per checkpoint):

```
AjelixAI/Ajelix-Fiber-130M
├── main                      ← final released model + card
├── stable-step40000-tokens21B   ← mid-training start point
├── final-step52991-tokens27.8B  ← raw Phase-B end
└── mid-<dataset>-<tokens>       ← each anneal experiment
```

Each checkpoint is ~12 MB (the tied-blocks file) — pushing every
checkpoint is trivial.

## 10. Cost summary

| item | node (8×H100) | local station (8×RTX, fast trainer) |
|---|---|---|
| Dolmino sample + tokenize | ~15 min | same |
| 1B-token anneal | ~40 min | **~20 min** |
| eval re-run | ~30 min | ~30 min |
| **total per anneal experiment** | **~1.5 h** | **~1 h** |

An anneal sweep (4–6 dataset variants) is therefore an afternoon — the
mid-training design question becomes empirical, not guessed.

## 11. Sequence

1. Phase-B finishes → eval table lands (in progress)
2. Commit `sample_dolmino.py` + the `--init-from` patch (this branch)
3. Verify `step40000` checkpoint loads + sanity-val matches (PREFLIGHT §5)
4. Run Path A (1B tokens, lr 1e-4, Dolmino-style slice)
5. Eval → compare vs raw base rows
6. If the effect replicates: sweep 3–4 anneal data variants (~1 h each)
7. Push best checkpoint to HF as `Ajelix-Fiber-130M-mid` with card update
