# The Optical Twin — why Ajelix-Fiber exists and what it is for

This document records the full-circle architecture story: the Ajelix-Fiber
model is the **training twin of the optical inference engine** — the
factorized form exists so that quality can be proven *before* the optics
are built. Written after Phase-B (27.8B tokens, harness-verified
benchmarks) and the inference-speed campaign.

---

## 1. The wall being attacked

GPU inference is **bandwidth-bound**: every generated token streams every
weight from memory to the compute units.

- Dense 27B bf16: 54 GB of weight reads per token → ~16 tok/s ceiling on a
  900 GB/s part, and 54–162 GB of VRAM no consumer card has.
- Measured on this project's host: a 135M-class dense model does
  **168 tok/s** batch-1 — and our Phase-B verification reproduced it
  exactly (Pythia-160m: 167.4 tok/s). The wall is real and reproducible.

## 2. The mechanism that removes the wall

The dense weight stream is replaced by **light traversing a fixed physical
matrix** (a multimode fiber / DMD stack). The base transform is performed
by propagation physics with **zero digital weight reads** — the weights
never exist in memory; they are the physics.

The digital remainder (what a coprocessor still computes per token):
trained corrections + attention + norms — for a 27B-class model
~1.9G MACs/token, ~0.6–1.2 GB of storage. The coprocessor needs < 2 GB:
**VRAM buys context, not model storage.**

Thesis projection: **7,800 tok/s at 0.013 J/token** on a 250 EUR
coprocessor — the 44× claim vs the bandwidth-bound GPU.

## 3. The mapping: Ajelix-Fiber ↔ the optical engine

| Ajelix-Fiber component | optical engine role |
|---|---|
| Frozen butterfly basis (seed-generated, **shared across all 22 layers**) | the physical cable/DMD stack — light propagating through the fiber *is* the basis matrix; one measured stack serves every layer (tied basis validated at +0.011 nats) |
| The **107 MB file** (measured): trained block-diagonal corrections + rank-64 residuals + the tied 50304-vocab emb/head pair | the digital corrections — the small part that exists in silicon |
| Attention, norms, KV cache | coprocessor work |
| The bandwidth wall | **eliminated by construction** — the base transform has zero weight reads |

Consequence for the model file: on the optical engine the file shrinks
again — the basis is physical glass ("copying the model = photocopying").
The digital file **is the corrections + the tied I/O**: the small part ever
meant for silicon (the basis itself is regenerable from the seed and can be
dropped from the file: -19.5 MB).

## 4. What Phase-B proved (the precondition the hardware needed)

1. **Quality survives the factorization.** The trained corrections against
   the seed-basis reach competitive quality: harness-verified wins at
   matched budget (arc 41.0 vs 35.4, lambada 16.4 vs 14.6 vs the dense
   160M referee at 2–4B tokens). This was the risk the hardware plan
   carried: if the factorized form could not carry quality, the optics
   would be multiplying garbage.
2. **The basis is seed-deterministic and layer-shared** — physically: one
   cable stack, no per-layer storage. Validated (+0.011 nats).
3. **The deployment execution model works**: the KV-cached CUDA-graph
   decoder does **150.8 tok/s eager-tied** with token-identical
   output (100% match vs the full-context decoder) — the same
   philosophy the optical engine embodies: tiny scheduled digital work,
   the heavy lifting in the substrate.
4. **The verification discipline exists**: the V3 proof suite
   cross-verified inference against LIVE wave propagation; every optical
   result will be checked against this digital twin the same way the
   kernel rewrites were (100%-token-match gates).

## 5. Honest scope

- **Delivered by the software model**: model file 2.4× smaller (bf16, measured: 106.7 MB vs the 260 MB dense
  export; an earlier note said "12 MB / 22×" — an accounting error,
  corrected 2026-09-15), competitive quality at matched budget, the
  factorization-tolerance proof, the training/deployment pipeline.
- **Not delivered by software, by design**: the speed and energy claims
  (44×, 7,800 tok/s) live in the optical hardware. In pure software the
  factorized form recomputes the basis (~3× FLOPs, ~10× kernel launches —
  THESIS.md:45 stated this from the start; the CUDA-graph KV decoder is
  the software approximation of the substrate's "one scheduled op").
- A software bridge exists (fused tied kernels) and may win on CPU/edge —
  but the full claim is the hardware stage.

## 6. What remains between here and the laser bench

| step | status |
|---|---|
| Wave-optics proof suite (validated vs live measurement, V3) | ✅ done (`proof/` in the parent repo) |
| BOM + bench procedures (~1,000 EUR: 4K DMD, multimode cables / liquid guides, line-scan cameras / photodiode banks, Xeon/FPGA for corrections) | ✅ documented (`hardware/`) |
| **Measure the REAL cable stack's transfer matrix** | ⏳ calibration study — the physical fiber ≠ the seed ideal |
| **Retrain the corrections against the MEASURED matrix** | ⏳ cheap here: the corrections are the small trained part — the pipeline swaps the seed-basis for the measured matrix natively (hours, not weeks) |
| **Analog-precision tolerance test** | ⏳ inject the measured photodiode/driver noise into the basis during eval; quantify the quality drop (bf16 tolerance suggests robustness; analog noise is a different noise model) |
| **End-to-end bench: optical inference cross-verified vs the digital twin** | the final proof — 100%-match gates, same as everything else in this project |

## 7. The sequence to the demo

1. Acquire the BOM parts (~1,000 EUR).
2. Bench the cable stack; measure its transfer matrix (calibration study).
3. Swap the measured matrix into the trainer as the basis; retrain the
   corrections against it (hours on this project's pipeline).
4. Inject the measured analog noise; re-run the eval suite (the tolerance
   certificate).
5. Live demo: optical inference of Ajelix-Fiber, cross-verified against
   the digital twin, token by token.

## 8. Sources

- `THESIS.md`, `README.md`, `RUNTIME_CONSTRAINTS.md`, `SCALING.md` in the
  parent repo (fiber-phi-engine) — the original exploration.
- `proof/`, `hardware/` — the V3 wave-optics validation and the BOMs.
- Phase-B: `runs/phase-a-8xh100.md` (this repo) + the HF model card
  (AjelixAI/Ajelix-Fiber-130M) — the quality proof + the 11 published
  checkpoints.
- `SPEED.md`, `EXPORT.md`, `ANNEAL_PLAN.md` (this repo, `anneal-plan`
  branch) — the training/deployment engineering record.
