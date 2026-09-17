# The optics interface — what transfers, what gets re-fit (the deployment specification)

## The principle

The tied-blocks architecture deliberately splits **the physics (the frozen seed-derived butterfly basis) from the learning (the trained corrections, residuals, tied table)**. The optical engine consumes the physics as hardware structure and the learning as small trained deltas. A dense model would need its n² weights physically materialized; ours needs only the O(n) trained deltas encoded onto hardware whose fixed structure is the basis.

## What transfers with zero retraining

| artifact | why it carries over |
|---|---|
| the trained model (the corrections Wt, the residuals U/V, the tied table) | a digital artifact — the SFT/instruct-tuned model is the same weights |
| the butterfly basis | a **seed**, not data — regenerable by any process, including the optical alignment procedure |
| the trained capability (the 87B tokens of learning) | lives in the ~18M correction + residual params — exactly what the optical engine consumes |

## The deployment path (instead of retraining)

### 1. The calibration (always, minutes)

The real optical basis ≠ the ideal seed-generated Mf (the fabrication tolerances, the thermal drift, the alignment). Characterize the real operator: the known input patterns → the recorded outputs → the measured transfer matrix (the speckle/ELM method from the project's founding survey). One measurement per hardware unit.

### 2. The correction re-fit (only if the real basis differs materially, hours)

Re-run the P-composition with `Mf_real @ Wt` and fine-tune **only the corrections + the residuals** (~18M params for the 1B model, the hours on one GPU). The body, the tied table, the capability: untouched. The re-fit is small by construction: the corrections start from their trained values and absorb only the ideal-vs-real physics difference — a perturbation, not a re-learn.

### 3. The noise-aware fine-tune (the analog deployment, hours)

The optical compute is analog (~4-6 effective bits, the photon noise, the drift). Add the measured noise model (the quantization + the readout noise) into the training forward pass; fine-tune the corrections a few hundred steps. The precedent: the FP8-build runs at the 8-bit-ish numerics at a +0.03% loss cost (the validated recipe); the 5-bit regime costs more and the correction/residual structure is where it's absorbed.

## The scenario table

| scenario | what gets retrained | cost |
|---|---|---|
| the pure digital model (today) | nothing | — |
| the optics added, the ideal-ish physics | the calibration only | the minutes of measurement |
| the optics with the real (imperfect) basis | the corrections + the residuals re-fit | the hours, 1 GPU, ~18M params |
| the analog noise (the 4-6 bit) | the corrections + the residuals, the noise-aware pass | the hours |
| **the full pretraining re-run** | **never required by the optics path** | — |

## The saturation context (why the model survives the token scaling to the optics era)

The small-model saturation (the Pythia 70M/160M's late-training loss *regression*) is a spectral pathology of the softmax bottleneck: the hidden dimension d < ~1,000 cannot map to the high-rank contextual distribution (the ~10k-15k rank needed; Godey et al. 2024, arXiv:2404.07647). **Our d=2048 sits 2× above the threshold, and the tied head's rank (≤2048) equals a dense model's — the architecture adds no extra bottleneck.** The tied-blocks compression lives in the transformer body; if a body-capacity wall exists it manifests as the *slowing* loss progress (the underfitting), not the regression — the monitored instrument is the milestone-to-milestone loss slope, already part of the run book.

## The architecture upgrade path

The same property enables the hardware iteration: a better diffuser, a different fiber, a new optic — each is a hours-long correction re-fit against the re-measured basis, never a weeks-long retrain. The model's intelligence is detachable from the physics by design.

## The implementation hooks (the repo)

- the correction re-fit: the trainer's `--resume` + the real-basis Mf swap (the `BflyBasis`'s stages ← the measured matrices; the seed path stays for the ideal case)
- the noise-aware pass: the `_FP8Build`'s quantize step + the added noise injection
- the calibration data: the known-pattern send/record harness (the eval scripts' pattern)
