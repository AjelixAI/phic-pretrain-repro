# Exporting the tied-blocks model to standard HF format (Llama-compatible)

Why: lm-eval (and the wider ecosystem) cannot load the custom tied-blocks
architecture. The export materializes it into a **LlamaForCausalLM** so it
can run through the standard harness for apples-to-apples benchmark
comparisons. The export is verified against the original model before use.

## Files

- `scripts/export_llama.py` — the exporter (includes the verification step)
- Output: `/root/phi/export_llama` — a stock Llama HF checkpoint
  (config: hidden 512, 22 layers, 8 q heads / 2 kv heads, intermediate
  1792, vocab 50304, context 2048, tied embeddings)

## How the materialization works (and why it is exact)

Each `TiedLinear` is exactly linear (pad → frozen butterfly bmm chain →
trained block-diagonal bmm → slice → rank-r residual; no nonlinearities),
so it has an exact dense equivalent:

```python
eye = torch.eye(tl.d_in, dtype=bf16)
W_out_in = tl.forward(eye.unsqueeze(0))[0].T   # run the ORIGINAL forward
```

**Rule: always materialize by running the original code path on the
identity matrix — never hand-derive the layouts.** The identity trick was
re-derived wrong twice by hand before this rule was adopted; the
`forward(eye)` form is self-verifying.

## The gotchas (each one cost debugging time — do not repeat)

1. **`rope_theta=64`, NOT 10000.** Our rotary uses the head_dim (64) as its
   base: `inv = head_dim ** (-arange(0,64,2)/64)`. The llama default
   (10000) silently produces a model that agrees only ~53% on next-token
   prediction while loading with zero missing/unexpected keys — the most
   deceptive failure possible. Symptom: per-position agreement OK at
   positions 0–1, diverging from position ~2 onward (small angles coincide
   for any base). Fix: `LlamaConfig(rope_theta=64)`.
2. **Rotary row permutation** (interleaved → half-split): q/k rows must be
   reordered per head `[0,2,4,...,62, 1,3,...,63]` (v is NOT permuted — no
   rope on v). Verified exact by unit test (`rope_test.py`, max|d| = 0.0).
3. **`rms_norm_eps=0.0078125`**: our `nn.RMSNorm(eps=None)` resolves to
   `finfo(bf16).eps` = 2⁻⁷ — set the same value in the llama config.
4. **Tied head**: `tie_word_embeddings=True` and omit `lm_head.weight`
   from the state dict (llama ties it).

## Mandatory verification before using an export

Run BOTH gates (in `export_llama.py`; keep them in any future exporter):

1. **Logit drift**: same 6-token input through both implementations —
   relative max diff must be < ~3% (22-layer bf16 accumulation; measured
   2.3% after the rope fix, 7.7% before it).
2. **Argmax agreement**: ≥16k next-token predictions on real data —
   agreement must be ≥ ~95% (measured 96.7% after the rope fix; 52.8%
   before it — near-ties flip from bf16 rounding; anything below ~90%
   means a structural bug).

If either gate fails, do NOT use the export for benchmarking — a divergent
export produces plausible-looking but wrong numbers (this is exactly what
happened the first time).

## The verification cascade that found the rope bug (for future reference)

generations coherent → manual lambada sample contradicted the mass eval →
100-example manual check isolated the slice bias → per-position agreement
showed divergence from position 2 → rotary unit test exact → frequency-base
comparison found the 64-vs-10000 mismatch. Each step eliminated a class of
causes; the pattern generalizes: **when an export "loads perfectly" but
behaves differently, diff position-by-position and check numeric
constants (bases, epsilons, scales) against the original implementation.**
