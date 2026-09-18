# AFS — Associative Factor Store
## The spec (2026-09-18)
Dense-1B knowledge (~707M FFN params, ~177MB at 2bit/param) at ~200M-dense
per-token compute and ~160x less traffic; optics-portable.

## Mechanism
The FFN becomes a retrieval over a persistent factorized table:
  scores = softmax(x @ K^T / sqrt(d))   # K: [E, d], trained keys
  w      = top-soft-k(scores)           # continuous (train) / hard ST (serve)
  y      = sum_k w_k * Row_k(x)         # rows: factorized FFN operators
Knowledge capacity = E * per-row corrections (2bit/param over all rows;
Allen-Zhu 2404.05405 covers MoE-style capacity counting).
Per-token traffic = k selected rows only — CONSTANT vs E. That is the
bandwidth solve: traffic is a function of retrieval width, not knowledge.

## Rows: v1 dense (capacity baseline), v2 factorized (optics tax measurement)
v1: each row = a dense mini-FFN (gate/up/down). Cleanest bits/param baseline.
v2: each row = tied-blocks factorized FFN sharing the layer's frozen basis
    (optics-renderable; measures the structured tax vs v1 directly).

## Probe protocol (Allen-Zhu bioS-style)
Synthetic biographies: N persons x M attributes, statements in train text,
query-answer eval. Capacity metric: recall accuracy -> stored bits ->
bits/trained-param. Arms:
  (1) dense-FFN param-matched  (high compute, capacity reference)
  (2) AFS small-E              (low compute, low capacity)
  (3) AFS large-E              (low compute, HIGH capacity) <- the claim:
      knowledge of (1) at the compute of (2).
Also measured: per-row exposure counts (routing consistency risk),
soft-vs-hard retrieval gap.

## Risks tracked
- structured tax inside rows (Wei 2406.16450) -> v1 vs v2 arm.
- routing consistency / exposure compounding -> per-row exposure telemetry.
- soft retrieval costs full-table reads on GPU -> hard top-k + gather for serving.
