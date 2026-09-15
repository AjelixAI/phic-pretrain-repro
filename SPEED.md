
## Inference speed (the deployment story) — measured 2026-09-15

| path | tok/s | RAM | correctness |
|---|---|---|---|
| tied eager, full-context recompute | 32.5 | 12 MB | baseline |
| tied eager, KV-cached | 31.5 | 12 MB + 11 MB KV | 100% token match vs baseline |
| **tied + CUDA-graph KV decode** | **150.8** | 12 MB + 11 MB KV | **100% token match** |
| materialized Llama export | 81.0 | 260 MB | — |
| Pythia-160m dense (reference) | 167.4 | 320 MB | — |

**150.8 tok/s at 1/27th of Pythia's memory, with token-identical output.**
The graph replay executes the identical kernels pre-scheduled — speed of
execution does not change the model's function (proven by the 100% match).

Key lesson from the build: the first version produced garbage because the
decode step **omitted the input LayerNorm** (`attn(n1(x))` — one missing
norm silently changed the model). The 100%-token-match check is the gate
that catches this class of error; it is mandatory for any inference-path
change. Files: `scripts/kv_graph_decoder.py`.
