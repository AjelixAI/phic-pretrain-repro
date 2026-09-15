
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

## Inference stack, measured (2026-09-15, batch-1 greedy, RTX PRO 6000)

| path | tok/s | runtime RAM (params+KV) | verification |
|---|---|---|---|
| eager fast tied (Mf buffers, 4 GEMMs/instance) | 150.8 | ~320 MB | reference |
| **P-mode** (Mf@Wt+residual merged at load, 1 GEMM) | **320.4** | ~260 MB | 100% token + top-1 match |
| **butterfly mode** (Triton fused kernels, tied params, no Mf) | **262.3** | **~129 MB** | 100% token + top-1 match |

All three paths verified in `scripts/stack_verify.py`: 30-token greedy decode,
token-match 100.0%, top-1 agreement 100.0%, median relative logit error
0.61% (bf16 reassociation level).

**CORRECTION of earlier claims:** the checkpoint is **106.7 MB** (bf16,
deduplicated; the tied emb/head pair at 50304 vocab is 49 MB of it), not
"12 MB" as stated in earlier notes. The honest savings vs the dense
equivalent (260 MB Llama export): **2.4x smaller file, 2x less runtime RAM**
(butterfly mode). The bottleneck decomposition and the 106.7 MB accounting
are in `scripts/stack_verify.py` outputs and this file.

Kernel files: `scripts/triton_kernels.py` (bfly_s1, bfly_s2_bd_res,
gemv_p), `scripts/p_merge_decode.py` (P-mode), `scripts/bfly_decode.py`
(butterfly mode).
