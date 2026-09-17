# The 7B-class model plan — Ajelix-Fiber-7B (the committed pre-launch specification)

## The model config (the exact math, validated in the notebook)

| | |
|---|---|
| config | **d=3840, L=31, ffn=13,440**, block 32, rank 64, stages 2 |
| tokenizer | OLMo 2, 100,352 vocab (the same as the 1B run) |
| RoPE | base 10,000 @ the 4,096 context (the 8k revisit at the config time) |
| **dense-equiv** | **6.74B** |
| **actual in-memory** | **0.727B** (the compression **9.3×**) |
| the stored file | ~0.61B params → **~1.2 GB bf16** |
| FLOPs/token | 43.0 GFLOP |
| the training tokens @90 TPP | ~600B |
| the total compute | ~490 ZFLOP |

## The hyperparameters (the same recipe, the 7B-tuned schedule)

| | |
|---|---|
| optimizer | AdamW β 0.9/0.95, wd 0.1, clip 1.0 — identical to the 1B run |
| LR | 3e-4 peak, warmup ~2,000 steps, **the WSD: the stable → the linear decay-to-zero over the final ~10%** |
| schedule length | ~6,100 steps at the 98.3M tokens/step (the bs 24 × 4,096 × 4 ranks × ...: the set at the launch) |
| precision | the FP8 P-build (the rowwise recipe) + the bf16 x@P GEMMs (the same measured hybrid; the FP8 GEMMs revisit on Hopper: the native FP8 tensor cores) |

## The data plan (the CHANGED part: a fresh carve, not the same files)

1. **~600B fresh tokens from the OLMo pool** (the ~3T available): the same proportions (the dclm ~87%, the starcoder ~6%, the math ~3%, the academic ~3.5%, the wiki ~0.5%), the reasoning upsample re-tuned at the 7B's capacity
2. the same streaming builder (`sota_data2.py`), the same gated shuffle, the fresh val carves (the generic + the anneal-domain, the disjoint by construction)
3. **NOT** the 1B run's 87.3B carve: at the 600B tokens the reuse would epoch it ~7× (the degradation territory; the TinyLlama 3T dip is the cautionary tale); the OLMo pool's 3T unique covers the 600B at <1 epoch
4. the reason upsample re-verification: the two-val harness (the general + the anneal-domain) at the first val checkpoint

## The hardware plan (the priced options, Sept 2026)

| option | config | days | cost |
|---|---|---|---|
| the budget pick | 4× A100 80GB PCIe (~$0.63/hr) | ~165 | **~$2,200** (the 5× 29-day renewal ladder) |
| **the default pick** | **4× H100 SXM (~$4.12/hr)** | **~165→ see note** | **~$3,700-6,000** |
| the interruptible | 8× RTX 4090 (~$0.20/GPU-hr) | ~250 | ~$2,100 (the watchdog eats the restarts) |

**Note**: the 7B at the 90 TPP is 6.6× the 1B's job (the 600B vs the 87.3B tokens at the 6.5× the FLOPs/token); the H100's effective throughput (~2 PFLOPS for 4) makes it ~40-50 days at the 45-50% MFU with the Hopper-native FP8 GEMMs.

## The gating experiment (BEFORE the money commits)

**The 1B architecture-tax head-to-head (~10 GPU-hours)**: the dense 1B vs the tied 1B, the same data, the same schedule, the same harness.
- if the tied's loss tracks the dense's within ~5%: the 7B commit is justified
- if not: the rank/block sweep at the 1B first (the hours, not the months)

## The pre-flight checklist (the same as the 1B run, plus)

- [ ] the 7B config's FP8-build gradient test (the d=3840's block-diagonal extraction)
- [ ] the config-math verification (the notebook's solve at the 7B target)
- [ ] the fresh cache build + the gates (the proportions, the EOS, the shuffle)
- [ ] the memory fit: the ~26 GB/GPU + the activations at the bs 24 (the 80GB cards: the trivially fits)
- [ ] the watchdog + the storage-box backup retargeting
- [ ] the graphs gate re-run at the 7B config (the Hopper: the different divergence profile)

## The run book

1. the head-to-head verdict
2. the data build (~4-6h)
3. the pre-flight gates
4. the launch: `pretrain_tb_fast.py --mode tied --bs 24 --seq 4096 --d 3840 --layers 31 --ffn 13440 --block 32 --rank 64 --vocab 100352 --rope-base 10000 --lr 3e-4 --warmup 2000 --steps <computed> --decay-start <93%> --project phic-fiber-7B --cache <fresh> --chunked-ce 8192 --save-every 700 --ckpt-every 4 --tag-suffix=-7B --force-save`
5. the monitoring: the two-vals every 500, the milestones every 2,800 (the patched cadence), the box sync every 10 min
