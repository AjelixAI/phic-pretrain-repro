# Fine-tuning the 1B-dense model — the paths, prepared before the base run completes

## 1. Adding custom tokens (the chat markers, the special tokens)

| step | procedure |
|---|---|
| the resize | the tied emb/head table: 100,352 × 2048 → (100,352 + N) × 2048 |
| the init | the new rows: the mean of the existing embedding vectors (the standard practice); the head extends automatically (the tied design) |
| the architecture | **zero changes** — the tied blocks operate on d-dimensional vectors, never see the vocabulary |
| the training | the new rows train during SFT (the standard) |

## 2. The chat template

The tokenizer's `apply_chat_template` attribute — the data-level formatting, no architecture interaction. The SFT data uses the templated prompts/responses; the model learns the format from the training.

## 3. The two fine-tuning paths

| path | how | pros |
|---|---|---|
| **the tied-native** (our trainer) | the same `pretrain_tb_fast.py` with the loss masked to the response tokens (~50 lines: the labels where the assistant speaks, −100 where the prompt) + the FP8-build + the FP8 GEMMs — the proven infrastructure carries over | the efficient: the 293M actual params, the tied form, the FP8 speed, the checkpoint system, the two-vals — all the pre-flight machinery works unchanged |
| **the Llama export** (the standard tools) | the verified export (`export_llama.py`, both gates) → `LlamaForCausalLM` → the HF TRL / axolotl / any SFT framework works out of the box | the ecosystem: the battle-tested tools, the community recipes, no custom code; the export's rope base and tokenizer are now parameterized (the fossil bugs fixed) |

## 4. The loss masking (the SFT's only code gap)

~50 lines in the trainer: the labels = the input ids where the assistant responds; −100 (the ignore index) where the user prompt or the system text. The same pattern every SFT implementation uses.

## 5. The optics-native fine-tuning — the unique advantage

The SFT trains the **corrections** (the blocks, U, V) — the same parameters the optical engine consumes. The fine-tuned model's corrections plug into the same tied-blocks architecture: **the instruct-tuned model is optics-compatible with zero conversion.** A dense model's fine-tuning would require re-factoring; ours doesn't.

## 6. The instruct ladder (the plan)

| stage | data | tool |
|---|---|---|
| SFT | Tulu-3-style open recipe (the provenance-clean) | the tied-native trainer or the export + the TRL |
| DPO | the Tulu-3 DPO mix / UltraFeedback | the TRL on the export |
| the result | `Ajelix-Fiber-1B-Instruct` | the card + the evals |
