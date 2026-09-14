# Pre-flight checklist — every run, no exceptions

Each item verified with a COMMAND + OUTPUT, not by assertion.

## Data
- [ ] EOS (id 0) between all documents: count EOS in cache == doc count
- [ ] **Documents globally shuffled** (source-blocked cache = invalid):
      verify no long same-source runs — e.g., batch losses must not oscillate
      by >1 nat between adjacent batches in the first 100 steps
- [ ] Source proportions match the target recipe (per-config token counts)
- [ ] Cache ids within tokenizer range: max(id) < len(tokenizer)
- [ ] Deduplication status known (upstream dataset property, documented)

## Training
- [ ] Step-0 loss == ln(vocab) ± tolerance (sanity gate armed, aborts on fail)
- [ ] Tokens/param disclosed; epochs = 1 unless documented otherwise
- [ ] Val slice disjoint from train batches (check index arithmetic incl. wraparound)
- [ ] LR justified (published recipe or swept); warmup/decay fractions stated

## Ops
- [ ] Cache load strategy fits node RAM (mmap when cache > 20% RAM)
- [ ] Periodic checkpoints for runs > 4 h
- [ ] Port 29500 free / previous torchrun processes verified dead before launch
      (verify with ps, not assumed from pkill exit code)
- [ ] W&B project/tag distinguishable per run

## Honesty
- [ ] Every "verified" claim above backed by a command output in the run record
