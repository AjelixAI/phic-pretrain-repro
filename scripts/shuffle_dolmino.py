#!/usr/bin/env python3
"""Document-level shuffle of the dolmino cache (PREFLIGHT: source-blocked
cache is invalid — the Phase-B standing warning). Verifies mixing, not just
asserting it."""
import torch, collections
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "/root/phi/data_cache_dolmino_raw.pt"
DST = sys.argv[2] if len(sys.argv) > 2 else "/root/phi/data_cache_dolmino_shuf.pt"
t = torch.load(SRC, weights_only=True, mmap=True)
EOS = 0
idx = (t == EOS).nonzero().flatten()
ends = (idx + 1).tolist()
starts = [0] + ends[:-1]
lens = [e - s for s, e in zip(starts, ends)]
docs = [torch.as_tensor(t[s:e].clone()) for s, e in zip(starts, ends)]
print(f"docs: {len(docs)} == EOS count {len(idx)}", flush=True)
g = torch.Generator().manual_seed(1234)
perm = torch.randperm(len(docs), generator=g)
docs = [docs[i] for i in perm]
flat = torch.cat(docs)
flat = flat[: (flat.numel() // 65536) * 65536]
torch.save(flat.to(torch.int32), DST)
print(f"saved {DST}: {flat.numel()/1e9:.3f}B tokens", flush=True)
# --- mixing verification (command output, not assertion) ---
n = len(lens)
head = [lens[i] for i in perm[:20]]
print("first 20 shuffled doc lengths:", head, flush=True)
# blockiness test: longest run of similar-length docs (len within 2x of the
# run's first member) vs the shuffled expectation
def longest_similar_run(seq):
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if b <= 2 * max(a, 1) else 1
        best = max(best, run)
    return best
lshuf = [lens[i] for i in perm]
raw_run = longest_similar_run(lens)
shuf_run = longest_similar_run(lshuf)
print(f"longest similar-length run: raw {raw_run} -> shuffled {shuf_run} "
      f"(global shuffle expected: near log(n)/log(3))", flush=True)
print(f"doc length median {sorted(lens)[n//2]}, p90 {sorted(lens)[int(n*0.9)]}",
      flush=True)
