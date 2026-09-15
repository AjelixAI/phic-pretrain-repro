#!/usr/bin/env python3
"""Document-level shuffle of the OLMo-mix cache (EOS-id-0 boundaries)."""
import torch
SRC = "/root/phi/data_cache_olmomix.pt"
DST = "/root/phi/data_cache_olmomix_shuf.pt"
t = torch.load(SRC, weights_only=True, mmap=True)
EOS = 0
idx = (t == EOS).nonzero().flatten()          # positions of EOS
ends = (idx + 1).tolist()
starts = [0] + ends[:-1]
docs = [torch.as_tensor(t[s:e].clone()) for s, e in zip(starts, ends)]
print(f"docs: {len(docs)}", flush=True)
g = torch.Generator().manual_seed(1234)
perm = torch.randperm(len(docs), generator=g)
docs = [docs[i] for i in perm]
flat = torch.cat(docs)
flat = flat[: (flat.numel() // 65536) * 65536]
torch.save(flat.to(torch.int32), DST)
print(f"saved {DST}: {flat.numel()/1e9:.2f}B tokens", flush=True)
