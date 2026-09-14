#!/usr/bin/env python3
"""Pre-tokenize FineWeb-Edu sample-10BT -> /root/phi/data_cache.pt (int32 1-D)."""
import numpy as np
import torch
from transformers import AutoTokenizer
import datasets

TARGET = 3_200_000_000   # 3.2B tokens (Chinchilla + margin)
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
ds = datasets.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                           split="train")

def tok_fn(batch):
    return {"ids": tok(batch["text"], add_special_tokens=False)["input_ids"]}

tds = ds.map(tok_fn, batched=True, batch_size=256, num_proc=32,
             remove_columns=ds.column_names)
print("tokenized rows:", len(tds), flush=True)

chunks, total = [], 0
for ex in tds["ids"]:
    a = np.asarray(ex, dtype=np.int32)
    chunks.append(a); total += a.size
    if total >= TARGET:
        break
allv = np.concatenate(chunks)[:TARGET]
print(f"total tokens: {allv.size}", flush=True)
t = torch.from_numpy(allv)
torch.save(t, "/root/phi/data_cache.pt")
print("saved /root/phi/data_cache.pt", flush=True)
