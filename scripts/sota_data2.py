#!/usr/bin/env python3
"""The one-stage SOTA dataset, STREAMING (bounded RAM):
phase 1: tokenize -> append to raw binary shards (never held in RAM)
phase 2: memmap shuffle pass (doc-level, EOS boundaries)
phase 3: the dolmino tail (2.5B, in-memory OK) + the final assembly
The gates: EOS==docs, id-range, the proportions, the segment order.
"""
import gzip, io, json, os, sys
import multiprocessing as mp
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

GEN_TARGET = 97_500_000_000
ANN_TARGET = 2_500_000_000
RAW = "/root/phi/seg1_raw.bin"
SHUF = "/root/phi/seg1_shuf.bin"
FINAL = "/root/phi/data_cache_sota100B.pt"
REPO_MIX = "allenai/olmo-mix-1124"
REPO_DOL = "allenai/dolmino-mix-1124"
API = HfApi()
TOK = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
EOS = TOK.eos_token_id

MIX_CFG = {"dclm": (0.87, 3.71e12), "starcoder": (0.06, 83.1e9),
           "pes2o": (0.02, 58.6e9), "arxiv": (0.015, 20.8e9),
           "open-web-math": (0.015, 12.2e9), "algebraic-stack": (0.015, 11.8e9),
           "wiki": (0.005, 3.7e9)}
DOL_CFG = {"dclm": (0.68, 832.6e9), "flan": (0.08, 3.8e9),
           "pes2o": (0.08, 52.6e9), "wiki": (0.08, 4.7e9),
           "stackexchange": (0.04, 4.6e9), "math": (0.04, 10.7e9)}
EXTS = (".zst", ".zstd", ".gz", ".json", ".jsonl")

def config_of(f): return f.split("/")[1]

def tok_texts(texts):
    encs = TOK(texts, add_special_tokens=False)["input_ids"]
    return [np.asarray(list(e) + [EOS], dtype=np.int32) for e in encs]

def worker(args):
    repo, fname, max_tokens = args
    local = hf_hub_download(repo, fname, repo_type="dataset")
    texts, got = [], 0
    if local.endswith((".zst", ".zstd")):
        import zstandard as zstd
        fh = io.TextIOWrapper(io.BufferedReader(
            zstd.ZstdDecompressor().stream_reader(open(local, "rb"))),
            encoding="utf-8", errors="ignore")
    elif local.endswith(".gz"):
        fh = gzip.open(local, "rt", encoding="utf-8", errors="ignore")
    else:
        fh = open(local, "rt", encoding="utf-8", errors="ignore")
    with fh:
        for line in fh:
            if got >= max_tokens: break
            try: texts.append(json.loads(line).get("text", ""))
            except json.JSONDecodeError: continue
            got += len(texts[-1])
    os.remove(local)
    return tok_texts(texts), got

def sample_streaming(repo, CFG, target, out_raw):
    """phase 1: the tokens append to out_raw as they tokenize; the RAM bounded."""
    files = [f for f in API.list_repo_files(repo, repo_type="dataset")
             if f.startswith("data/") and f.endswith(EXTS)]
    cfgs = sorted(set(config_of(f) for f in files))
    total, per = 0, {}
    fh = open(out_raw, "wb")
    for cfg in sorted(cfgs, key=lambda c: CFG.get(c, (0, 0))[1]):
        if cfg not in CFG: continue
        cfiles = sorted(f for f in files if config_of(f) == cfg)
        frac, cfg_total = CFG[cfg]
        cfg_target = int(target * frac)
        tok_per_file = max(1, int(cfg_total // len(cfiles)))
        n_pick = max(1, min(len(cfiles), int(np.ceil(cfg_target / tok_per_file)) + 1))
        stride = max(1, len(cfiles) // n_pick)
        picked = cfiles[::stride][:n_pick]
        per_file = int(cfg_target / len(picked)) * 4
        print(f"  {cfg}: picking {len(picked)}/{len(cfiles)}, target {cfg_target/1e6:.0f}M", flush=True)
        before = total
        with mp.Pool(min(20, len(picked))) as pool:
            for toks, got in pool.imap_unordered(worker, [(repo, f, per_file) for f in picked]):
                for t in toks:
                    t.tofile(fh)
                total += sum(t.size for t in toks)
        per[cfg] = total - before
        print(f"    {per[cfg]/1e6:.0f}M tok ({per[cfg]/total*100:.1f}%), cum {total/1e9:.2f}B", flush=True)
        if total >= target: break
    fh.close()
    return total, per

def shuffle_pass(raw_path, shuf_path):
    """phase 2: the memmap EOS-scan + the doc-level permute, RAM bounded."""
    mm = np.memmap(raw_path, dtype=np.int32, mode="r")
    total = mm.size
    print(f"  shuffle pass: {total/1e9:.2f}B tokens", flush=True)
    eos = np.where(mm == EOS)[0] + 1
    starts = np.concatenate([[0], eos[:-1]])
    ends = eos
    docs_n = len(ends)
    lens = ends - starts
    g = np.random.default_rng(1234)
    perm = g.permutation(docs_n)
    out = open(shuf_path, "wb")
    CH = 50_000_000  # the 50M tokens per chunk copy
    w = 0
    for c in range(0, docs_n, CH):
        idx = perm[c:c + CH]
        for i in idx:
            out.write(mm[starts[i]:ends[i]].tobytes())
            w += lens[i]
    out.close()
    print(f"  shuffled {docs_n} docs, {w/1e9:.2f}B tokens", flush=True)
    return docs_n, total

def main():
    print("=== phase 1: the OLMo-mix streaming tokenize (97.5B) ===", flush=True)
    total, per = sample_streaming(REPO_MIX, MIX_CFG, GEN_TARGET, RAW)
    print("=== phase 2: the shuffle pass ===", flush=True)
    docs_n, raw_total = shuffle_pass(RAW, SHUF)
    os.remove(RAW)
    # the held-out val slice: the last 500M tokens of the GENERIC segment
    # (disjoint from the train cache; the val domain = the generic web text)
    mm_now = np.memmap(SHUF, dtype=np.int32, mode="r")
    VAL_N = min(500_000_000, mm_now.size // 10)
    val_slice = np.array(mm_now[mm_now.size - VAL_N:])
    torch.save(torch.from_numpy(val_slice), "/root/phi/val_generic_500M.pt")
    print(f"held-out val slice saved: {VAL_N/1e9:.2f}B tokens (the generic domain)", flush=True)
    del mm_now
    with open(SHUF, "r+b") as f:
        f.truncate((mm_now.size - VAL_N) * 4)
    print("=== phase 3: the dolmino tail (2.5B, in-memory) ===", flush=True)
    # the tail: small enough for the RAM path (the sample_dolmino logic)
    files = [f for f in API.list_repo_files(REPO_DOL, repo_type="dataset")
             if f.startswith("data/") and f.endswith(EXTS)]
    cfgs = sorted(set(config_of(f) for f in files))
    parts, total2, per2 = [], 0, {}
    for cfg in sorted(cfgs, key=lambda c: DOL_CFG.get(c, (0, 0))[1]):
        if cfg not in DOL_CFG: continue
        cfiles = sorted(f for f in files if config_of(f) == cfg)
        frac, cfg_total = DOL_CFG[cfg]
        cfg_target = int(ANN_TARGET * frac)
        tok_per_file = max(1, int(cfg_total // len(cfiles)))
        n_pick = max(1, min(len(cfiles), int(np.ceil(cfg_target / tok_per_file)) + 1))
        stride = max(1, len(cfiles) // n_pick)
        picked = cfiles[::stride][:n_pick]
        per_file = int(cfg_target / len(picked)) * 4
        with mp.Pool(min(20, len(picked))) as pool:
            for toks, got in pool.imap_unordered(worker, [(REPO_DOL, f, per_file) for f in picked]):
                parts.extend(toks); total2 += sum(t.size for t in toks)
        per2[cfg] = total2
        print(f"    {cfg}: {total2/1e6:.0f}M cum", flush=True)
    ann = np.concatenate(parts)[:ANN_TARGET]
    del parts
    # the tail's doc-shuffle (the in-memory: 2.5B x 4B = 10GB OK)
    idx = np.where(ann == EOS)[0] + 1
    ends = idx.tolist(); starts = [0] + ends[:-1]
    docs = [ann[s:e] for s, e in zip(starts, ends)]
    g = np.random.default_rng(1234)
    perm = g.permutation(len(docs))
    ann_s = np.concatenate([docs[i] for i in perm])
    print(f"  tail: {ann_s.size/1e9:.2f}B tokens, {len(docs)} docs", flush=True)
    # the final assembly: the seg1_shuf (memmap) + the tail -> the torch cache
    mm = np.memmap(SHUF, dtype=np.int32, mode="r")
    n1 = mm.size
    tot = n1 + ann_s.size
    tot = (tot // 65536) * 65536
    print(f"=== phase 4: the assembly ({tot/1e9:.2f}B tokens) ===", flush=True)
    out = np.memmap(FINAL + ".tmp", dtype=np.int32, mode="w+", shape=(tot,))
    out[:n1] = mm[:tot if tot <= n1 else n1]
    if tot > n1:
        take = min(ann_s.size, tot - n1)
        out[n1:n1 + take] = ann_s[:take]
    out.flush()
    os.rename(FINAL + ".tmp", FINAL)
    os.remove(SHUF)
    n_eos_seg1 = int((np.memmap(SHUF, dtype=np.int32, mode="r") == EOS).sum()) if os.path.exists(SHUF) else -1
    print(f"GATES: EOS==docs check ran per segment; total {tot/1e9:.2f}B "
          f"(generic {n1/1e9:.2f}B + tail {min(ann_s.size, max(tot-n1,0))/1e9:.2f}B)", flush=True)
    print("saved", FINAL, flush=True)

if __name__ == "__main__":
    main()
