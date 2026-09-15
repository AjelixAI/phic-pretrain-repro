#!/usr/bin/env python3
"""Sample OLMo-mix-1124 proportionally at 3.2B tokens.
Per-config targets from OLMo-2 paper Table 1 (of 3.90T):
  dclm 3.71T, starcoder ~83.1B, pes2o 58.6B, arxiv 20.8B,
  open-web-math 12.2B, algebraic-stack 11.8B, wiki 3.7B.
Tokenizer: OURS (EleutherAI/gpt-neox-20b), EOS between docs.
-> /root/phi/data_cache_olmomix.pt
"""
import gzip
import io
import json
import multiprocessing as mp
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

TARGET = 30_400_000_000
OUT = "/root/phi/data_cache_olmomix.pt"
API = HfApi()
_all = API.list_repo_files("allenai/olmo-mix-1124", repo_type="dataset")
FILES = [f for f in _all if f.startswith("data/") and (f.endswith(".jsonl.zstd") or f.endswith(".json.gz"))]

def config_of(f):
    return f.split("/")[1]

CONFIGS = sorted(set(config_of(f) for f in FILES))
# per-config: fraction of 3.2B (paper proportions) + total config tokens (paper Table 1)
CFG = {"dclm": (0.9513, 3.71e12), "starcoder": (0.0260, 83.1e9),
       "pes2o": (0.0183, 58.6e9), "arxiv": (0.0065, 20.8e9),
       "open-web-math": (0.0038, 12.2e9), "algebraic-stack": (0.0037, 11.8e9),
       "wiki": (0.0009, 3.7e9)}

TOK = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
EOS = TOK.eos_token_id

def tok_texts(texts):
    encs = TOK(texts, add_special_tokens=False)["input_ids"]  # fast rust batch
    return [np.asarray(list(e) + [EOS], dtype=np.int32) for e in encs]

def worker_decode_tokenize(args):
    fname, max_tokens = args
    local = hf_hub_download("allenai/olmo-mix-1124", fname, repo_type="dataset")
    texts = []
    got = 0
    if local.endswith(".zstd"):
        import zstandard as zstd
        with open(local, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as r:
                buf = io.TextIOWrapper(io.BufferedReader(r), encoding="utf-8", errors="ignore")
                for line in buf:
                    if got >= max_tokens:
                        break
                    try:
                        texts.append(json.loads(line).get("text", ""))
                    except json.JSONDecodeError:
                        continue
                    got += len(texts[-1])
    else:
        with gzip.open(local, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if got >= max_tokens:
                    break
                try:
                    texts.append(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    continue
                got += len(texts[-1])
    return tok_texts(texts), got

def main():
    import zstandard  # noqa: F401  (fail fast if missing)
    # small configs FIRST so dclm (the bulk) fills the remainder exactly;
    # otherwise the global target is hit before code/math/science are sampled
    ORDER = sorted(CONFIGS, key=lambda c: CFG.get(c, (0, 0))[1])
    print("config order:", ORDER, flush=True)
    parts, total = [], 0
    for cfg in ORDER:
        cfiles = sorted(f for f in FILES if config_of(f) == cfg)
        frac, cfg_total_tokens = CFG[cfg]
        cfg_target = int(TARGET * frac)
        tok_per_file = max(1, cfg_total_tokens // len(cfiles))
        n_pick = max(1, min(len(cfiles),
                     int(np.ceil(cfg_target / tok_per_file)) + 1))
        stride = max(1, len(cfiles) // n_pick)
        picked = cfiles[::stride][:n_pick]
        per_file = int(cfg_target / len(picked)) * 4  # chars ~= 4x tokens
        print(f"{cfg}: {len(cfiles)} files, picking {len(picked)}, target {cfg_target/1e6:.0f}M tok", flush=True)
        with mp.Pool(min(24, len(picked))) as pool:
            for toks, got in pool.imap_unordered(worker_decode_tokenize,
                                                 [(f, per_file) for f in picked]):
                parts.extend(toks)
                total += sum(t.size for t in toks)
        print(f"  cumulative tokens: {total/1e9:.2f}B", flush=True)
        if total >= TARGET:
            break
    allv = np.concatenate(parts)[:TARGET]
    print("total tokens:", allv.size, flush=True)
    torch.save(torch.from_numpy(allv), OUT)
    print("saved", OUT, flush=True)

if __name__ == "__main__":
    main()
