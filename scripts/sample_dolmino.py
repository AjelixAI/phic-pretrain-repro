#!/usr/bin/env python3
"""Sample dolmino-mix-1124 (OLMo 2 mid-training data) at 1.05B tokens.
Per-config targets (ANNEAL_PLAN.md §4, merged web share into dclm):
  dclm 68%, flan 8%, pes2o 8%, wiki 8%, stackexchange 4%, math 4%.
Tokenizer: OURS (EleutherAI/gpt-neox-20b), EOS between docs.
-> /root/phi/data_cache_dolmino_raw.pt  (then shuffle_cache.py -> _shuf.pt)
"""
import gzip, io, json, os
import multiprocessing as mp
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

TARGET = 1_050_000_000
OUT = "/root/phi/data_cache_dolmino_raw.pt"
REPO = "allenai/dolmino-mix-1124"
API = HfApi()
_all = API.list_repo_files(REPO, repo_type="dataset")
EXTS = (".zst", ".zstd", ".gz", ".json", ".jsonl")
FILES = [f for f in _all if f.startswith("data/") and f.endswith(EXTS)]

def config_of(f):
    return f.split("/")[1]

CONFIGS = sorted(set(config_of(f) for f in FILES))
CFG = {"dclm": (0.68, 832.6e9), "flan": (0.08, 3.8e9),
       "pes2o": (0.08, 52.6e9), "wiki": (0.08, 4.7e9),
       "stackexchange": (0.04, 4.6e9), "math": (0.04, 10.7e9)}

TOK = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
EOS = TOK.eos_token_id

def tok_texts(texts):
    encs = TOK(texts, add_special_tokens=False)["input_ids"]
    return [np.asarray(list(e) + [EOS], dtype=np.int32) for e in encs]

def worker_decode_tokenize(args):
    fname, max_tokens = args
    local = hf_hub_download(REPO, fname, repo_type="dataset")
    texts = []
    got = 0
    import zstandard as zstd
    if local.endswith((".zst", ".zstd")):
        fh = io.TextIOWrapper(io.BufferedReader(
            zstd.ZstdDecompressor().stream_reader(open(local, "rb"))),
            encoding="utf-8", errors="ignore")
    elif local.endswith(".gz"):
        fh = gzip.open(local, "rt", encoding="utf-8", errors="ignore")
    else:
        fh = open(local, "rt", encoding="utf-8", errors="ignore")
    with fh:
        for line in fh:
            if got >= max_tokens:
                break
            try:
                texts.append(json.loads(line).get("text", ""))
            except json.JSONDecodeError:
                continue
            got += len(texts[-1])
    os.remove(local)  # disk budget: delete after reading
    return tok_texts(texts), got

def main():
    ORDER = sorted(CONFIGS, key=lambda c: CFG.get(c, (0, 0))[1])
    print("config order:", ORDER, flush=True)
    parts, total, per_cfg = [], 0, {}
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
        print(f"{cfg}: {len(cfiles)} files, picking {len(picked)}, "
              f"target {cfg_target/1e6:.0f}M tok", flush=True)
        before = total
        with mp.Pool(min(24, len(picked))) as pool:
            for toks, got in pool.imap_unordered(worker_decode_tokenize,
                                                 [(f, per_file)
                                                  for f in picked]):
                parts.extend(toks)
                total += sum(t.size for t in toks)
        per_cfg[cfg] = total - before
        print(f"  {cfg}: {per_cfg[cfg]/1e6:.0f}M tok "
              f"({per_cfg[cfg]/total*100 if total else 0:.1f}%), "
              f"cumulative {total/1e9:.2f}B", flush=True)
    allv = np.concatenate(parts)[:TARGET]
    print("total tokens:", allv.size, flush=True)
    # ---- PREFLIGHT GATES ----
    n_eos = int((allv == EOS).sum())
    print(f"GATE EOS: {n_eos} EOS == {len(parts)} docs -> "
          f"{'PASS' if n_eos == len(parts) else 'FAIL'}", flush=True)
    mx = int(allv.max())
    print(f"GATE id-range: max id {mx} < 50304 -> "
          f"{'PASS' if mx < 50304 else 'FAIL'}", flush=True)
    for cfg, n in per_cfg.items():
        print(f"GATE proportion {cfg}: {n/1e6:.0f}M tok = "
              f"{n/1e9:.3f}B ({n/1e9/TARGET*1e9*100:.1f}%)", flush=True)
    torch.save(torch.from_numpy(allv), OUT)
    print("saved", OUT, flush=True)

if __name__ == "__main__":
    main()
