#!/usr/bin/env python3
"""The one-stage SOTA run dataset: OLMo-mix (97.5B) + the Dolmino-style
annealing tail (2.5B), concatenated IN ORDER (the generic first, the
curated tail last), each segment document-shuffled internally, one stream.
uint16 cache (the vocab 50,277 < 65,535). -> data_cache_sota100B.pt
The WSD schedule's decay window lands on the Dolmino tail.
"""
import gzip, io, json, os, sys
import multiprocessing as mp
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

GEN_TARGET = 97_500_000_000
ANN_TARGET = 2_500_000_000
OUT = "/root/phi/data_cache_sota100B.pt"
REPO_MIX = "allenai/olmo-mix-1124"
REPO_DOL = "allenai/dolmino-mix-1124"
API = HfApi()
TOK = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")  # the OLMo 2 tokenizer, 100,352 vocab, Apache 2.0
EOS = TOK.eos_token_id

# the OLMo-2 paper Table 1 proportions (of the pretrain mix)
# the math/code/science UPSAMPLED (10.5% vs OLMo's 4%) - the llama-3-style
# reasoning weighting, since our one-stage run has no math-heavy mid-train
MIX_CFG = {"dclm": (0.87, 3.71e12), "starcoder": (0.06, 83.1e9),
           "pes2o": (0.02, 58.6e9), "arxiv": (0.015, 20.8e9),
           "open-web-math": (0.015, 12.2e9), "algebraic-stack": (0.015, 11.8e9),
           "wiki": (0.005, 3.7e9)}
# the Dolmino-style annealing composition (ANNEAL_PLAN §4)
DOL_CFG = {"dclm": (0.68, 832.6e9), "flan": (0.08, 3.8e9),
           "pes2o": (0.08, 52.6e9), "wiki": (0.08, 4.7e9),
           "stackexchange": (0.04, 4.6e9), "math": (0.04, 10.7e9)}

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

def sample_segment(repo, CFG, target, seed_shift=0):
    files = [f for f in API.list_repo_files(repo, repo_type="dataset")
             if f.startswith("data/") and f.endswith(EXTS)]
    cfgs = sorted(set(config_of(f) for f in files))
    parts, total, per = [], 0, {}
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
        with mp.Pool(min(24, len(picked))) as pool:
            for toks, got in pool.imap_unordered(worker,
                    [(repo, f, per_file) for f in picked]):
                parts.extend(toks); total += sum(t.size for t in toks)
        per[cfg] = total - before
        print(f"    {per[cfg]/1e6:.0f}M tok, cumulative {total/1e9:.2f}B", flush=True)
    allv = np.concatenate(parts)[:target] if total >= target else np.concatenate(parts)
    print(f"  segment: {allv.size/1e9:.2f}B tokens (target {target/1e9:.2f}B)", flush=True)
    return allv, per

EXTS = (".zst", ".zstd", ".gz", ".json", ".jsonl")

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gatedata":
        global GEN_TARGET, OUT
        GEN_TARGET = 1_200_000_000
        OUT = "/root/phi/data_cache_gatedata.pt"
        global MIX_CFG
        MIX_CFG = {"dclm": (0.94, 3.71e12), "starcoder": (0.06, 83.1e9)}
        gen, _ = sample_segment(REPO_MIX, MIX_CFG, GEN_TARGET)
        torch.save(torch.from_numpy(gen.astype(np.int32)), OUT)
        print("saved", OUT, gen.size, flush=True)
        return
    print("=== segment 1: OLMo-mix generic (97.5B) ===", flush=True)
    gen, per1 = sample_segment(REPO_MIX, MIX_CFG, GEN_TARGET)
    print("=== segment 2: Dolmino annealing tail (2.5B) ===", flush=True)
    ann, per2 = sample_segment(REPO_DOL, DOL_CFG, ANN_TARGET)
    # document-level shuffle WITHIN each segment (the preflight rule)
    def shuffle(v):
        idx = (v == EOS).nonzero()[0]
        ends = (idx + 1).tolist(); starts = [0] + ends[:-1]
        docs = [v[s:e] for s, e in zip(starts, ends)]
        g = np.random.default_rng(1234)
        perm = g.permutation(len(docs))
        return np.concatenate([docs[i] for i in perm]), len(docs)
    gen_s, n1 = shuffle(gen)
    ann_s, n2 = shuffle(ann)
    stream = np.concatenate([gen_s, ann_s])
    stream = stream[:(stream.size // 65536) * 65536]
    n_eos = int((stream == EOS).sum())
    print(f"GATE EOS: {n_eos} == docs {n1 + n2} -> {'PASS' if n_eos == n1 + n2 else 'FAIL'}", flush=True)
    print(f"GATE id-range: max {stream.max()} < 100352 -> "
          f"{'PASS' if stream.max() < 100352 else 'FAIL'}", flush=True)
    print(f"total: {stream.size/1e9:.2f}B tokens "
          f"(generic {gen_s.size/1e9:.2f}B + anneal {ann_s.size/1e9:.2f}B)", flush=True)
    torch.save(torch.from_numpy(stream.astype(np.int32)), OUT)  # the vocab 100352 > uint16
    print("saved", OUT, flush=True)

if __name__ == "__main__":
    main()
