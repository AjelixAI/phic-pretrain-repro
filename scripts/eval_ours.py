#!/usr/bin/env python3
"""Downstream eval for OUR tied-blocks pretrained model.
Methodology matches lm-eval: multiple-choice via per-option mean logprob,
lambada via next-token accuracy. Plus OOD perplexity (wikitext-103 test).
Usage: python eval_ours.py --ckpt /root/phi/ckpt_pretrain_tied-22L512d-b32-r64.pt
"""
import argparse
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/phi")

def build_model(ckpt):
    import pretrain_tb as pt
    import argparse as _ap
    cfg = _ap.Namespace(mode="tied", d=512, layers=22, ffn=1792,
                        block=32, rank=64, stages=2, bs=32, seq=2048,
                        steps=5722, gpu=0, lr=3e-4, warmup=115,
                        decay_start=4580, project="x")
    m = pt.LM(cfg).cuda().eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    return m

@torch.no_grad()
def option_logprob(model, tok, ctx_ids, opt_ids):
    ids = (ctx_ids + opt_ids).unsqueeze(0).cuda()
    logits, _ = model(ids)
    lp = torch.log_softmax(logits[0].float(), -1)
    start = len(ctx_ids) - 1
    tot = 0.0
    for j, t in enumerate(opt_ids):
        tot += lp[start + j, t].item()
    return tot / max(len(opt_ids), 1)

@torch.no_grad()
def mc_task(model, tok, docs, n_docs=500):
    correct = 0
    for doc in docs[:n_docs]:
        ctx, opts = doc
        ctx_ids = tok(ctx, add_special_tokens=False).input_ids[-1024:]
        scores = []
        for text, label in opts:
            opt_ids = tok(text, add_special_tokens=False).input_ids[-64:]
            scores.append(option_logprob(model, tok, ctx_ids, opt_ids))
        if scores.index(max(scores)) == opts[0][1]:
            correct += 1
    return correct / min(len(docs), n_docs)

@torch.no_grad()
def lambada(model, tok, n_docs=500):
    import datasets
    ds = datasets.load_dataset("EleutherAI/lambada_openai", "en",
                               split="test", streaming=True)
    correct = total = 0
    for ex in ds:
        if total >= n_docs:
            break
        ids = tok(ex["text"], add_special_tokens=False).input_ids
        if len(ids) < 8:
            continue
        ctx, tgt = ids[:-1], ids[-1]
        logits, _ = model(torch.tensor([ctx[-1024:]]).cuda())
        pred = logits[0, -1].argmax().item()
        correct += (pred == tgt)
        total += 1
    return correct / max(total, 1)

@torch.no_grad()
def ood_ppl(model, tok, n_tokens=2_000_000):
    import datasets
    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                               split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, add_special_tokens=False).input_ids
    ids = torch.tensor(ids[:n_tokens], dtype=torch.long).cuda()
    S = 2048
    n = (len(ids) - 1) // S
    tot, cnt = 0.0, 0
    for i in range(n):
        x = ids[i * S:(i + 1) * S].unsqueeze(0)
        y = ids[i * S + 1:(i + 1) * S + 1].unsqueeze(0)
        logits, _ = model(x)
        lp = torch.log_softmax(logits[:, :-1].float(), -1)
        tot += lp.gather(-1, y[:, :-1].unsqueeze(-1)).sum().item()
        cnt += y[:, :-1].numel()
    return -tot / cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    import datasets
    model = build_model(a.ckpt)
    print("model loaded", flush=True)

    def mc_docs(task, cfgname):
        if cfgname:
            ds = datasets.load_dataset(task, cfgname, split="validation",
                                       streaming=True)
        else:
            ds = datasets.load_dataset(task, split="validation",
                                       streaming=True)
        out = []
        for ex in ds:
            if len(out) >= 500:
                break
            if task == "piqa":
                out.append((ex["goal"], [(ex["sol1"], 0), (ex["sol2"], 1)]))
            elif task == "hellaswag":
                out.append((ex["ctx"], [(t, i) for i, t in
                                        enumerate(ex["endings"])]))
            elif task == "arc_easy":
                out.append((ex["question"],
                            [(t, l) for t, l in zip(ex["choices"]["text"],
                                                    ex["choices"]["label"])]))
        return out

    for name, task, cfgname in [("piqa", "piqa", None),
                                ("hellaswag", "hellaswag", None),
                                ("arc_easy", "allenai/ai2_arc", "ARC-Easy")]:
        try:
            acc = mc_task(model, tok, mc_docs(task, cfgname))
            print(f"OUR-EVAL {name}: {acc:.4f}", flush=True)
        except Exception as e:
            print(f"OUR-EVAL {name}: FAILED {e}", flush=True)
    try:
        acc = lambada(model, tok)
        print(f"OUR-EVAL lambada_openai: {acc:.4f}", flush=True)
    except Exception as e:
        print(f"OUR-EVAL lambada_openai: FAILED {e}", flush=True)
    try:
        ppl = ood_ppl(model, tok)
        print(f"OUR-EVAL wikitext_test_ppl: {math.exp(ppl):.2f} "
              f"(loss {ppl:.4f})", flush=True)
    except Exception as e:
        print(f"OUR-EVAL ood_ppl: FAILED {e}", flush=True)

if __name__ == "__main__":
    main()
