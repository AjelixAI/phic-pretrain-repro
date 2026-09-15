import sys, torch
sys.path.insert(0, "/root/phi")
import argparse
import pretrain_tb as pt
from transformers import AutoTokenizer

cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32,
                         rank=64, stages=2, bs=32, seq=2048, steps=52991,
                         gpu=0, lr=3e-4, warmup=115, decay_start=4580,
                         project="x")
m = pt.LM(cfg).cuda().eval()
sd = torch.load("/root/phi/ckpt_pretrain_tied-22L512d-b32-r64.pt",
                map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=True)
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
print("model loaded", flush=True)

# ---- 1. REAL GENERATION ------------------------------------------------
@torch.no_grad()
def gen(prompt, n=60):
    ids = tok(prompt, add_special_tokens=False).input_ids[-1024:]
    x = torch.tensor([ids]).cuda()
    for _ in range(n):
        logits, _ = m(x)
        nxt = logits[0, -1].argmax().item()
        x = torch.cat([x, torch.tensor([[nxt]]).cuda()], 1)
        if x.shape[1] > 2048:
            x = x[:, -2048:]
    return tok.decode(x[0].tolist()[len(ids):])

print("\n=== GENERATION TEST ===", flush=True)
print("[P1] 'The capital of France is'", flush=True)
print("GEN:", repr(gen("The capital of France is")[:200]), flush=True)
print("[P2] 'Once upon a time, a small robot discovered'", flush=True)
print("GEN:", repr(gen("Once upon a time, a small robot discovered")[:200]), flush=True)
print("[P3] 'Water boils at a temperature of'", flush=True)
print("GEN:", repr(gen("Water boils at a temperature of")[:200]), flush=True)

# ---- 2. MANUAL LAMBADA CHECK -------------------------------------------
import datasets
print("\n=== MANUAL LAMBADA (5 examples) ===", flush=True)
lds = datasets.load_dataset("EleutherAI/lambada_openai", "en", split="test",
                            streaming=True)
correct = 0
for i, ex in enumerate(lds):
    if i >= 5:
        break
    ids = tok(ex["text"], add_special_tokens=False).input_ids
    ctx, tgt = ids[:-1], ids[-1]
    with torch.no_grad():
        logits, _ = m(torch.tensor([ctx[-1024:]]).cuda())
    lp = torch.log_softmax(logits[0, -1].float(), -1)
    top5 = lp.topk(5).indices.tolist()
    pred_ok = top5[0] == tgt
    correct += pred_ok
    print(f"[{i}] ctx tail: ...{tok.decode(ctx[-25:])!r}", flush=True)
    print(f"    target: {tok.decode([tgt])!r} | rank of target: "
          f"{(lp > lp[tgt]).sum().item() + 1} | top-5 preds: "
          f"{[tok.decode([t]) for t in top5]}", flush=True)
print(f"manual lambada: {correct}/5", flush=True)

# ---- 3. MANUAL PIQA/ARC INSPECTION --------------------------------------
print("\n=== MANUAL PIQA (2 examples, per-option scores) ===", flush=True)
pds = datasets.load_dataset("baber/piqa", split="validation")
for i in (0, 3):
    ex = pds[i]
    ctx = ex["goal"]
    ctx_ids = tok(ctx, add_special_tokens=False).input_ids[-1024:]
    with torch.no_grad():
        for j, opt in enumerate((ex["sol1"], ex["sol2"])):
            opt_ids = tok(opt, add_special_tokens=False).input_ids[-64:]
            ids = torch.tensor([ctx_ids + opt_ids]).cuda()
            logits, _ = m(ids)
            lp = torch.log_softmax(logits[0].float(), -1)
            st = len(ctx_ids) - 1
            tot = sum(lp[st + j2, t].item() for j2, t in enumerate(opt_ids))
            print(f"[{i}] opt{j+1} meanLP {tot/max(len(opt_ids),1):.3f}: {opt!r}",
                  flush=True)
    print(f"    label: {ex['label']}", flush=True)

print("\n=== MANUAL ARC (2 examples) ===", flush=True)
ads = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
for i in (0, 2):
    ex = ads[i]
    ctx = ex["question"]
    ctx_ids = tok(ctx, add_special_tokens=False).input_ids[-1024:]
    labs = ex["choices"]["label"]
    with torch.no_grad():
        for j, (t, l) in enumerate(zip(ex["choices"]["text"], labs)):
            opt_ids = tok(t, add_special_tokens=False).input_ids[-64:]
            ids = torch.tensor([ctx_ids + opt_ids]).cuda()
            logits, _ = m(ids)
            lp = torch.log_softmax(logits[0].float(), -1)
            st = len(ctx_ids) - 1
            tot = sum(lp[st + j2, t].item() for j2, t in enumerate(opt_ids))
            print(f"[{i}] opt {l} meanLP {tot/max(len(opt_ids),1):.3f}: {t!r}",
                  flush=True)
    print(f"    answerKey: {ex['answerKey']}", flush=True)
print("INSPECT-DONE", flush=True)
