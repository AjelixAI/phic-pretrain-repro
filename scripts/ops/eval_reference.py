import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
DEV='cuda:3'; B=16
MODEL = sys.argv[1]
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
def ll_batch(pairs):
    jobs = []
    for ci, (ctx, conts, gold) in enumerate(pairs):
        ce = tok(ctx).input_ids
        for ri, cont in enumerate(conts):
            jobs.append((ci, ri, ce + tok(cont).input_ids, len(ce), gold))
    jobs.sort(key=lambda j: len(j[2]))
    s = {}
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i+B]
        L = max(len(j[2]) for j in chunk)
        inp = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, (_, _, ids, _, _) in enumerate(chunk): inp[k,:len(ids)] = torch.tensor(ids)
        with torch.no_grad():
            lg = m(inp.to(DEV)).logits
        lsm = torch.log_softmax(lg.float(), -1)
        for k, (ci, ri, ids, nctx, gold) in enumerate(chunk):
            pos = torch.arange(nctx-1, len(ids)-1, device=DEV)
            tg = torch.tensor(ids[nctx:], device=DEV)
            s[(ci,ri)] = lsm[k,pos,tg].sum().item()
    acc = n = 0
    for ci, (ctx, conts, gold) in enumerate(pairs):
        a = max(range(len(conts)), key=lambda r: s[(ci,r)])
        n += 1; acc += a == gold
    return acc/n
ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
pairs = [(f"Question: {d['question']}\nAnswer:", [" "+c for c in d['choices']['text']],
          d['choices']['label'].index(d['answerKey'])) for d in ds
         if d['choices']['text'] and d['answerKey'] in d['choices']['label']]
arc = ll_batch(pairs)
ds = load_dataset("Rowan/hellaswag", split="validation").select(range(1500))
pairs = [(f"{d['activity_label']}: {d['ctx_a']} {d['ctx_b'].capitalize()}", [" "+e for e in d['endings']],
          int(d['label'])) for d in ds]
hs = ll_batch(pairs)
print(f"{MODEL}: ARC-Easy {arc:.4f} HellaSwag {hs:.4f}", flush=True)
