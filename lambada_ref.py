import torch, sys, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
DEV='cuda:0'
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)   # the model's OWN tokenizer

tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(DEV).eval()
ds = load_dataset('EleutherAI/lambada_openai', split='test')['text']
nonascii = re.compile(r'[^\x20-\x7e\u2018\u2019\u201c\u201d\u00e9\u00e8\u00e0\u00e7]')
en = [t for t in ds if len(nonascii.findall(t)) <= 2 and ' the ' in t.lower()][:2000]
acc = n = 0
with torch.no_grad():
    for i in range(0, len(en), 8):
        docs = en[i:i+8]
        ctxs = [d.rsplit(' ', 1)[0] for d in docs]
        tgts = [" "+d.rsplit(' ', 1)[1] for d in docs]
        enc = [tok(c).input_ids for c in ctxs]
        order = sorted(range(len(docs)), key=lambda k: len(enc[k]))
        for bi in range(0, len(order), 8):
            chunk = order[bi:bi+8]
            L = max(len(enc[k]) for k in chunk)
            inp = torch.zeros((len(chunk), L), dtype=torch.long)
            for k, ci in enumerate(chunk): inp[k,:len(enc[ci])] = torch.tensor(enc[ci])
            am = m(inp.to(DEV)).logits.argmax(-1)
            for k, ci in enumerate(chunk):
                t = tok(tgts[ci]).input_ids[0]
                n += 1; acc += am[k, len(enc[ci])-1].item() == t
print(f"{name}: LAMBADA acc {acc/n:.4f} (n={n}, same 2000-probe protocol)")
