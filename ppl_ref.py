import torch, math, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
DEV='cuda:0'; SEQ=1024
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)   # the model's OWN tokenizer

ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')
text = "\n".join(ds['text'])[:1_500_000]
ids = tok(text).input_ids[:150_000]
wc = (len(ids)-1)//(SEQ+1)
arr = torch.tensor(ids[:wc*(SEQ+1)]).view(-1, SEQ+1)
name = sys.argv[1]

m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(DEV).eval()
ce_sum, cnt = 0.0, 0
with torch.no_grad():
    for i in range(arr.shape[0]):
        lg = m(arr[i:i+1, :-1].to(DEV)).logits
        ce = torch.nn.functional.cross_entropy(lg[0].float(), arr[i,1:].to(DEV), reduction='sum')
        ce_sum += ce.item(); cnt += arr.shape[1]-1
        del lg
print(f"{name}: wikitext-103 PPL {math.exp(ce_sum/cnt):.2f} (CE {ce_sum/cnt:.4f}, same 150k slice, OLMo tokenizer)")
