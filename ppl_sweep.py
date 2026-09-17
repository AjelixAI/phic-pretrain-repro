import torch, sys
sys.path.insert(0, '/root/phi')
from pretrain_tb_fast import LM
from transformers import AutoTokenizer
from types import SimpleNamespace as NS
from datasets import load_dataset
DEV='cuda:0'; B=1; SEQ=1024
tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
cfg = NS(mode='tied', d=2048, ffn=7168, layers=16, block=32, rank=64, stages=2,
         vocab=100352, seq=4096, rope_base=10000.0, chunked_ce=0)
ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')
text = "\n".join(ds['text'])[:1_500_000]
ids = tok(text).input_ids
ids = ids[:150_000]   # fixed 150k-token slice of wikitext-103 test
wc = (len(ids)-1)//(SEQ+1)
arr = torch.tensor(ids[:wc*(SEQ+1)]).view(-1, SEQ+1)
print(f"wikitext-103 test slice: {arr.numel():,} tokens, {arr.shape[0]} windows", flush=True)
for ck in ['mile_56000','mile_61600','mile_65100','mile_67900']:
    m = LM(cfg)
    sd = torch.load(f'/root/phi/ckpt_tied-16L2048d-b32-r64-sota1B_{ck}.pt', map_location='cpu', weights_only=True)
    m.load_state_dict(sd.get('model', sd), strict=False); m.to(DEV).eval()
    ce_sum, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, arr.shape[0], B):
            chunk = arr[i:i+B]
            lg, _ = m(chunk[:, :-1].to(DEV))
            for c in range(0, lg.shape[1], 256):
                n = min(256, lg.shape[1]-c)
                ce = torch.nn.functional.cross_entropy(
                    lg[:, c:c+n].reshape(-1, lg.shape[-1]).float(),
                    chunk[:, 1:].reshape(-1).to(DEV)[c*chunk.shape[0]:(c+n)*chunk.shape[0]],
                    reduction='sum')
                ce_sum += ce.item(); cnt += n*chunk.shape[0]
            del lg
    import math
    print(f"{ck}: wikitext-103 PPL {math.exp(ce_sum/cnt):.2f}  (CE {ce_sum/cnt:.4f})", flush=True)
    del m, sd; torch.cuda.empty_cache()
print("DONE", flush=True)
