import sys, torch, json, os
sys.path.insert(0, "/root/phi")
import pretrain_tb_fast as pt
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
import argparse
cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32, rank=64, stages=2)
torch.manual_seed(0)
m = pt.LM(cfg).cuda().eval()
sd = torch.load("/root/phi/ckpt_pretrain_tied-22L512d-b32-r64-rope10000.pt", map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=True)
tok = AutoTokenizer.from_pretrained("/root/phi/export_rope10000")
data = torch.load("/root/phi/data_cache_olmomix_shuf.pt", weights_only=True, mmap=True)
xb = torch.as_tensor(data[:4096].numpy() if hasattr(data, 'numpy') else data[:4096]).long().cuda().view(1, 1, 4096)[:, 0, :].unsqueeze(0)
with torch.no_grad():
    o_logits, _ = m(xb)
    o_arg = o_logits[0].argmax(-1)
for theta in [64.0, 10000.0, 500000.0]:
    cpath = "/root/phi/export_rope10000/config.json"
    c = json.load(open(cpath))
    c["rope_theta"] = theta
    json.dump(c, open(cpath, "w"))
    ll = LlamaForCausalLM.from_pretrained("/root/phi/export_rope10000", torch_dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        l_logits = ll(xb).logits
    agree = (o_arg == l_logits[0].argmax(-1)).float().mean().item()
    print(f"theta={theta:.0f}: argmax agreement {agree*100:.1f}%", flush=True)
    del ll; torch.cuda.empty_cache()
