import sys, torch, argparse
sys.path.insert(0, "/root/phi")
import pretrain_tb as pt
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32,
                         rank=64, stages=2)
m = pt.LM(cfg).cuda().eval()
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/root/phi/ckpt_pretrain_tied-22L512d-b32-r64.pt"
OUTD = sys.argv[2] if len(sys.argv) > 2 else "/root/phi/export_llama"
print(f"exporting {CKPT} -> {OUTD}", flush=True)
sd = torch.load(CKPT,
                map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=True)
print("model loaded", flush=True)

def dense_weight(tl):
    """Materialize a TiedLinear as [out, in] (nn.Linear layout) by running
    the ORIGINAL verified forward on the identity: W_full[:, j] = f(e_j),
    so W_full = forward(I)[0].T and nn.Linear weight = W_full.T."""
    with torch.no_grad():
        eye = torch.eye(tl.d_in, dtype=torch.bfloat16, device="cuda")
        W_out_in = tl.forward(eye.unsqueeze(0))[0].T   # [out, in]
        return W_out_in.contiguous()

def rot_half_perm(d):
    """Permutation converting interleaved-rope rows to half-split rows."""
    idx = list(range(0, d, 2)) + list(range(1, d, 2))
    return torch.tensor(idx)

sd_ll = {}
sd_ll["model.embed_tokens.weight"] = m.emb.weight.data.clone()
sd_ll["model.norm.weight"] = m.nf.weight.data.clone()
per_head = rot_half_perm(64)
for i, blk in enumerate(m.blocks):
    p = f"model.layers.{i}."
    sd_ll[p + "input_layernorm.weight"] = blk.n1.weight.data.clone()
    sd_ll[p + "post_attention_layernorm.weight"] = blk.n2.weight.data.clone()
    q = dense_weight(blk.attn.q)
    k = dense_weight(blk.attn.k)
    v = dense_weight(blk.attn.v)
    o = dense_weight(blk.attn.o)
    # reorder q/k rows: interleaved -> half-split, per head
    q = q.view(8, 64, 512)[:, per_head, :].reshape(512, 512)
    k = k.view(2, 64, 512)[:, per_head, :].reshape(128, 512)
    sd_ll[p + "self_attn.q_proj.weight"] = q.contiguous()
    sd_ll[p + "self_attn.k_proj.weight"] = k.contiguous()
    sd_ll[p + "self_attn.v_proj.weight"] = v.contiguous()
    sd_ll[p + "self_attn.o_proj.weight"] = o.contiguous()
    sd_ll[p + "mlp.gate_proj.weight"] = dense_weight(blk.gate).contiguous()
    sd_ll[p + "mlp.up_proj.weight"] = dense_weight(blk.up).contiguous()
    sd_ll[p + "mlp.down_proj.weight"] = dense_weight(blk.down).contiguous()

lcfg = LlamaConfig(vocab_size=50304, hidden_size=512,
                   intermediate_size=1792, num_hidden_layers=22,
                   num_attention_heads=8, num_key_value_heads=2,
                   max_position_embeddings=2048, rms_norm_eps=0.0078125,
                   tie_word_embeddings=True, rope_theta=64)
ll = LlamaForCausalLM(lcfg)
missing, unexpected = ll.load_state_dict(sd_ll, strict=False)
missing = [k for k in missing if "lm_head" not in k]
print("missing:", missing, "unexpected:", unexpected, flush=True)
assert not missing and not unexpected, "export layout mismatch"
ll = ll.cuda().to(torch.bfloat16).eval()

# ---- equivalence check: our logits vs llama logits -----------------------
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
x = torch.tensor([tok("The capital of France is",
                      add_special_tokens=False).input_ids]).cuda()
with torch.no_grad():
    lo, _ = m(x)
    ll_o = ll(x).logits
d = (lo[0, -1].float() - ll_o[0, -1].float()).abs().max().item()
rel = d / lo[0, -1].float().abs().max().item()
print(f"logit equivalence: max|d|={d:.4e} rel={rel:.1e}", flush=True)
# argmax-agreement gate: >=95% over >=16k real-data next-token predictions
import numpy as np
_stream = torch.load('/root/phi/data_cache_dolmino_shuf.pt',
                     weights_only=True, mmap=True)
agree_n = agree_hits = 0
with torch.no_grad():
    for i in range(160):
        seg = torch.as_tensor(np.asarray(_stream[i * 4096:(i + 1) * 4096])).long().cuda()
        if seg.numel() < 1025:
            break
        o_logits, _ = m(seg.unsqueeze(0))
        l_logits = ll(seg.unsqueeze(0)).logits
        agree_n += l_logits.shape[1] - 1
        agree_hits += (o_logits[0, :-1].argmax(-1) == l_logits[0, :-1].argmax(-1)).sum().item()
agree = agree_hits / max(agree_n, 1)
print(f"argmax agreement over {agree_n} predictions: {agree*100:.1f}%", flush=True)
ok = rel < 0.05 and agree >= 0.95
print("EXPORT VERIFIED" if ok else "EXPORT DIVERGES", flush=True)
ll.save_pretrained(OUTD)
tok.save_pretrained(OUTD)
print("saved /root/phi/export_llama", flush=True)
