import sys, torch, argparse
sys.path.insert(0, "/root/phi")
import pretrain_tb as pt
from transformers import LlamaForCausalLM
cfg = argparse.Namespace(mode="tied", d=512, layers=22, ffn=1792, block=32,
                         rank=64, stages=2)
m = pt.LM(cfg).cuda().eval()
sd = torch.load("/root/phi/ckpt_pretrain_tied-22L512d-b32-r64.pt",
                map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=True)
ll = LlamaForCausalLM.from_pretrained("/root/phi/export_llama",
                                      torch_dtype=torch.bfloat16).cuda().eval()
b0 = m.blocks[0]; l0 = ll.model.layers[0]
x = (torch.randn(4, 1, 512) * 0.5).bfloat16().cuda()
with torch.no_grad():
    # 1. our original function vs the materialized dense weights
    Wg = l0.mlp.gate_proj.weight            # [1792, 512] (nn.Linear: y = x@W.T)
    y_our_gate = b0.gate(x)
    y_ll_gate = (x.reshape(4, 512) @ Wg.T)
    print("gate: our vs materialized:",
          (y_our_gate.reshape(4, 1792) - y_ll_gate).abs().max().item(),
          "| our norm:", y_our_gate.abs().mean().item(), flush=True)
    xh = (torch.randn(4, 1, 1792) * 0.5).bfloat16().cuda()
    Wd = l0.mlp.down_proj.weight            # [512, 1792]
    y_our_down = b0.down(xh)
    y_ll_down = xh.reshape(4, 1792) @ Wd.T
    print("down: our vs materialized:",
          (y_our_down.reshape(4, 512) - y_ll_down).abs().max().item(),
          "| our norm:", y_our_down.abs().mean().item(), flush=True)
    # 2. norms of the exported weights vs a direct eye-trick re-materialization
    eye = torch.eye(1792, dtype=torch.bfloat16, device="cuda")
    W_our_down = b0.down(eye.unsqueeze(0))[0].T   # [512, 1792]
    print("down weight diff (ours re-mat vs exported):",
          (W_our_down - Wd).abs().max().item(), flush=True)
    print("down weight norms: re-mat", W_our_down.abs().mean().item(),
          "exported", Wd.abs().mean().item(), flush=True)
