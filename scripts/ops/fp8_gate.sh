#!/bin/bash
# The FP8 quality gate: FP8-vs-bf16 at the SOTA-run config, matched budget.
# 200M tokens per arm, identical data/seed/schedule; the loss curves and
# the val must match within the bf16-tolerance for the 100B run to use FP8.
cd /root/phi
source /root/.wandb_local
CACHE=/root/phi/data_cache_gatedata.pt
COMMON="--mode tied --steps 6104 --bs 8 --seq 1024 --d 2048 --layers 16 --ffn 7168 --block 32 --rank 64 --vocab 100352 --lr 3e-4 --warmup 100 --decay-start 6104 --project phic-fp8gate --cache $CACHE --no-ckpt --rope-base 10000"
echo "=== FP8 GATE bf16 arm $(date) ===" > /root/phi/fp8_gate_results.md
FP8_GEMM=0 /root/trainenv/bin/torchrun --nproc_per_node=4 --master_port=29511 /root/phi/pretrain_tb_fast.py $COMMON --gpu 0 > gate_bf16.log 2>&1
grep -E 'VAL' gate_bf16.log | tail -1 >> /root/phi/fp8_gate_results.md
echo "=== FP8 GATE fp8 arm $(date) ===" >> /root/phi/fp8_gate_results.md
FP8_GEMM=1 /root/trainenv/bin/torchrun --nproc_per_node=4 --master_port=29511 /root/phi/pretrain_tb_fast.py $COMMON --gpu 0 > gate_fp8.log 2>&1
grep -E 'VAL' gate_fp8.log | tail -1 >> /root/phi/fp8_gate_results.md
python3 - <<'PY' >> /root/phi/fp8_gate_results.md
import re
def curve(path):
    ls = []
    for line in open(path):
        m = re.search(r'step\s+(\d+) loss ([\d.]+)', line)
        if m: ls.append((int(m.group(1)), float(m.group(2))))
    return dict(ls)
b = curve('/root/phi/gate_bf16.log')
f = curve('/root/phi/gate_fp8.log')
common = sorted(set(b) & set(f))
diffs = [abs(b[s] - f[s]) / max(b[s], 1e-6) for s in common]
print(f"steps compared: {len(common)} | max relative loss diff: {max(diffs):.4f}")
print("GATE:", "PASS (FP8 within bf16 tolerance)" if max(diffs) < 0.02 else "FAIL (FP8 diverges - use bf16)")
PY
echo '=== FP8 GATE DONE ===' >> /root/phi/fp8_gate_results.md
