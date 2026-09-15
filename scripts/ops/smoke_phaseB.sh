#!/bin/bash
cd /root/phi
source /root/.wandb_local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=/root/phi/ckpt_tied-16L2048d-b32-r64-smoke_running.pt
while [ ! -f $CKPT ]; do sleep 20; done
# the phase A completes its 1000 steps and saves; wait for the run to exit
while pgrep -f 'smoke_a\|pretrain_tb_fas[t]' > /dev/null && ! grep -q 'DONE' smoke_a.log 2>/dev/null; do sleep 20; done
sleep 10
export FP8_GEMM=1
echo "=== SMOKE phase B: FP8 resume from the trained state ==="
/root/trainenv/bin/torchrun --nproc_per_node=4 --master_port=29521 /root/phi/pretrain_tb_fast.py \
  --mode tied --bs 8 --seq 4096 --d 2048 --layers 16 --ffn 7168 --block 32 --rank 64 \
  --vocab 100352 --lr 3e-4 --warmup 100 --project phic-smoke --cache /root/phi/data_cache_gatedata.pt \
  --no-ckpt --chunked-ce 8192 --rope-base 10000 \
  --val-file /root/phi/smoke_val_general.pt --val-file2 /root/phi/smoke_val_anneal.pt \
  --tag-suffix=-smoke --save-every 250 --force-save \
  --steps 2000 --decay-start 1500 --resume $CKPT > smoke_b.log 2>&1
echo "=== SMOKE PHASE B DONE ==="
grep -E 'FULL RESUME|VAL |step ' smoke_b.log | tail -6
