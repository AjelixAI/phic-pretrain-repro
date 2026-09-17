#!/bin/bash
# Relaunch with bs 16 + CUDA graphs, resuming from the latest save.
# Token budget preserved: total steps scaled by bs ratio 12/16 = 0.75
#   434080 * 0.75 = 325560 ; decay_start 423900 * 0.75 = 317925
set -u
source /root/.wandb_local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FP8_BUILD=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /root/phi
nohup /root/trainenv/bin/torchrun --nproc_per_node=4 --master_port=29500 \
  /root/phi/pretrain_tb_fast.py --mode tied --bs 16 --seq 4096 --d 2048 --layers 16 \
  --ffn 7168 --block 32 --rank 64 --vocab 100352 --lr 3e-4 --warmup 500 \
  --steps 325560 --decay-start 317925 --project phic-sota1B \
  --cache /root/phi/data_cache_sota100B.pt \
  --val-file /root/phi/val_generic_500M.pt --val-file2 /root/phi/val_anneal_250M.pt \
  --chunked-ce 8192 --rope-base 10000 --save-every 700 --ckpt-every 4 \
  --tag-suffix=-sota1B --force-save \
  --resume /root/phi/ckpt_tied-16L2048d-b32-r64-sota1B_running.pt \
  >> /root/phi/sota_run.log 2>&1 &
echo "launched pid $!"
sleep 300
echo "=== status after 5 min:"
grep -aE 'RESUME|graph|step |Error|error' /root/phi/sota_run.log | tail -8
