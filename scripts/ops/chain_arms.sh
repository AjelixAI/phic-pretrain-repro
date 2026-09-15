#!/bin/bash
# Staged-anneal response curve, all arms at matched 27.8B budget.
# Wait for the 7B shuffled cache, then run the arms sequentially.
cd /root/phi
while [ ! -f /root/phi/data_cache_dolmino_7B_shuf.pt ]; do sleep 30; done
sleep 10
source /root/.wandb_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--mode tied --init-from /root/phi/ckpt_tied-22L512d-b32-r64_step40000.pt --steps 12991 --bs 32 --seq 2048 --d 512 --layers 22 --ffn 1792 --block 32 --rank 64 --lr 3e-4 --warmup 1 --decay-start 2400 --project phic-midtrain --no-ckpt --chunked-ce 8192"
echo "=== ARM B1 (1.81B curated tail) $(date) ==="
/root/venv/bin/torchrun --nproc_per_node=8 /root/phi/pretrain_tb_fast.py $COMMON \
  --cache /root/phi/data_cache_olmomix_shuf.pt \
  --cache2 /root/phi/data_cache_dolmino_7B_shuf.pt --switch-step 9538 \
  --tag-suffix=-midB1-dolmino-1818M > arm_B1.log 2>&1
echo "=== ARM B2 (3B curated tail) $(date) ==="
/root/venv/bin/torchrun --nproc_per_node=8 /root/phi/pretrain_tb_fast.py $COMMON \
  --cache /root/phi/data_cache_olmomix_shuf.pt \
  --cache2 /root/phi/data_cache_dolmino_7B_shuf.pt --switch-step 7269 \
  --tag-suffix=-midB2-dolmino-3000M > arm_B2.log 2>&1
echo "=== ARM AFULL (6.81B curated tail) $(date) ==="
/root/venv/bin/torchrun --nproc_per_node=8 /root/phi/pretrain_tb_fast.py $COMMON \
  --cache /root/phi/data_cache_dolmino_7B_shuf.pt \
  --tag-suffix=-midAFULL-dolmino-6810M > arm_AFULL.log 2>&1
echo "=== ALL ARMS DONE $(date) ==="
