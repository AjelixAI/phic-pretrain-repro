#!/bin/bash
cd /root/phi
# wait for the sweep + reconstruction to free the GPUs
while [ ! -f /root/phi/lmeval_reconstructed/lmeval_reconstructed.log ] && ! pgrep -f 'reconstruct_after_swee[p]' > /dev/null; do sleep 60; done
while pgrep -f 'torchrun' > /dev/null || pgrep -f 'reconstruct_final' > /dev/null; do sleep 60; done
while pgrep -f 'pretrain_tb_fas[t]' > /dev/null; do sleep 60; done
source /root/.wandb_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for RB in 64 10000; do
  echo "=== ROPE PROBE base=$RB $(date) ===" >> /root/phi/rope_probe_results.md
  /root/venv/bin/torchrun --nproc_per_node=8 /root/phi/pretrain_tb_fast.py --mode tied --steps 3815 --bs 32 --seq 2048 --d 512 --layers 22 --ffn 1792 --block 32 --rank 64 --lr 3e-4 --warmup 150 --decay-start 3815 --project phic-rope --cache /root/phi/data_cache_olmomix_shuf.pt --no-ckpt --chunked-ce 8192 --rope-base $RB --tag-suffix=-rope$RB > rope_$RB.log 2>&1
done
echo '=== ROPE PROBES TRAINED ===' >> /root/phi/rope_probe_results.md
