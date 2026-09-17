#!/bin/bash
# The 24/7 watchdog: the run dies -> auto-resume from the latest full checkpoint.
cd /root/phi
source /root/.wandb_local
export FP8_BUILD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAX_RESTARTS=50
COUNT=0
while [ $COUNT -lt $MAX_RESTARTS ]; do
  if ! pgrep -f 'pretrain_tb_fast.*sota1B' > /dev/null; then
    # the check: the run completed?
    if grep -q 'DONE' /root/phi/sota_run.log 2>/dev/null && ! pgrep -f 'pretrain_tb_fast' > /dev/null; then
      # the DONE in the log could be from the previous phase - verify the step count
      LAST_STEP=$(grep -aoE 'step +[0-9]+' /root/phi/sota_run.log | tail -1 | grep -oE '[0-9]+')
      if [ -n "$LAST_STEP" ] && [ "$LAST_STEP" -ge 325559 ]; then
        echo "$(date) RUN COMPLETE at step $LAST_STEP" >> /root/phi/watchdog.log
        break
      fi
    fi
    CKPT=/root/phi/ckpt_tied-16L2048d-b32-r64-sota1B_running.pt
    COUNT=$((COUNT+1))
    echo "$(date) run died (restart $COUNT/$MAX_RESTARTS) - resuming" >> /root/phi/watchdog.log
    if [ -f "$CKPT" ]; then
      /root/trainenv/bin/torchrun --nproc_per_node=4 --master_port=29500 /root/phi/pretrain_tb_fast.py \
        --mode tied --bs 16 --seq 4096 --d 2048 --layers 16 --ffn 7168 --block 32 --rank 64 \
        --vocab 100352 --lr 3e-4 --warmup 500 --steps 325560 --decay-start 317925 \
        --project phic-sota1B --cache /root/phi/data_cache_sota100B.pt \
        --val-file /root/phi/val_generic_500M.pt --val-file2 /root/phi/val_anneal_250M.pt \
        --chunked-ce 8192 --rope-base 10000 --save-every 700 --tag-suffix=-sota1B --force-save \
        --resume $CKPT >> /root/phi/sota_run.log 2>&1
    else
      /root/trainenv/bin/torchrun --nproc_per_node=4 --master_port=29500 /root/phi/pretrain_tb_fast.py \
        --mode tied --bs 16 --seq 4096 --d 2048 --layers 16 --ffn 7168 --block 32 --rank 64 \
        --vocab 100352 --lr 3e-4 --warmup 500 --steps 325560 --decay-start 317925 \
        --project phic-sota1B --cache /root/phi/data_cache_sota100B.pt \
        --val-file /root/phi/val_generic_500M.pt --val-file2 /root/phi/val_anneal_250M.pt \
        --chunked-ce 8192 --rope-base 10000 --save-every 700 --tag-suffix=-sota1B --force-save \
        >> /root/phi/sota_run.log 2>&1
    fi
  fi
  sleep 300
done
