#!/bin/bash
export CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
P=/root/trainenv/bin/python
S=/tmp/phic-pretrain-repro/afs/probe/probe_train.py
$P $S --ffn pk --E 1024 --bs 8 --n-persons 2000 --exposures 25 --steps 15000 --hard-k 0 --tag v3_N2k_pk4096 2>&1 | grep -aE '^\[|RESULT|Error'
$P $S --ffn pk --E 1024 --bs 8 --n-persons 20000 --exposures 25 --steps 15000 --hard-k 0 --tag v3_N20k_pk4096 2>&1 | grep -aE '^\[|RESULT|Error'
echo PK_DONE
