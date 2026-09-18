#!/bin/bash
# The structured-tax measurement: corrections-on-frozen-shared-prior vs free rows.
# v2 r=128: the real-config regime (corrections 25% of row params)
# v2 r=512: matched trained params/row with the shared frozen prior
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
P=/root/trainenv/bin/python
S=/tmp/phic-pretrain-repro/afs/probe/probe_train.py
T="--ffn afs --E 8 --f-row 256 --n-persons 2000 --exposures 25 --steps 15000 --bs 32 --hard-k 2"
( CUDA_VISIBLE_DEVICES=0 $P $S $T --shared-rank 128 --tag v2tax_r128 2>&1 | grep -aE 'RESULT|Error' ) &
( CUDA_VISIBLE_DEVICES=1 $P $S $T --shared-rank 512 --tag v2tax_r512 2>&1 | grep -aE 'RESULT|Error' ) &
wait
echo V2TAX_DONE
