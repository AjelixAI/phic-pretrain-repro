#!/bin/bash
# AFS/PK training arms: soft/bmm path (hard_k=0). Dense cells already done.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
P=/root/trainenv/bin/python
S=/tmp/phic-pretrain-repro/afs/probe/probe_train.py
COMMON="--exposures 25 --steps 15000 --bs 32"
run() { CUDA_VISIBLE_DEVICES=$1 $P $S $2 $COMMON --tag "$3" 2>&1 | grep -aE '^\[|RESULT'; }
( run 1 "--ffn afs --E 32  --n-persons 2000"  "v3_N2k_afs32"  ; run 1 "--ffn afs --E 32  --n-persons 20000" "v3_N20k_afs32" ) &
( run 2 "--ffn afs --E 128 --n-persons 2000"  "v3_N2k_afs128" ; run 2 "--ffn afs --E 128 --n-persons 20000" "v3_N20k_afs128" ) &
( run 3 "--ffn pk  --E 4096 --bs 16 --n-persons 2000"  "v3_N2k_pk4096" ; run 3 "--ffn pk  --E 4096 --bs 16 --n-persons 20000" "v3_N20k_pk4096" ) &
wait
echo ARMS3_DONE
