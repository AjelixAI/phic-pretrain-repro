#!/bin/bash
# The clean 4-arm comparison: N=2k (8k facts), 15k steps (3,400 exp/fact),
# gradient-connected rows, E-independent top-k training. GPUs 0-3 parallel.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
P=/root/trainenv/bin/python
S=/tmp/phic-pretrain-repro/afs/probe/probe_train.py
T="--d 256 --L 4 --nh 4 --f-row 256 --n-persons 2000 --exposures 25 --steps 15000 --bs 32"
run() { CUDA_VISIBLE_DEVICES=$1 $P $S $2 $T --tag "$3" 2>&1 | grep -aE 'RESULT|Error'; }
( run 0 "--ffn dense --dense-f 450" "c_dense" ) &
( run 1 "--ffn afs --E 8"           "c_afs8" ) &
( run 2 "--ffn afs --E 32"          "c_afs32" ) &
( run 3 "--ffn afs --E 128"         "c_afs128" ) &
wait
echo CLEAN_DONE
