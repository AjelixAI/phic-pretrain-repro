#!/bin/bash
# The falsification controls for the 81% result:
#  C1: AFS-8 with FROZEN RANDOM keys (no routing learning) -> tests the routing mechanism
#  C2: dense with MATCHED TOTAL params (~8.45M)            -> tests "just more parameters"
#  C3: dense seed-replicate                               -> measures dense run variance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
P=/root/trainenv/bin/python
S=/tmp/phic-pretrain-repro/afs/probe/probe_train.py
T="--d 256 --L 4 --nh 4 --f-row 256 --n-persons 2000 --exposures 25 --steps 15000 --bs 32"
run() { CUDA_VISIBLE_DEVICES=$1 $P $S $2 $T --tag "$3" 2>&1 | grep -aE 'RESULT|Error'; }
( CUDA_VISIBLE_DEVICES=0 $P $S --ffn afs --E 8 $T --tag ctl_randomkeys 2>&1 | grep -aE 'RESULT|Error' ) &
( CUDA_VISIBLE_DEVICES=1 $P $S --ffn afs --E 32 $T --tag ctl_afs32_seedrep 2>&1 | grep -aE 'RESULT|Error' ) &
( CUDA_VISIBLE_DEVICES=2 $P $S --ffn dense --dense-f 2500 $T --tag ctl_dense_parammatched 2>&1 | grep -aE 'RESULT|Error' ) &
( CUDA_VISIBLE_DEVICES=3 $P $S --ffn dense --dense-f 450 $T --tag ctl_dense_seed2 2>&1 | grep -aE 'RESULT|Error' ) &
wait
echo CONTROLS_DONE
