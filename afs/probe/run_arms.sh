#!/bin/bash
# The capacity arms: fixed compute, varying FFN knowledge capacity.
export CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/root/trainenv/bin/python
S=afs/probe/probe_train.py
COMMON="--n-persons 1000 --exposures 300 --steps 4000 --hard-k 2 --bs 16"
$P $S --ffn dense --dense-f 850 $COMMON --tag dense_cm 2>&1 | grep -aE '^\[|RESULT'
$P $S --ffn afs --E 32  $COMMON --tag afs_E32  2>&1 | grep -aE '^\[|RESULT'
$P $S --ffn afs --E 128 $COMMON --tag afs_E128 2>&1 | grep -aE '^\[|RESULT'
$P $S --ffn afs --E 512 $COMMON --tag afs_E512 2>&1 | grep -aE '^\[|RESULT'
echo ARMS_DONE
