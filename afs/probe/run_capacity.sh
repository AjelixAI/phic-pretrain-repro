#!/bin/bash
# The capacity-binding regime: 800k facts (6.4 Mbit) > dense ceiling (1.8 Mbit)
# 200 exposures/fact, 400M tokens (~49k steps). Dense should SATURATE;
# AFS-128 (200 Mbit ceiling) should hold all facts if the design works.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
P=/root/trainenv/bin/python
S=/tmp/phic-pretrain-repro/afs/probe/probe_train.py
$P $S --ffn dense --dense-f 450 --n-persons 100000 --exposures 200 --steps 49000 --bs 32 --hard-k 0 --tag cap_N100k_dense 2>&1 | grep -aE '^\[|RESULT|Error'
$P $S --ffn afs --E 128 --n-persons 100000 --exposures 200 --steps 49000 --bs 32 --hard-k 0 --tag cap_N100k_afs128 2>&1 | grep -aE '^\[|RESULT|Error'
echo CAPACITY_DONE
