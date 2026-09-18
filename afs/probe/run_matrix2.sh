#!/bin/bash
# v2 protocol: FIXED token budget (60M tokens = 7,325 steps at bs32x256);
# N sweeps the per-fact exposure: N=2k -> ~1,250 exp/fact, 20k -> ~125, 200k -> ~12.
# Arms: dense compute-matched | AFS E=32 | AFS E=128 | product-key 4096 rows.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
P=/root/trainenv/bin/python
S=/tmp/phic-pretrain-repro/afs/probe/probe_train.py
COMMON="--exposures 25 --steps 7325 --bs 32 --hard-k 2"
run() { CUDA_VISIBLE_DEVICES=$1 $P $S $2 $COMMON --tag "$3" 2>&1 | grep -aE '^\[|RESULT'; }
( run 0 "--ffn dense --dense-f 450 --n-persons 2000"  "v2_N2k_dense" ; \
  run 0 "--ffn dense --dense-f 450 --n-persons 20000" "v2_N20k_dense" ) &
( run 1 "--ffn afs --E 32  --n-persons 2000"  "v2_N2k_afs32" ; \
  run 1 "--ffn afs --E 32  --n-persons 20000" "v2_N20k_afs32" ) &
( run 2 "--ffn afs --E 128 --n-persons 2000"  "v2_N2k_afs128"; \
  run 2 "--ffn afs --E 128 --n-persons 20000" "v2_N20k_afs128" ) &
( run 3 "--ffn pk --E 4096 --bs 16 --n-persons 2000"  "v2_N2k_pk4096"; \
  run 3 "--ffn pk --E 4096 --bs 16 --n-persons 20000" "v2_N20k_pk4096" ) &
wait
echo MATRIX2_DONE
