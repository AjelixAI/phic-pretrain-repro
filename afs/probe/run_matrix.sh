#!/bin/bash
# The saturation-protocol matrix: N-persons sweep x FFN arms, GPUs 0-3.
# d=256, 8 bits/fact (vpa=256), statements-only training, zero-shot queries.
# Arms: dense compute-matched | AFS flat E=32/128/512 | product-key 64x64 (4096 rows).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
P=/root/trainenv/bin/python
S=/tmp/phic-pretrain-repro/afs/probe/probe_train.py
COMMON="--exposures 5 --bs 32 --steps 2500"
run() { CUDA_VISIBLE_DEVICES=$1 $P $S $2 $COMMON --tag "$3" 2>&1 | grep -aE '^\[|RESULT'; }
( run 0 "--ffn dense --dense-f 450" "N2k_dense"     ; run 0 "--ffn dense --dense-f 450 --n-persons 20000" "N20k_dense" ) &
( run 1 "--ffn afs --E 32  --n-persons 2000" "N2k_afs32"   ; run 1 "--ffn afs --E 32  --n-persons 200000" "N200k_afs32" ) &
( run 2 "--ffn afs --E 128 --n-persons 2000" "N2k_afs128"  ; run 2 "--ffn afs --E 128 --n-persons 200000" "N200k_afs128" ) &
( run 3 "--ffn pk  --E 4096 --n-persons 2000" "N2k_pk4096" ; run 3 "--ffn pk  --E 4096 --n-persons 200000" "N200k_pk4096" ) &
wait
echo MATRIX_DONE
