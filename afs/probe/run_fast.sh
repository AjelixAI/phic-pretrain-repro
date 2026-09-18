#!/bin/bash
# FAST tiny-scale capacity matrix: d=128/L=2, 5 arms x 3 N-points, 7 GPUs parallel.
# Question: does AFS recall/param hold as E grows, vs the compute-matched dense?
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
P=/root/trainenv/bin/python
S=/tmp/phic-pretrain-repro/afs/probe/probe_train.py
T="--d 128 --L 2 --nh 2 --f-row 128 --dense-f 225 --bs 32 --hard-k 2"
run() { CUDA_VISIBLE_DEVICES=$1 $P $S $2 $3 $T --tag "$4" 2>&1 | grep -aE 'RESULT|Error'; }
( run 0 "--ffn dense"          "--n-persons 1000 --steps 1320" "f_N1k_dense"  ; \
  run 0 "--ffn dense"          "--n-persons 4000 --steps 5270" "f_N4k_dense"  ; \
  run 0 "--ffn dense --exposures 100 --n-persons 16000 --steps 7000" "f_N16k_dense" ) &
( run 1 "--ffn afs --E 8"      "--n-persons 1000 --steps 1320" "f_N1k_afs8"   ; \
  run 1 "--ffn afs --E 8"      "--n-persons 4000 --steps 5270" "f_N4k_afs8"   ; \
  run 1 "--ffn afs --E 8 --exposures 100 --n-persons 16000 --steps 7000" "f_N16k_afs8" ) &
( run 2 "--ffn afs --E 32"     "--n-persons 1000 --steps 1320" "f_N1k_afs32"  ; \
  run 2 "--ffn afs --E 32"     "--n-persons 4000 --steps 5270" "f_N4k_afs32"  ; \
  run 2 "--ffn afs --E 32 --exposures 100 --n-persons 16000 --steps 7000" "f_N16k_afs32" ) &
( run 3 "--ffn afs --E 128"    "--n-persons 1000 --steps 1320" "f_N1k_afs128" ; \
  run 3 "--ffn afs --E 128"    "--n-persons 4000 --steps 5270" "f_N4k_afs128" ; \
  run 3 "--ffn afs --E 128 --exposures 100 --n-persons 16000 --steps 7000" "f_N16k_afs128" ) &
( run 5 "--ffn pk --E 1024 --bs 16" "--n-persons 1000 --steps 1320" "f_N1k_pk"   ; \
  run 5 "--ffn pk --E 1024 --bs 16" "--n-persons 4000 --steps 5270" "f_N4k_pk"   ; \
  run 5 "--ffn pk --E 1024 --bs 16 --exposures 100 --n-persons 16000 --steps 7000" "f_N16k_pk" ) &
( run 6 "--ffn afs --E 512"    "--n-persons 1000 --steps 1320" "f_N1k_afs512" ; \
  run 6 "--ffn afs --E 512"    "--n-persons 4000 --steps 5270" "f_N4k_afs512" ; \
  run 6 "--ffn afs --E 512 --exposures 100 --n-persons 16000 --steps 7000" "f_N16k_afs512" ) &
( run 7 "--ffn afs --E 32 --hard-k 0" "--n-persons 1000 --steps 1320" "f_N1k_afs32soft" ; \
  run 7 "--ffn afs --E 32 --hard-k 0" "--n-persons 4000 --steps 5270" "f_N4k_afs32soft" ) &
wait
echo FAST_DONE
