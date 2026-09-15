#!/bin/bash
cd /root/phi
for ARM in midB2-dolmino-3000M midAFULL-dolmino-6810M; do
  CKPT=/root/phi/ckpt_pretrain_tied-22L512d-b32-r64-$ARM.pt
  while [ ! -f $CKPT ]; do sleep 60; done
  sleep 30
  echo "=== EVAL $ARM $(date) ===" >> /root/phi/arm_results.md
  CUDA_VISIBLE_DEVICES=0 /root/venv/bin/python export_llama.py $CKPT /root/phi/export_$ARM > export_$ARM.log 2>&1
  grep -E 'agreement|VERIFIED' export_$ARM.log >> /root/phi/arm_results.md
  CUDA_VISIBLE_DEVICES=0 /root/venv/bin/lm_eval --model hf \
    --model_args pretrained=/root/phi/export_$ARM,dtype=bfloat16 \
    --tasks lambada_openai,piqa,hellaswag,arc_easy --batch_size auto \
    --device cuda:0 --output_path /root/phi/lmeval_$ARM > lmeval_$ARM.log 2>&1
  for i in 1 2 3 4 5 6; do
    python3 - <<PY >> /root/phi/arm_results.md && break
import json, glob, sys, time
fs = sorted(glob.glob('/root/phi/lmeval_/**/results*.json', recursive=True))
assert fs, 'results not written yet'
r = json.load(open(fs[-1]))['results']
print('|  |', end=' ')
for t in ['arc_easy', 'hellaswag', 'piqa', 'lambada_openai']:
    v = r.get(t, {})
    row = {k.split(',')[0]: round(val*100, 1) for k, val in v.items() if k.startswith('acc') and 'stderr' not in k}
    print(f"{row.get('acc')}/{row.get('acc_norm')}", end=' | ')
print()
PY
    sleep 30
  done
done
echo '=== ALL ARM EVALS DONE ===' >> /root/phi/arm_results.md
