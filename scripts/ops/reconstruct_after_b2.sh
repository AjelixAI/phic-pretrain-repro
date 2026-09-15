#!/bin/bash
cd /root/phi
CKPT=/root/phi/ckpt_pretrain_tied-22L512d-b32-r64-midB2-dolmino-3000M.pt
while [ ! -f $CKPT ]; do sleep 60; done
sleep 30
source /root/.wandb_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== RECONSTRUCT final (base-64, faithful) $(date) ===" >> /root/phi/arm_results.md
/root/venv/bin/torchrun --nproc_per_node=8 /root/phi/pretrain_tb_fast.py --mode tied --init-from /root/phi/ckpt_tied-22L512d-b32-r64_step50000.pt --steps 2991 --bs 32 --seq 2048 --d 512 --layers 22 --ffn 1792 --block 32 --rank 64 --lr 8.5e-5 --warmup 1 --decay-start 0 --project phic-midtrain --cache /root/phi/data_cache_olmomix_shuf.pt --no-ckpt --chunked-ce 8192 --rope-base 64 --tag-suffix=-reconstructed-final --force-save > reconstruct_final.log 2>&1
CUDA_VISIBLE_DEVICES=0 /root/venv/bin/python export_llama.py /root/phi/ckpt_pretrain_tied-22L512d-b32-r64-reconstructed-final.pt /root/phi/export_reconstructed > export_reconstructed.log 2>&1
grep -E 'agreement|VERIFIED' export_reconstructed.log >> /root/phi/arm_results.md
CUDA_VISIBLE_DEVICES=0 /root/venv/bin/lm_eval --model hf --model_args pretrained=/root/phi/export_reconstructed,dtype=bfloat16 --tasks lambada_openai,piqa,hellaswag,arc_easy --batch_size auto --device cuda:0 --output_path /root/phi/lmeval_reconstructed > lmeval_reconstructed.log 2>&1
python3 - <<'PY' >> /root/phi/arm_results.md
import json, glob
fs = sorted(glob.glob('/root/phi/lmeval_reconstructed/**/results*.json', recursive=True))
r = json.load(open(fs[-1]))['results']
print('| reconstructed-final (base64) | 27.8B |', end=' ')
for t in ['arc_easy', 'hellaswag', 'piqa', 'lambada_openai']:
    v = r.get(t, {})
    row = {k.split(',')[0]: round(val*100, 1) for k, val in v.items() if k.startswith('acc') and 'stderr' not in k}
    print(f"{row.get('acc')}/{row.get('acc_norm')}", end=' | ')
print()
PY
echo '=== RECONSTRUCTION DONE ===' >> /root/phi/arm_results.md
