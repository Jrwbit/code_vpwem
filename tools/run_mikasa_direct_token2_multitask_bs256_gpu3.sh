#!/usr/bin/env bash
set -u

repo=/home/jrw/code_vpwem_fair_history
python_bin=/home/jrw/anaconda3/envs/vpwem/bin/python
work=/home/jrw/vpwem_outputs/direct_token2_real_batch/mikasa_multitask/real_bs256_grad1
mkdir -p "$work"
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"

printf '[%s] START tasks=4 method=direct_token2 physical_bs=256 grad_accum=1 gpu=3 keep_all_checkpoints=true\n' \
  "$(date -Is)" >>"$work/supervisor.log"
CUDA_VISIBLE_DEVICES=3 exec "$python_bin" pipelines/fair_history_mikasa_multitask.py \
  history_method=direct_token2 \
  batch_size=256 \
  gradient_steps=600000 \
  save_freq=100000 \
  keep_checkpoints=1000000 \
  work_dir="$work" \
  hydra.run.dir="$work"
