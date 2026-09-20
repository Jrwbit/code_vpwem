#!/usr/bin/env bash
set -euo pipefail

user_root=/home/jrw
repo=$user_root/code_vpwem
python_bin=$user_root/anaconda3/envs/vpwem/bin/python
output_root=$user_root/vpwem_outputs/vpwem_multitask_seed100
log_root=$user_root/vpwem_logs/vpwem_multitask_seed100

mkdir -p "$output_root" "$log_root"
cd "$repo"

run_suite() {
  local suite=$1 gpu=$2 pipeline=$3 config=$4 steps=$5
  local work_dir=$output_root/$suite/100
  local log_file=$log_root/${suite}.log
  mkdir -p "$work_dir"
  CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline HYDRA_FULL_ERROR=1 \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
    "$python_bin" "$pipeline" --config-name="$config" \
      save_path="$output_root" work_dir="$work_dir" \
      hydra.run.dir="$work_dir" seed=100 device=cuda:0 \
      gradient_steps="$steps" save_freq=100000 log_freq=1000 \
      +keep_checkpoints=3 >> "$log_file" 2>&1
}

run_suite momart 6 pipelines/longmemdpptp_momart_image_emb.py \
  momart_multitask_emb 1000000 &
momart_pid=$!
run_suite mikasa 1 pipelines/longmemdpptp_mikasa_image.py \
  mikasa_multitask 600000 &
mikasa_pid=$!

status=0
wait "$momart_pid" || status=1
wait "$mikasa_pid" || status=1
exit "$status"
