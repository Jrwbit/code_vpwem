#!/usr/bin/env bash
set -uo pipefail

user_root=/home/jrw
repo=$user_root/code_vpwem
python_bin=$user_root/anaconda3/envs/vpwem/bin/python
data_root=$user_root/vpwem_data/mikasa
output_root=$user_root/vpwem_outputs/per_vpwem_screen_seed100/fifo_controls/mikasa
momart_log_root=$user_root/vpwem_logs/per_vpwem_screen_seed100_eval_momart_gpu7
log_root=$user_root/vpwem_logs/per_vpwem_screen_seed100_fifo_intercept_gpu7
gpu=7
train_steps=400000
env_name=InterceptGrabMedium-v0
config=intercept_grab_medium
work_dir=$output_root/$env_name/100
final_model=$work_dir/models/model_${train_steps}.pt

source "$repo/tools/gpu_queue_policy.sh"

mkdir -p "$work_dir" "$log_root/jobs"
exec 9>"$user_root/.per-vpwem-screen-fifo-intercept-gpu7.lock"
flock -n 9 || exit 0
exec > >(tee -a "$log_root/supervisor.log") 2>&1

export CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

while [[ ! -s "$momart_log_root/QUEUE_FINISHED" ]]; do
  printf '%s WAIT MOMART_GPU7\n' "$(date -Is)"
  sleep 60
done

wait_for_gpu_turn "$gpu"

if [[ -s "$final_model" ]]; then
  printf '%s SKIP existing=%s\n' "$(date -Is)" "$final_model"
  date -Is > "$log_root/QUEUE_FINISHED"
  exit 0
fi

cd "$repo" || exit 1
printf '%s FIFO_TRAIN_START %s gpu=%s target=%s\n' \
  "$(date -Is)" "$env_name" "$gpu" "$train_steps"
"$python_bin" pipelines/longmemdpptp_mikasa_image.py \
  --config-name="$config" dataset_path="$data_root/$env_name" \
  save_path="$output_root" work_dir="$work_dir" \
  hydra.run.dir="$work_dir" seed=100 gradient_steps="$train_steps" \
  save_freq=20000 log_freq=1000 diffusion_num_layers=8 device=cuda:0 \
  mem_compress_method=qformer mem_compress_length=2 \
  qformer_num_layers=2 qformer_cache_compress=fifo \
  qformer_cache_max_length=8 +inference_action_steps=8 \
  +keep_checkpoints=3 \
  >> "$log_root/jobs/${config}_fifo_train.log" 2>&1
status=$?
printf '%s FIFO_TRAIN_EXIT %s status=%s\n' \
  "$(date -Is)" "$env_name" "$status"
if (( status == 0 )) && [[ -s "$final_model" ]]; then
  date -Is > "$log_root/QUEUE_FINISHED"
else
  exit 1
fi
