#!/usr/bin/env bash
set -uo pipefail

user_root=/home/jrw
repo=$user_root/code_vpwem
python_bin=$user_root/anaconda3/envs/vpwem/bin/python
data_root=$user_root/vpwem_data/mikasa
output_root=$user_root/vpwem_outputs/per_vpwem_screen_seed100/mikasa
log_root=$user_root/vpwem_logs/per_vpwem_screen_seed100_mikasa_gpu3
gpu=3

source "$repo/tools/gpu_queue_policy.sh"
train_steps=400000

mkdir -p "$output_root" "$log_root/jobs"
exec 9>"$user_root/.per-vpwem-screen-mikasa-gpu3.lock"
flock -n 9 || exit 0
exec > >(tee -a "$log_root/supervisor.log") 2>&1
export CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

jobs=(
  "shell_game_touch ShellGameTouch-v0"
  "intercept_grab_medium InterceptGrabMedium-v0"
)

cd "$repo" || exit 1
wait_for_gpu_turn "$gpu"
for job in "${jobs[@]}"; do
  read -r config env_name <<< "$job"
  work_dir=$output_root/$env_name/100
  final_model=$work_dir/models/model_${train_steps}.pt
  [[ -s "$final_model" ]] && continue
  mkdir -p "$work_dir"
  printf '%s START %s seed=100 target=%s gpu=%s\n' \
    "$(date -Is)" "$env_name" "$train_steps" "$gpu"
  "$python_bin" pipelines/longmemdpptp_mikasa_image.py \
    --config-name="$config" dataset_path="$data_root/$env_name" \
    save_path="$output_root" work_dir="$work_dir" hydra.run.dir="$work_dir" \
    seed=100 gradient_steps="$train_steps" save_freq=20000 log_freq=1000 \
    diffusion_num_layers=8 device=cuda:0 mem_compress_method=per \
    mem_compress_length=2 qformer_num_layers=2 qformer_cache_max_length=8 \
    +inference_action_steps=8 +per_evidence_length=0 \
    +per_event_score=delta +per_event_threshold=0.05 \
    +per_max_write_interval=20 +per_merge_similarity=0.98 \
    +per_score_decay=0.995 +per_pos_emb_scale=1.0 \
    +memory_aux_coeff=0.0 +counterfactual_memory_coeff=0.0 \
    +counterfactual_memory_margin=0.01 \
    +counterfactual_memory_probability=0.0 +keep_checkpoints=3 \
    >> "$log_root/jobs/${config}.log" 2>&1
  printf '%s EXIT %s status=%s\n' "$(date -Is)" "$env_name" "$?"
done
date -Is > "$log_root/QUEUE_FINISHED"
