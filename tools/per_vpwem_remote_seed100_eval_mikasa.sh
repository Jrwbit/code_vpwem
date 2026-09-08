#!/usr/bin/env bash
set -uo pipefail

user_root=/home/jrw
repo=$user_root/code_vpwem
python_bin=$user_root/anaconda3/envs/vpwem/bin/python
data_root=$user_root/vpwem_data/mikasa
per_root=$user_root/vpwem_outputs/per_vpwem_screen_seed100/mikasa
fifo_control_root=$user_root/vpwem_outputs/per_vpwem_screen_seed100/fifo_controls/mikasa
eval_root=$user_root/vpwem_outputs/per_vpwem_screen_seed100/eval/mikasa
log_root=$user_root/vpwem_logs/per_vpwem_screen_seed100_eval_mikasa_rebalanced_gpu6
gpu=6
train_steps=400000
eval_seed=20260807
eval_episodes=100
num_envs=10

source "$repo/tools/gpu_queue_policy.sh"

mkdir -p "$fifo_control_root" "$eval_root" "$log_root/jobs"
exec 9>"$user_root/.per-vpwem-screen-eval-mikasa-rebalanced-gpu6.lock"
flock -n 9 || exit 0
exec > >(tee -a "$log_root/supervisor.log") 2>&1

export CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

jobs=(
  "remember_color_3 RememberColor3-v0 5"
  "shell_game_touch ShellGameTouch-v0 5"
  "remember_color_5 RememberColor5-v0 20"
  "intercept_grab_medium InterceptGrabMedium-v0 20"
)

wait_for_file() {
  local path=$1 label=$2
  while [[ ! -s "$path" ]]; do
    printf '%s WAIT %s path=%s\n' "$(date -Is)" "$label" "$path"
    sleep 60
  done
}

fifo_model_path() {
  case "$1" in
    remember_color_3)
      echo "$user_root/vpwem_outputs/remember_color_3/ours/mikasa/RememberColor3-v0/obs_2_act_8/8layer_5_qformer_2_sin_pos_2_fifo_8_0.0001/100/models/model_400000.pt"
      ;;
    shell_game_touch)
      echo "$user_root/vpwem_outputs/shell_game_touch/ours/mikasa/ShellGameTouch-v0/obs_2_act_8/8layer_5_qformer_2_sin_pos_2_fifo_8_0.0001/100/models/model_400000.pt"
      ;;
    remember_color_5|intercept_grab_medium)
      local env_name=$2
      echo "$fifo_control_root/$env_name/100/models/model_400000.pt"
      ;;
  esac
}

train_missing_fifo() {
  local config=$1 env_name=$2 model
  model=$(fifo_model_path "$config" "$env_name")
  [[ -s "$model" ]] && return 0
  local work_dir=$fifo_control_root/$env_name/100
  mkdir -p "$work_dir"
  wait_for_gpu_turn "$gpu"
  printf '%s FIFO_TRAIN_START %s\n' "$(date -Is)" "$env_name"
  "$python_bin" pipelines/longmemdpptp_mikasa_image.py \
    --config-name="$config" dataset_path="$data_root/$env_name" \
    save_path="$fifo_control_root" work_dir="$work_dir" \
    hydra.run.dir="$work_dir" seed=100 gradient_steps="$train_steps" \
    save_freq=20000 log_freq=1000 diffusion_num_layers=8 device=cuda:0 \
    mem_compress_method=qformer mem_compress_length=2 \
    qformer_num_layers=2 qformer_cache_compress=fifo \
    qformer_cache_max_length=8 +inference_action_steps=8 \
    +keep_checkpoints=3 \
    >> "$log_root/jobs/${config}_fifo_train.log" 2>&1
  local status=$?
  printf '%s FIFO_TRAIN_EXIT %s status=%s\n' \
    "$(date -Is)" "$env_name" "$status"
  return "$status"
}

eval_one() {
  local config=$1 env_name=$2 subsample=$3 method=$4 model=$5
  local work_dir=$eval_root/$env_name/$method
  local result=$work_dir/result.npy
  local log=$log_root/jobs/${config}_${method}_eval.log
  [[ -s "$result" ]] && return 0
  mkdir -p "$work_dir"
  wait_for_gpu_turn "$gpu"
  local memory_args=(
    mem_compress_method=qformer mem_compress_length=2
    qformer_num_layers=2 qformer_cache_compress=fifo
    qformer_cache_max_length=8
  )
  if [[ "$method" == per ]]; then
    memory_args=(
      mem_compress_method=per mem_compress_length=2
      qformer_num_layers=2 qformer_cache_max_length=8
      +per_evidence_length=0 +per_event_score=delta
      +per_event_threshold=0.05 +per_max_write_interval=20
      +per_merge_similarity=0.98 +per_score_decay=0.995
      +per_pos_emb_scale=1.0
    )
  fi
  printf '%s EVAL_START %s method=%s episodes=%s seed=%s\n' \
    "$(date -Is)" "$env_name" "$method" "$eval_episodes" "$eval_seed"
  "$python_bin" pipelines/longmemdpptp_mikasa_image_eval.py \
    --config-name="$config" dataset_path="$data_root/$env_name" \
    seed="$eval_seed" +ckpt="$train_steps" model_path="$model" \
    +result_path="$result" num_envs="$num_envs" \
    eval_episodes="$eval_episodes" save_video=false \
    diffusion_num_layers=8 device=cuda:0 \
    dataset_memory_subsample_ratio="$subsample" \
    +inference_action_steps=8 "${memory_args[@]}" \
    work_dir="$work_dir" hydra.run.dir="$work_dir" \
    >> "$log" 2>&1
  local status=$?
  if (( status == 0 )) && [[ -s "$result" ]]; then
    printf '%s EVAL_COMPLETE %s method=%s\n' \
      "$(date -Is)" "$env_name" "$method"
  else
    printf '%s EVAL_FAILED %s method=%s status=%s\n' \
      "$(date -Is)" "$env_name" "$method" "$status"
    return 1
  fi
}

cd "$repo" || exit 1
printf '%s QUEUE_START protocol_seed=%s episodes=%s\n' \
  "$(date -Is)" "$eval_seed" "$eval_episodes"

# Wait until every PER run has reached its frozen final checkpoint.
for job in "${jobs[@]}"; do
  read -r config env_name subsample <<< "$job"
  wait_for_file "$per_root/$env_name/100/models/model_400000.pt" \
    "PER:$env_name"
done

failures=0

# RememberColor5 is trained by the already-running GPU6 worker. Do not begin
# evaluations until that worker has released GPU6.
wait_for_file \
  "$fifo_control_root/RememberColor5-v0/100/models/model_400000.pt" \
  "FIFO:RememberColor5-v0"

for job in "${jobs[@]}"; do
  read -r config env_name subsample <<< "$job"
  per_model=$per_root/$env_name/100/models/model_400000.pt
  fifo_model=$(fifo_model_path "$config" "$env_name")
  wait_for_file "$fifo_model" "FIFO:$env_name"
  eval_one "$config" "$env_name" "$subsample" fifo "$fifo_model" \
    || failures=$((failures + 1))
  eval_one "$config" "$env_name" "$subsample" per "$per_model" \
    || failures=$((failures + 1))
done

printf '%s QUEUE_FINISHED failures=%s\n' "$(date -Is)" "$failures"
date -Is > "$log_root/QUEUE_FINISHED"
exit "$((failures > 0))"
