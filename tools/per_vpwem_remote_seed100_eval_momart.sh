#!/usr/bin/env bash
set -uo pipefail

user_root=/home/jrw
repo=$user_root/code_vpwem
python_bin=$user_root/anaconda3/envs/vpwem/bin/python
raw_data_root=$user_root/vpwem_data/momart
per_root=$user_root/vpwem_outputs/per_vpwem_screen_seed100/momart
fifo_root=$user_root/vpwem_outputs/ours/momart_emb
context_root=$user_root/vpwem_outputs/momart/dpptp
eval_root=$user_root/vpwem_outputs/per_vpwem_screen_seed100/eval/momart
log_root=$user_root/vpwem_logs/per_vpwem_screen_seed100_eval_momart_gpu7
gpu=7
train_steps=500000
eval_seed=20260807
eval_episodes=100
num_envs=10
signature=obs_2_act_8/8layer_20_qformer_2_sin_pos_2_fifo_8

source "$repo/tools/gpu_queue_policy.sh"

mkdir -p "$eval_root" "$log_root/jobs"
exec 9>"$user_root/.per-vpwem-screen-eval-momart-gpu7.lock"
flock -n 9 || exit 0
exec > >(tee -a "$log_root/supervisor.log") 2>&1

export CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline HYDRA_FULL_ERROR=1
export GIBSON_DEVICE_ID=$gpu
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

jobs=(
  "unload_dishwasher_to_dresser suboptimal"
  "unload_dishwasher_to_dresser expert"
  "table_cleanup_to_sink suboptimal"
  "table_cleanup_to_sink expert"
  "table_cleanup_to_dishwasher suboptimal"
  "table_cleanup_to_dishwasher expert"
  "table_setup_from_dishwasher suboptimal"
  "table_setup_from_dishwasher expert"
  "table_setup_from_dresser suboptimal"
  "table_setup_from_dresser expert"
)

wait_for_file() {
  local path=$1 label=$2
  while [[ ! -s "$path" ]]; do
    printf '%s WAIT %s path=%s\n' "$(date -Is)" "$label" "$path"
    sleep 60
  done
}

eval_one() {
  local env_name=$1 env_type=$2 method=$3 model=$4
  local name=${env_name}_${env_type}
  local work_dir=$eval_root/$name/$method
  local result=$work_dir/result.npy
  local log=$log_root/jobs/${name}_${method}_eval.log
  local context_model=$context_root/$name/obs_2_act_8_8layer/100/models/model_1000000.pt
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
    "$(date -Is)" "$name" "$method" "$eval_episodes" "$eval_seed"
  "$python_bin" pipelines/longmemdpptp_momart_image_emb_eval.py \
    --config-name=momart dataset_path="$raw_data_root/$name.hdf5" \
    env_name="$env_name" env_type="$env_type" seed="$eval_seed" \
    obs_steps=2 action_steps=8 horizon=9 num_envs="$num_envs" \
    eval_episodes="$eval_episodes" save_video=false \
    diffusion_num_layers=8 device=cuda:0 \
    dataset_memory_type=random_from_full memory_pos_emb=sin_pos \
    dataset_memory_subsample_ratio=20 \
    short_cond_dropout=0.3 long_cond_dropout=0.3 \
    +context_model_path="$context_model" +ckpt="$train_steps" \
    model_path="$model" +result_path="$result" \
    +inference_action_steps=8 "${memory_args[@]}" \
    work_dir="$work_dir" hydra.run.dir="$work_dir" \
    >> "$log" 2>&1
  local status=$?
  if (( status == 0 )) && [[ -s "$result" ]]; then
    printf '%s EVAL_COMPLETE %s method=%s\n' \
      "$(date -Is)" "$name" "$method"
  else
    printf '%s EVAL_FAILED %s method=%s status=%s\n' \
      "$(date -Is)" "$name" "$method" "$status"
    return 1
  fi
}

cd "$repo" || exit 1
printf '%s QUEUE_START protocol_seed=%s episodes=%s\n' \
  "$(date -Is)" "$eval_seed" "$eval_episodes"

for job in "${jobs[@]}"; do
  read -r env_name env_type <<< "$job"
  name=${env_name}_${env_type}
  wait_for_file "$per_root/$name/100/models/model_500000.pt" "PER:$name"
done

# Serialize behind the two-worker GPU6 training supervisor.
exec 8>"$user_root/.per-vpwem-screen-momart-gpu6.lock"
flock 8

failures=0
for job in "${jobs[@]}"; do
  read -r env_name env_type <<< "$job"
  name=${env_name}_${env_type}
  per_model=$per_root/$name/100/models/model_500000.pt
  fifo_model=$fifo_root/$name/$signature/100/models/model_500000.pt
  context_model=$context_root/$name/obs_2_act_8_8layer/100/models/model_1000000.pt
  if [[ ! -s "$fifo_model" || ! -s "$context_model" ]]; then
    printf '%s MISSING_INPUT %s fifo=%s context=%s\n' \
      "$(date -Is)" "$name" "$fifo_model" "$context_model"
    failures=$((failures + 1))
    continue
  fi
  eval_one "$env_name" "$env_type" fifo "$fifo_model" \
    || failures=$((failures + 1))
  eval_one "$env_name" "$env_type" per "$per_model" \
    || failures=$((failures + 1))
done

printf '%s QUEUE_FINISHED failures=%s\n' "$(date -Is)" "$failures"
date -Is > "$log_root/QUEUE_FINISHED"
exit "$((failures > 0))"
