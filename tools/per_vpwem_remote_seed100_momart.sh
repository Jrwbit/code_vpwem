#!/usr/bin/env bash
set -uo pipefail

user_root=/home/jrw
repo=$user_root/code_vpwem
python_bin=$user_root/anaconda3/envs/vpwem/bin/python
data_root=$user_root/vpwem_data/momart_emb_8layer
output_root=$user_root/vpwem_outputs/per_vpwem_screen_seed100/momart
log_root=$user_root/vpwem_logs/per_vpwem_screen_seed100_momart_gpu6
gpu=6
train_steps=500000
max_workers=2

source "$repo/tools/gpu_queue_policy.sh"

mkdir -p "$output_root" "$log_root/jobs"
exec 9>"$user_root/.per-vpwem-screen-momart-gpu6.lock"
flock -n 9 || exit 0
exec > >(tee -a "$log_root/supervisor.log") 2>&1

export CUDA_VISIBLE_DEVICES=$gpu
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

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

run_job() {
  local env_name=$1 env_type=$2 name work_dir final_model log_file status
  name=${env_name}_${env_type}
  work_dir=$output_root/$name/100
  final_model=$work_dir/models/model_${train_steps}.pt
  log_file=$log_root/jobs/${name}.log
  if [[ -s "$final_model" ]]; then
    printf '%s SKIP %s existing=%s\n' "$(date -Is)" "$name" "$final_model"
    return 0
  fi
  mkdir -p "$work_dir"
  printf '%s START %s seed=100 target=%s gpu=%s\n' \
    "$(date -Is)" "$name" "$train_steps" "$gpu"
  "$python_bin" pipelines/longmemdpptp_momart_image_emb.py \
    --config-name=momart_emb \
    dataset_path="$data_root/$name.hdf5" save_path="$output_root" \
    env_name="$env_name" env_type="$env_type" \
    work_dir="$work_dir" hydra.run.dir="$work_dir" seed=100 \
    gradient_steps="$train_steps" save_freq=100000 log_freq=1000 \
    diffusion_num_layers=8 device=cuda:0 \
    mem_compress_method=per mem_compress_length=2 \
    dataset_memory_subsample_ratio=20 qformer_num_layers=2 \
    qformer_cache_max_length=8 +inference_action_steps=8 \
    +per_evidence_length=0 +per_event_score=delta \
    +per_event_threshold=0.05 +per_max_write_interval=20 \
    +per_merge_similarity=0.98 +per_score_decay=0.995 \
    +per_pos_emb_scale=1.0 +memory_aux_coeff=0.0 \
    +counterfactual_memory_coeff=0.0 \
    +counterfactual_memory_margin=0.01 \
    +counterfactual_memory_probability=0.0 +keep_checkpoints=3 \
    >> "$log_file" 2>&1
  status=$?
  if (( status == 0 )); then
    printf '%s COMPLETE %s\n' "$(date -Is)" "$name"
  else
    printf '%s FAILED %s status=%s\n' "$(date -Is)" "$name" "$status"
  fi
  return "$status"
}

cd "$repo" || exit 1
wait_for_gpu_turn "$gpu"
active=0
failures=0
declare -A pid_names=()
for job in "${jobs[@]}"; do
  read -r env_name env_type <<< "$job"
  run_job "$env_name" "$env_type" &
  pid=$!
  pid_names[$pid]=${env_name}_${env_type}
  active=$((active + 1))
  if (( active >= max_workers )); then
    finished=
    if wait -n -p finished; then status=0; else status=$?; fi
    printf '%s REAP %s status=%s\n' \
      "$(date -Is)" "${pid_names[$finished]:-unknown}" "$status"
    unset 'pid_names[$finished]'
    active=$((active - 1))
    (( status == 0 )) || failures=$((failures + 1))
  fi
done
while (( active > 0 )); do
  finished=
  if wait -n -p finished; then status=0; else status=$?; fi
  printf '%s REAP %s status=%s\n' \
    "$(date -Is)" "${pid_names[$finished]:-unknown}" "$status"
  unset 'pid_names[$finished]'
  active=$((active - 1))
  (( status == 0 )) || failures=$((failures + 1))
done
printf '%s QUEUE_FINISHED failures=%s\n' "$(date -Is)" "$failures"
date -Is > "$log_root/QUEUE_FINISHED"
exit "$((failures > 0))"
