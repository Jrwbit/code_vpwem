#!/usr/bin/env bash
set -uo pipefail

# Seed-200 confirmation for the six tasks where delta-PER underperformed FIFO
# at training seed 100. Run one independent supervisor per reserved GPU:
#   bash tools/per_vpwem_seed200_negative6.sh gpu6
#   bash tools/per_vpwem_seed200_negative6.sh gpu7

mode=${1:-}
if [[ "$mode" != gpu6 && "$mode" != gpu7 ]]; then
  echo "usage: $0 gpu6|gpu7" >&2
  exit 2
fi

user_root=/home/jrw
repo=$user_root/code_vpwem
python_bin=$user_root/anaconda3/envs/vpwem/bin/python
train_seed=200
eval_seed=20260807
eval_episodes=100
num_envs=10
per_root=$user_root/vpwem_outputs/per_vpwem_screen_seed200
eval_root=$per_root/eval
log_root=$user_root/vpwem_logs/per_vpwem_screen_seed200_negative6/$mode
momart_emb_root=$user_root/vpwem_data/momart_emb_8layer
momart_raw_root=$user_root/vpwem_data/momart
momart_fifo_root=$user_root/vpwem_outputs/ours/momart_emb
momart_context_root=$user_root/vpwem_outputs/momart/dpptp
momart_signature=obs_2_act_8/8layer_20_qformer_2_sin_pos_2_fifo_8

if [[ "$mode" == gpu6 ]]; then gpu=6; else gpu=7; fi

mkdir -p "$per_root/mikasa" "$per_root/momart" "$per_root/fifo_controls/momart" \
  "$eval_root/mikasa" "$eval_root/momart" "$log_root/jobs"
exec 9>"$user_root/.per-vpwem-seed200-negative6-$mode.lock"
flock -n 9 || exit 0
exec > >(tee -a "$log_root/supervisor.log") 2>&1

export CUDA_VISIBLE_DEVICES=$gpu
export GIBSON_DEVICE_ID=$gpu
export WANDB_MODE=offline HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

log() { printf '%s %s\n' "$(date -Is)" "$*"; }

train_shell_per() {
  local work=$per_root/mikasa/ShellGameTouch-v0/$train_seed
  local model=$work/models/model_400000.pt
  [[ -s "$model" ]] && { log "SKIP_TRAIN shell_game_touch existing=$model"; return 0; }
  mkdir -p "$work"
  log "TRAIN_START shell_game_touch method=per seed=$train_seed gpu=$gpu"
  "$python_bin" pipelines/longmemdpptp_mikasa_image.py \
    --config-name=shell_game_touch \
    dataset_path="$user_root/vpwem_data/mikasa/ShellGameTouch-v0" \
    save_path="$per_root/mikasa" work_dir="$work" hydra.run.dir="$work" \
    seed="$train_seed" gradient_steps=400000 save_freq=20000 log_freq=1000 \
    diffusion_num_layers=8 device=cuda:0 mem_compress_method=per \
    mem_compress_length=2 dataset_memory_subsample_ratio=5 \
    qformer_num_layers=2 qformer_cache_max_length=8 \
    +inference_action_steps=8 +per_evidence_length=0 \
    +per_event_score=delta +per_event_threshold=0.05 \
    +per_max_write_interval=20 +per_merge_similarity=0.98 \
    +per_score_decay=0.995 +per_pos_emb_scale=1.0 \
    +memory_aux_coeff=0.0 +counterfactual_memory_coeff=0.0 \
    +counterfactual_memory_margin=0.01 \
    +counterfactual_memory_probability=0.0 +keep_checkpoints=3 \
    >> "$log_root/jobs/shell_game_touch_per_train.log" 2>&1
  local status=$?
  log "TRAIN_EXIT shell_game_touch method=per status=$status"
  (( status == 0 )) && [[ -s "$model" ]]
}

train_momart() {
  local env_name=$1 env_type=$2 method=$3
  local name=${env_name}_${env_type} work model
  if [[ "$method" == per ]]; then
    work=$per_root/momart/$name/$train_seed
  else
    work=$per_root/fifo_controls/momart/$name/$train_seed
  fi
  model=$work/models/model_500000.pt
  [[ -s "$model" ]] && { log "SKIP_TRAIN $name method=$method existing=$model"; return 0; }
  mkdir -p "$work"
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
      +per_pos_emb_scale=1.0 +memory_aux_coeff=0.0
      +counterfactual_memory_coeff=0.0
      +counterfactual_memory_margin=0.01
      +counterfactual_memory_probability=0.0
    )
  fi
  log "TRAIN_START $name method=$method seed=$train_seed gpu=$gpu"
  "$python_bin" pipelines/longmemdpptp_momart_image_emb.py \
    --config-name=momart_emb dataset_path="$momart_emb_root/$name.hdf5" \
    save_path="$per_root/$method" env_name="$env_name" env_type="$env_type" \
    work_dir="$work" hydra.run.dir="$work" seed="$train_seed" \
    gradient_steps=500000 save_freq=100000 log_freq=1000 \
    diffusion_num_layers=8 device=cuda:0 dataset_memory_subsample_ratio=20 \
    +inference_action_steps=8 +keep_checkpoints=3 "${memory_args[@]}" \
    >> "$log_root/jobs/${name}_${method}_train.log" 2>&1
  local status=$?
  log "TRAIN_EXIT $name method=$method status=$status"
  (( status == 0 )) && [[ -s "$model" ]]
}

momart_fifo_model() {
  local name=$1
  if [[ "$name" == table_cleanup_to_sink_expert ]]; then
    echo "$per_root/fifo_controls/momart/$name/$train_seed/models/model_500000.pt"
  else
    echo "$momart_fifo_root/$name/$momart_signature/$train_seed/models/model_500000.pt"
  fi
}

eval_shell() {
  local method=$1 model=$2
  local work=$eval_root/mikasa/ShellGameTouch-v0/$method
  local result=$work/result.npy
  [[ -s "$result" ]] && { log "SKIP_EVAL shell_game_touch method=$method"; return 0; }
  mkdir -p "$work"
  local memory_args=(
    mem_compress_method=qformer mem_compress_length=2
    qformer_num_layers=2 qformer_cache_compress=fifo qformer_cache_max_length=8
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
  log "EVAL_START shell_game_touch method=$method episodes=$eval_episodes"
  "$python_bin" pipelines/longmemdpptp_mikasa_image_eval.py \
    --config-name=shell_game_touch \
    dataset_path="$user_root/vpwem_data/mikasa/ShellGameTouch-v0" \
    seed="$eval_seed" +ckpt=400000 model_path="$model" \
    +result_path="$result" num_envs="$num_envs" \
    eval_episodes="$eval_episodes" save_video=false \
    diffusion_num_layers=8 device=cuda:0 dataset_memory_subsample_ratio=5 \
    +inference_action_steps=8 "${memory_args[@]}" \
    work_dir="$work" hydra.run.dir="$work" \
    >> "$log_root/jobs/shell_game_touch_${method}_eval.log" 2>&1
  local status=$?
  log "EVAL_EXIT shell_game_touch method=$method status=$status"
  (( status == 0 )) && [[ -s "$result" ]]
}

eval_momart() {
  local env_name=$1 env_type=$2 method=$3 model=$4
  local name=${env_name}_${env_type}
  local work=$eval_root/momart/$name/$method
  local result=$work/result.npy
  local context_model=$momart_context_root/$name/obs_2_act_8_8layer/100/models/model_1000000.pt
  [[ -s "$result" ]] && { log "SKIP_EVAL $name method=$method"; return 0; }
  mkdir -p "$work"
  local memory_args=(
    mem_compress_method=qformer mem_compress_length=2
    qformer_num_layers=2 qformer_cache_compress=fifo qformer_cache_max_length=8
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
  [[ -s "$model" && -s "$context_model" ]] || {
    log "MISSING_EVAL_INPUT $name method=$method model=$model context=$context_model"
    return 1
  }
  log "EVAL_START $name method=$method episodes=$eval_episodes"
  "$python_bin" pipelines/longmemdpptp_momart_image_emb_eval.py \
    --config-name=momart dataset_path="$momart_raw_root/$name.hdf5" \
    env_name="$env_name" env_type="$env_type" seed="$eval_seed" \
    obs_steps=2 action_steps=8 horizon=9 num_envs="$num_envs" \
    eval_episodes="$eval_episodes" save_video=false \
    diffusion_num_layers=8 device=cuda:0 dataset_memory_type=random_from_full \
    memory_pos_emb=sin_pos dataset_memory_subsample_ratio=20 \
    short_cond_dropout=0.3 long_cond_dropout=0.3 \
    +context_model_path="$context_model" +ckpt=500000 \
    model_path="$model" +result_path="$result" \
    +inference_action_steps=8 "${memory_args[@]}" \
    work_dir="$work" hydra.run.dir="$work" \
    >> "$log_root/jobs/${name}_${method}_eval.log" 2>&1
  local status=$?
  log "EVAL_EXIT $name method=$method status=$status"
  (( status == 0 )) && [[ -s "$result" ]]
}

eval_momart_pair() {
  local env_name=$1 env_type=$2
  local name=${env_name}_${env_type}
  local fifo per
  fifo=$(momart_fifo_model "$name")
  per=$per_root/momart/$name/$train_seed/models/model_500000.pt
  eval_momart "$env_name" "$env_type" fifo "$fifo"
  eval_momart "$env_name" "$env_type" per "$per"
}

run_train_pair() {
  local e1=$1 t1=$2 m1=$3 e2=$4 t2=$5 m2=$6
  train_momart "$e1" "$t1" "$m1" & local p1=$!
  train_momart "$e2" "$t2" "$m2" & local p2=$!
  local failed=0
  wait "$p1" || failed=1
  wait "$p2" || failed=1
  return "$failed"
}

cd "$repo" || exit 1
log "QUEUE_START mode=$mode gpu=$gpu train_seed=$train_seed eval_seed=$eval_seed"
failures=0

if [[ "$mode" == gpu6 ]]; then
  train_shell_per || failures=$((failures + 1))
  shell_fifo=$user_root/vpwem_outputs/shell_game_touch/ours/mikasa/ShellGameTouch-v0/obs_2_act_8/8layer_5_qformer_2_sin_pos_2_fifo_8_0.0001/$train_seed/models/model_400000.pt
  shell_per=$per_root/mikasa/ShellGameTouch-v0/$train_seed/models/model_400000.pt
  eval_shell fifo "$shell_fifo" || failures=$((failures + 1))
  eval_shell per "$shell_per" || failures=$((failures + 1))

  run_train_pair \
    unload_dishwasher_to_dresser expert per \
    table_cleanup_to_dishwasher expert per \
    || failures=$((failures + 1))
  eval_momart_pair unload_dishwasher_to_dresser expert \
    || failures=$((failures + 1))
  eval_momart_pair table_cleanup_to_dishwasher expert \
    || failures=$((failures + 1))
else
  run_train_pair \
    table_cleanup_to_dishwasher suboptimal per \
    table_cleanup_to_sink expert per \
    || failures=$((failures + 1))
  eval_momart_pair table_cleanup_to_dishwasher suboptimal \
    || failures=$((failures + 1))

  # The seed-200 FIFO checkpoint for sink/expert is absent. Train it together
  # with the last PER task after the first pair has been evaluated.
  run_train_pair \
    table_setup_from_dresser expert per \
    table_cleanup_to_sink expert fifo \
    || failures=$((failures + 1))
  eval_momart_pair table_cleanup_to_sink expert \
    || failures=$((failures + 1))
  eval_momart_pair table_setup_from_dresser expert \
    || failures=$((failures + 1))
fi

log "QUEUE_FINISHED mode=$mode failures=$failures"
date -Is > "$log_root/QUEUE_FINISHED"
exit "$((failures > 0))"
