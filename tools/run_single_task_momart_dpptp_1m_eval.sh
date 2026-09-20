#!/usr/bin/env bash
set -euo pipefail

repo=/home/jrw/code_vpwem_momart_fair
python=/home/jrw/anaconda3/envs/vpwem/bin/python
data_root=/home/jrw/vpwem_data/momart
train_root=/home/jrw/vpwem_outputs/momart/dpptp
output_root=/home/jrw/vpwem_outputs/momart/dpptp_single_task_eval_1m
eval_seed=20260917

tasks=(
  table_cleanup_to_dishwasher_expert
  table_cleanup_to_dishwasher_suboptimal
  table_cleanup_to_sink_expert
  table_cleanup_to_sink_suboptimal
  table_setup_from_dishwasher_expert
  table_setup_from_dishwasher_suboptimal
  table_setup_from_dresser_expert
  table_setup_from_dresser_suboptimal
  unload_dishwasher_to_dresser_expert
  unload_dishwasher_to_dresser_suboptimal
)

mkdir -p "$output_root/logs"
exec 9>"$output_root/launcher.lock"
if ! flock -n 9; then
  echo "Another evaluation launcher already holds $output_root/launcher.lock" >&2
  exit 2
fi

run_task() {
  local gpu=$1
  local task=$2
  local env_name=${task%_*}
  local env_type=${task##*_}
  local train_dir="$train_root/$task/obs_2_act_8_8layer/100"
  local result_dir="$output_root/$task"
  local result_file="$result_dir/result_1000000.npy"
  local log_file="$output_root/logs/$task.log"

  if [[ -s "$result_file" ]]; then
    echo "[$(date --iso-8601=seconds)] SKIP $task: result exists"
    return
  fi
  test -s "$train_dir/models/model_1000000.pt"
  test -s "$data_root/$task.hdf5"
  mkdir -p "$result_dir"
  echo "[$(date --iso-8601=seconds)] START GPU=$gpu task=$task"
  (
    cd "$repo"
    CUDA_VISIBLE_DEVICES="$gpu" GIBSON_DEVICE_ID="$gpu" "$python" \
      pipelines/dpptp_momart_image_eval.py --config-name=momart \
      env_name="$env_name" env_type="$env_type" \
      dataset_path="$data_root/$task.hdf5" \
      work_dir="$train_dir" hydra.run.dir="$result_dir" \
      +checkpoint_step=1000000 +eval_seed="$eval_seed" \
      +eval_output_dir="$result_dir" \
      num_envs=5 eval_episodes=30 save_video=false device=cuda:0
  ) >"$log_file" 2>&1
  test -s "$result_file"
  echo "[$(date --iso-8601=seconds)] DONE GPU=$gpu task=$task"
}

run_queue() {
  local gpu=$1
  local first=$2
  local last=$3
  local i
  for ((i=first; i<last; i++)); do
    run_task "$gpu" "${tasks[$i]}"
  done
}

run_queue 2 0 5 &
pid_gpu2=$!
run_queue 3 5 10 &
pid_gpu3=$!

status=0
wait "$pid_gpu2" || status=$?
wait "$pid_gpu3" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "At least one GPU queue failed; inspect $output_root/logs" >&2
  exit "$status"
fi

"$python" "$repo/tools/summarize_single_task_momart_dpptp.py" "$output_root"
echo "[$(date --iso-8601=seconds)] ALL DONE: $output_root/summary.md"
