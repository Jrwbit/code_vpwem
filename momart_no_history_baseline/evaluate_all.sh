#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
train_root=${TRAIN_ROOT:?Set TRAIN_ROOT to the seed100 training directory}
data_root=${MOMART_DATA_ROOT:-/home/jrw/vpwem_data/momart}
output_root=${EVAL_OUTPUT_ROOT:-${train_root}/eval_1m}
step=${CHECKPOINT_STEP:-1000000}

checkpoint="${train_root}/models/model_${step}.pt"
config="${train_root}/.hydra/config.yaml"
test -s "$checkpoint"
test -s "$config"
mkdir -p "$output_root"

jobs=(
  'unload_dishwasher_to_dresser suboptimal'
  'unload_dishwasher_to_dresser expert'
  'table_cleanup_to_sink suboptimal'
  'table_cleanup_to_sink expert'
  'table_cleanup_to_dishwasher suboptimal'
  'table_cleanup_to_dishwasher expert'
  'table_setup_from_dishwasher suboptimal'
  'table_setup_from_dishwasher expert'
  'table_setup_from_dresser suboptimal'
  'table_setup_from_dresser expert'
)

for job in "${jobs[@]}"; do
  read -r env_name env_type <<<"$job"
  name="${env_name}_${env_type}"
  result="${output_root}/${name}/result.npy"
  if [[ -s "$result" ]]; then
    printf 'SKIP %s\n' "$name"
    continue
  fi
  mkdir -p "${output_root}/${name}"
  "$here/evaluate.sh" \
    --checkpoint "$checkpoint" --train-config "$config" \
    --dataset-path "${data_root}/${name}.hdf5" \
    --env-name "$env_name" --env-type "$env_type" \
    --work-dir "${output_root}/${name}" --result-path "$result" \
    >"${output_root}/${name}/eval.log" 2>&1
  printf 'DONE %s\n' "$name"
done
