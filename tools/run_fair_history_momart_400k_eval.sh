#!/usr/bin/env bash
set -uo pipefail

repo=/home/jrw/code_vpwem_momart_fair
python_bin=/home/jrw/anaconda3/envs/vpwem/bin/python
copy_root=/home/jrw/vpwem_checkpoint_copies/active_20260917/home/jrw/vpwem_outputs
output_root=/home/jrw/vpwem_outputs/fair_history_momart_multitask_eval_400k
data_root=/home/jrw/vpwem_data/momart
context_root=/home/jrw/vpwem_outputs/momart/dpptp
seed=${EVAL_SEED:-20260917}
episodes=${EVAL_EPISODES:-30}
num_envs=${NUM_ENVS:-10}

mkdir -p "${output_root}/logs"
cd "${repo}" || exit 1
export PYTHONPATH="${repo}:${repo}/pipelines"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

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
methods=(${EVAL_METHODS:-vpwem direct direct_token2})
gpus=(${EVAL_GPUS:-2 7 3})
queue_finished_marker=${QUEUE_FINISHED_MARKER:-${output_root}/QUEUE_FINISHED}

if (( ${#gpus[@]} < ${#methods[@]} )); then
  echo "EVAL_GPUS must provide at least ${#methods[@]} GPU ids" >&2
  exit 2
fi

checkpoint_for() {
  case "$1" in
    vpwem)
      printf '%s\n' "${copy_root}/fair_history_momart_multitask_emb/vpwem_bs256/seed100/models/model_400000.pt"
      ;;
    direct|direct_token2)
      printf '%s\n' "${copy_root}/fair_history_momart_multitask/$1/seed100/models/model_400000.pt"
      ;;
  esac
}

config_for() {
  case "$1" in
    vpwem)
      printf '%s\n' "/home/jrw/vpwem_outputs/fair_history_momart_multitask_emb/vpwem_bs256/seed100/.hydra/config.yaml"
      ;;
    direct|direct_token2)
      printf '%s\n' "/home/jrw/vpwem_outputs/fair_history_momart_multitask/$1/seed100/.hydra/config.yaml"
      ;;
  esac
}

run_method() {
  local method=$1
  local gpu=$2
  local failures=0
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export GIBSON_DEVICE_ID="${gpu}"
  checkpoint=$(checkpoint_for "${method}")
  train_config=$(config_for "${method}")
  for job in "${jobs[@]}"; do
    read -r env_name env_type <<< "${job}"
    name=${env_name}_${env_type}
    work=${output_root}/${method}/${name}
    result=${work}/result.npy
    log=${output_root}/logs/${method}_${name}.log
    context=${context_root}/${name}/obs_2_act_8_8layer/100/models/model_1000000.pt
    if [[ -s "${result}" ]]; then
      printf '[%s] SKIP method=%s task=%s\n' "$(date -Is)" "${method}" "${name}"
      continue
    fi
    mkdir -p "${work}"
    printf '[%s] START method=%s task=%s episodes=%s gpu=%s\n' \
      "$(date -Is)" "${method}" "${name}" "${episodes}" "${gpu}"
    context_args=()
    if [[ "${method}" == vpwem ]]; then
      context_args=(--context-checkpoint "${context}")
    fi
    "${python_bin}" pipelines/eval_fair_history_momart_multitask_emb.py \
      --checkpoint "${checkpoint}" \
      --train-config "${train_config}" \
      "${context_args[@]}" \
      --dataset-path "${data_root}/${name}.hdf5" \
      --env-name "${env_name}" --env-type "${env_type}" \
      --work-dir "${work}" --result-path "${result}" \
      --seed "${seed}" --num-envs "${num_envs}" \
      --eval-episodes "${episodes}" --num-video-episodes 1 \
      > "${log}" 2>&1
    status=$?
    if (( status == 0 )) && [[ -s "${result}" ]]; then
      printf '[%s] DONE method=%s task=%s\n' \
        "$(date -Is)" "${method}" "${name}"
    else
      printf '[%s] FAIL method=%s task=%s status=%s log=%s\n' \
        "$(date -Is)" "${method}" "${name}" "${status}" "${log}"
      failures=$((failures + 1))
    fi
  done
  printf '[%s] METHOD_FINISHED method=%s gpu=%s failures=%s\n' \
    "$(date -Is)" "${method}" "${gpu}" "${failures}"
  return "$((failures > 0))"
}

pids=()
for index in "${!methods[@]}"; do
  run_method "${methods[$index]}" "${gpus[$index]}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

printf '[%s] FINISHED failed_method_queues=%s\n' "$(date -Is)" "${failures}"
date -Is > "${queue_finished_marker}"
exit "$((failures > 0))"
