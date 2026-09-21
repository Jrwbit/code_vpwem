#!/usr/bin/env bash
set -uo pipefail

vpwem_root_dir=${VPWEM_ROOT:-/home/jrw/code_vpwem_momart_fair}
runtime_root=${VPWEM_RUNTIME_ROOT:-/home/jrw/anaconda3/envs/vpwem}
data_base=${VPWEM_DATA_ROOT:-/home/jrw/vpwem_data}
output_base=${VPWEM_OUTPUT_ROOT:-/home/jrw/vpwem_outputs}
transfer_root=${MOMART_TRANSFER_ROOT:-}
repo=${MOMART_REPO:-${vpwem_root_dir}}
python_bin=${VPWEM_PYTHON:-${runtime_root}/bin/python}
output_root=${MOMART_EVAL_OUTPUT_ROOT:-${output_base}/fair_history_momart_multitask_eval_1m}
data_root=${MOMART_RAW_DATA_ROOT:-${data_base}/momart}
context_root=${MOMART_CONTEXT_ROOT:-${output_base}/momart/dpptp}
vpwem_root=${MOMART_VPWEM_ROOT:-${output_base}/fair_history_momart_multitask_emb/vpwem_bs256/seed100}
raw_root=${MOMART_RAW_POLICY_ROOT:-${output_base}/fair_history_momart_multitask}
if [[ -n ${transfer_root} ]]; then
  context_root=${MOMART_CONTEXT_ROOT:-${transfer_root}/context_checkpoints}
  vpwem_root=${MOMART_VPWEM_ROOT:-${transfer_root}/experiments/fair_history_momart_multitask_emb/vpwem_bs256/seed100}
  raw_root=${MOMART_RAW_POLICY_ROOT:-${transfer_root}/experiments/fair_history_momart_multitask}
fi
seed=${EVAL_SEED:-20260917}
episodes=${EVAL_EPISODES:-30}
num_envs=${NUM_ENVS:-10}
poll_seconds=${POLL_SECONDS:-60}

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

checkpoint_for() {
  case "$1" in
    vpwem) printf '%s\n' "${vpwem_root}/models/model_1000000.pt" ;;
    direct|direct_token2)
      printf '%s\n' "${raw_root}/$1/seed100/models/model_1000000.pt"
      ;;
  esac
}

config_for() {
  case "$1" in
    vpwem) printf '%s\n' "${vpwem_root}/.hydra/config.yaml" ;;
    direct|direct_token2)
      printf '%s\n' "${raw_root}/$1/seed100/.hydra/config.yaml"
      ;;
  esac
}

wait_for_checkpoint() {
  local method=$1 checkpoint previous_size=-1 current_size
  checkpoint=$(checkpoint_for "${method}")
  while true; do
    if [[ -s "${checkpoint}" ]]; then
      current_size=$(stat -c %s "${checkpoint}")
      if [[ "${current_size}" == "${previous_size}" ]]; then
        printf '[%s] READY method=%s checkpoint=%s bytes=%s\n' \
          "$(date -Is)" "${method}" "${checkpoint}" "${current_size}"
        return 0
      fi
      previous_size=${current_size}
    fi
    printf '[%s] WAIT method=%s checkpoint=%s\n' \
      "$(date -Is)" "${method}" "${checkpoint}"
    sleep "${poll_seconds}"
  done
}

run_method() {
  local method=$1 gpu=$2 failures=0 checkpoint train_config
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export GIBSON_DEVICE_ID="${gpu}"
  wait_for_checkpoint "${method}"
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
      --checkpoint "${checkpoint}" --checkpoint-step 1000000 \
      --train-config "${train_config}" "${context_args[@]}" \
      --dataset-path "${data_root}/${name}.hdf5" \
      --env-name "${env_name}" --env-type "${env_type}" \
      --work-dir "${work}" --result-path "${result}" \
      --seed "${seed}" --num-envs "${num_envs}" \
      --eval-episodes "${episodes}" --num-video-episodes 1 \
      > "${log}" 2>&1
    status=$?
    if (( status == 0 )) && [[ -s "${result}" ]]; then
      printf '[%s] DONE method=%s task=%s\n' "$(date -Is)" "${method}" "${name}"
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

# Keep the evaluation strictly on GPUs 2, 3, and 7, with one method per GPU.
# Sharing an assigned GPU with an existing training process is intentional.
run_method direct 7 &
direct_pid=$!
run_method direct_token2 3 &
token_pid=$!
run_method vpwem 2 &
vpwem_pid=$!

failed_queues=0
wait "${direct_pid}" || failed_queues=$((failed_queues + 1))
wait "${token_pid}" || failed_queues=$((failed_queues + 1))
wait "${vpwem_pid}" || failed_queues=$((failed_queues + 1))
printf '[%s] FINISHED failed_method_queues=%s\n' "$(date -Is)" "${failed_queues}"
date -Is > "${output_root}/QUEUE_FINISHED"
exit "$((failed_queues > 0))"
