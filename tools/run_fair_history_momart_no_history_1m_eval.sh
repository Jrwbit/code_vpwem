#!/usr/bin/env bash
set -uo pipefail

repo=/home/jrw/code_vpwem_momart_fair
python_bin=/home/jrw/anaconda3/envs/vpwem/bin/python
output_root=/home/jrw/vpwem_outputs/fair_history_momart_multitask_eval_1m
data_root=/home/jrw/vpwem_data/momart
train_root=/home/jrw/vpwem_outputs/fair_history_momart_multitask/no_history/seed100
checkpoint=${train_root}/models/model_1000000.pt
train_config=${train_root}/.hydra/config.yaml
seed=${EVAL_SEED:-20260917}
episodes=${EVAL_EPISODES:-30}
num_envs=${NUM_ENVS:-10}
gpu=${GPU:-2}
disable_cudnn=${DISABLE_CUDNN:-0}
job_start=${JOB_START:-0}
job_end=${JOB_END:-10}

cudnn_args=()
if [[ "${disable_cudnn}" == "1" ]]; then
  cudnn_args+=(--disable-cudnn)
fi

mkdir -p "${output_root}/logs" "${output_root}/no_history"
exec 9>"${output_root}/.no_history_1m_eval_gpu${gpu}.lock"
flock -n 9 || exit 0

if [[ ! -s "${checkpoint}" || ! -s "${train_config}" ]]; then
  printf '[%s] MISSING checkpoint_or_config\n' "$(date -Is)"
  exit 1
fi

cd "${repo}" || exit 1
export CUDA_VISIBLE_DEVICES="${gpu}"
export GIBSON_DEVICE_ID="${gpu}"
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

failures=0
for job_index in "${!jobs[@]}"; do
  if (( job_index < job_start || job_index >= job_end )); then
    continue
  fi
  job=${jobs[$job_index]}
  read -r env_name env_type <<< "${job}"
  name=${env_name}_${env_type}
  work=${output_root}/no_history/${name}
  result=${work}/result.npy
  log=${output_root}/logs/no_history_${name}.log
  if [[ -s "${result}" ]]; then
    printf '[%s] SKIP task=%s\n' "$(date -Is)" "${name}"
    continue
  fi
  mkdir -p "${work}"
  printf '[%s] START task=%s index=%s episodes=%s gpu=%s\n' \
    "$(date -Is)" "${name}" "${job_index}" "${episodes}" "${gpu}"
  "${python_bin}" pipelines/eval_fair_history_momart_multitask_emb.py \
    --checkpoint "${checkpoint}" --checkpoint-step 1000000 \
    --train-config "${train_config}" \
    --dataset-path "${data_root}/${name}.hdf5" \
    --env-name "${env_name}" --env-type "${env_type}" \
    --work-dir "${work}" --result-path "${result}" \
    --seed "${seed}" --num-envs "${num_envs}" \
    --eval-episodes "${episodes}" --num-video-episodes 1 \
    "${cudnn_args[@]}" \
    >"${log}" 2>&1
  status=$?
  if (( status == 0 )) && [[ -s "${result}" ]]; then
    printf '[%s] DONE task=%s\n' "$(date -Is)" "${name}"
  else
    printf '[%s] FAIL task=%s status=%s log=%s\n' \
      "$(date -Is)" "${name}" "${status}" "${log}"
    failures=$((failures + 1))
  fi
done

printf '[%s] FINISHED failures=%s\n' "$(date -Is)" "${failures}"
date -Is >"${output_root}/NO_HISTORY_QUEUE_FINISHED_GPU${gpu}_${job_start}_${job_end}"
exit "$((failures > 0))"
