#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 || ( $1 != "mikasa" && $1 != "robomimic" ) ]]; then
  echo "usage: $0 {mikasa|robomimic}" >&2
  exit 2
fi

benchmark=$1
case "${benchmark}" in
  mikasa)
    gpu=3
    momart_method=direct_token2
    repo=/home/jrw/code_vpwem_fair_history
    gradient_steps=600000
    ;;
  robomimic)
    gpu=7
    momart_method=direct
    repo=/home/jrw/code_vpwem_fair_history
    gradient_steps=1000000
    ;;
esac

python_bin=/home/jrw/anaconda3/envs/vpwem/bin/python
momart_eval=/home/jrw/vpwem_outputs/fair_history_momart_multitask_eval_1m
output_root=/home/jrw/vpwem_outputs/direct_token2_real_batch/${benchmark}
supervisor_log=${output_root}/supervisor_gpu${gpu}.log
batch_sizes=(256 192 128)

mkdir -p "${output_root}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTHONPATH=${repo}
export WANDB_MODE=offline PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${supervisor_log}"
}

wait_for_momart() {
  local count
  while true; do
    count=$(find "${momart_eval}/${momart_method}" -mindepth 2 \
      -maxdepth 2 -type f -name result.npy 2>/dev/null | wc -l)
    if (( count >= 10 )); then
      log "MOMART_READY method=${momart_method} results=${count}/10 gpu=${gpu}"
      return 0
    fi
    log "WAIT_MOMART method=${momart_method} results=${count}/10 gpu=${gpu}"
    sleep 60
  done
}

run_with_fallback() {
  local name=$1 config=$2 dataset=$3 extra=$4
  local bs work train_log status
  for bs in "${batch_sizes[@]}"; do
    work=${output_root}/${name}/real_bs${bs}_grad1
    train_log=${work}/train.log
    if [[ -s "${work}/models/model_${gradient_steps}.pt" ]]; then
      log "SKIP_COMPLETE benchmark=${benchmark} task=${name} real_bs=${bs} grad_accum=1"
      return 0
    fi
    mkdir -p "${work}"
    log "START benchmark=${benchmark} task=${name} real_bs=${bs} grad_accum=1 gpu=${gpu}"
    cd "${repo}" || return 1
    if [[ ${benchmark} == mikasa ]]; then
      "${python_bin}" pipelines/fair_history_mikasa.py \
        env_name="${name}" history_method=direct_token2 mode=train \
        batch_size="${bs}" gradient_steps="${gradient_steps}" \
        keep_checkpoints=20 work_dir="${work}" hydra.run.dir="${work}" \
        >"${train_log}" 2>&1
    else
      read -r env_name env_type <<< "${extra}"
      "${python_bin}" pipelines/direct_token2_robomimic_image.py \
        --config-name="${config}" env_name="${env_name}" env_type="${env_type}" \
        dataset_path="${dataset}" history_frames=16 \
        dataset_maximum_memory_length=16 dataset_memory_subsample_ratio=20 \
        batch_size="${bs}" gradient_steps="${gradient_steps}" \
        keep_checkpoints=20 save_freq=100000 log_freq=1000 \
        work_dir="${work}" hydra.run.dir="${work}" \
        >"${train_log}" 2>&1
    fi
    status=$?
    if (( status == 0 )) && [[ -s "${work}/models/model_${gradient_steps}.pt" ]]; then
      log "DONE benchmark=${benchmark} task=${name} real_bs=${bs} grad_accum=1"
      printf '%s\n' "${bs}" > "${output_root}/${name}/SELECTED_REAL_BATCH"
      return 0
    fi
    if (( status == 137 )) || \
       grep -Eqi "CUDA out of memory|OutOfMemoryError" "${train_log}"; then
      log "OOM benchmark=${benchmark} task=${name} real_bs=${bs} grad_accum=1"
      continue
    fi
    log "FAIL benchmark=${benchmark} task=${name} real_bs=${bs} status=${status} log=${train_log}"
    return 1
  done
  log "OOM_ALL benchmark=${benchmark} task=${name} tried=256,192,128 grad_accum=1"
  return 1
}

wait_for_momart
failures=0
if [[ ${benchmark} == mikasa ]]; then
  run_with_fallback ShellGameTouch-v0 base unused unused || failures=$((failures + 1))
  run_with_fallback RememberColor3-v0 base unused unused || failures=$((failures + 1))
  run_with_fallback RememberColor5-v0 base unused unused || failures=$((failures + 1))
  run_with_fallback InterceptGrabMedium-v0 base unused unused || failures=$((failures + 1))
else
  data=/home/jrw/vpwem_data/robomimic/datasets
  run_with_fallback square_ph square_abs "${data}/square/ph/image_abs.hdf5" "square ph" || failures=$((failures + 1))
  run_with_fallback square_mh square_abs "${data}/square/mh/image_abs.hdf5" "square mh" || failures=$((failures + 1))
  run_with_fallback transport_ph transport_abs "${data}/transport/ph/image_abs.hdf5" "transport ph" || failures=$((failures + 1))
  run_with_fallback transport_mh transport_abs "${data}/transport/mh/image_abs.hdf5" "transport mh" || failures=$((failures + 1))
fi

log "QUEUE_FINISHED benchmark=${benchmark} failures=${failures}"
date -Is > "${output_root}/QUEUE_FINISHED"
exit "$((failures > 0))"
