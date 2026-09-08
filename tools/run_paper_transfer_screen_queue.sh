#!/usr/bin/env bash
set -uo pipefail

# One persistent worker for the VPWEM paper-transfer screen. Run one worker on
# each admitted GPU; methods are processed sequentially to avoid GPU OOM.

ROOT=${VPWEM_TRANSFER_ROOT:-/home/jrw/code_vpwem}
PY=${VPWEM_TRANSFER_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${VPWEM_TRANSFER_OUTPUT:-/home/jrw/vpwem_outputs/paper_transfer_screen_20260902}
DATA_ROOT=${VPWEM_TRANSFER_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}
DATA=${VPWEM_TRANSFER_DATA:-${DATA_ROOT}/RememberColor5NeutralHoldOracle-v0}
SOURCE=${VPWEM_TRANSFER_NORMALIZER_DATA:-${DATA_ROOT}/RememberColor5-v0}
BASE=${VPWEM_TRANSFER_BASE:-/home/jrw/vpwem_outputs/per_vpwem_screen_seed100/fifo_controls/mikasa/RememberColor5-v0/100/models/model_400000.pt}
GPU=${1:?usage: $0 GPU METHOD...}
shift
METHODS=("$@")
WORKER_TAG=${VPWEM_TRANSFER_WORKER_TAG:-gpu${GPU}}

if (( ${#METHODS[@]} == 0 )); then
  echo "at least one method is required" >&2
  exit 2
fi

export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

LOG_DIR=${OUT}/logs
STATUS_DIR=${OUT}/status
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
QUEUE_LOG=${LOG_DIR}/${WORKER_TAG}_queue.log

log() {
  printf '[%s] worker=%s gpu=%s %s\n' \
    "$(date -Is)" "${WORKER_TAG}" "${GPU}" "$*" | tee -a "${QUEUE_LOG}"
}

method_config() {
  case "$1" in
    vpwem) echo remember_color_5_ncm ;;
    gmp) echo remember_color_5_gmp_screen ;;
    aem) echo remember_color_5_aem_screen ;;
    dssp) echo remember_color_5_dssp_screen ;;
    vq) echo remember_color_5_vq_screen ;;
    chameleon) echo remember_color_5_chameleon_screen ;;
    cycle) echo remember_color_5_cycle_screen ;;
    *) return 2 ;;
  esac
}

common_args() {
  printf '%s\n' \
    "dataset_path=${DATA}" \
    "normalizer_dataset_path=${SOURCE}" \
    "save_path=${OUT}" \
    "mem_compress_length=2" \
    "qformer_num_layers=2" \
    "qformer_cache_compress=fifo" \
    "qformer_cache_max_length=8" \
    "diffusion_num_layers=8" \
    "batch_size=16" \
    "lr=0.00001" \
    "device=cuda:0" \
    "wandb_mode=offline" \
    "num_workers=4" \
    "+dataset_preserve_episode_anchor=true"
}

run_train() {
  local method=$1 config=$2 train_dir=$3 checkpoint=$4
  if [[ -s "${checkpoint}" ]]; then
    log "SKIP train method=${method} checkpoint_exists=${checkpoint}"
    return 0
  fi

  mkdir -p "${train_dir}"
  mapfile -t common < <(common_args)
  local extra=()
  if [[ "${method}" == vpwem ]]; then
    extra+=("+memory_read_mode=context")
  fi

  log "START train method=${method} config=${config}"
  CUDA_VISIBLE_DEVICES=${GPU} "${PY}" \
    "${ROOT}/pipelines/longmemdpptp_mikasa_image.py" \
    --config-name="${config}" \
    "${common[@]}" "${extra[@]}" \
    "init_model_path=${BASE}" \
    "work_dir=${train_dir}" \
    gradient_steps=5000 save_freq=5000 keep_checkpoints=2 \
    log_freq=100 seed=100 \
    >"${LOG_DIR}/${method}_train.log" 2>&1
  local rc=$?
  if (( rc == 0 )) && [[ -s "${checkpoint}" ]]; then
    log "DONE train method=${method}"
    return 0
  fi
  log "FAIL train method=${method} rc=${rc}"
  return 1
}

run_eval() {
  local method=$1 config=$2 checkpoint=$3 eval_dir=$4 result=$5
  if [[ -s "${result}" ]]; then
    log "SKIP eval method=${method} result_exists=${result}"
    return 0
  fi
  if [[ ! -s "${checkpoint}" ]]; then
    log "FAIL eval method=${method} reason=checkpoint_missing"
    return 1
  fi

  mkdir -p "${eval_dir}"
  mapfile -t common < <(common_args)
  local extra=()
  if [[ "${method}" == vpwem ]]; then
    extra+=("+memory_read_mode=context")
  fi

  log "START eval method=${method} episodes=30"
  CUDA_VISIBLE_DEVICES=${GPU} "${PY}" \
    "${ROOT}/pipelines/longmemdpptp_mikasa_image_eval.py" \
    --config-name="${config}" \
    "${common[@]}" "${extra[@]}" \
    "model_path=${checkpoint}" \
    "work_dir=${eval_dir}" \
    "result_path=${result}" \
    +ckpt=5000 eval_episodes=30 num_envs=10 save_video=false \
    inference_action_steps=1 seed=20260902 \
    >"${LOG_DIR}/${method}_eval.log" 2>&1
  local rc=$?
  if (( rc == 0 )) && [[ -s "${result}" ]]; then
    log "DONE eval method=${method}"
    return 0
  fi
  log "FAIL eval method=${method} rc=${rc}"
  return 1
}

test -x "${PY}" || { log "FATAL python_not_executable=${PY}"; exit 2; }
test -s "${BASE}" || { log "FATAL base_missing=${BASE}"; exit 2; }
test -d "${DATA}" || { log "FATAL dataset_missing=${DATA}"; exit 2; }
test -d "${SOURCE}" || { log "FATAL normalizer_dataset_missing=${SOURCE}"; exit 2; }

log "QUEUE_BEGIN methods=${METHODS[*]}"
failures=0
for method in "${METHODS[@]}"; do
  config=$(method_config "${method}") || {
    log "FAIL method=${method} reason=unknown_method"
    failures=$((failures + 1))
    continue
  }
  train_dir=${OUT}/remember_color5/${method}/train
  checkpoint=${train_dir}/models/model_5000.pt
  eval_dir=${OUT}/remember_color5/${method}/eval_5000
  result=${eval_dir}/result_5000.npy

  if run_train "${method}" "${config}" "${train_dir}" "${checkpoint}"; then
    run_eval "${method}" "${config}" "${checkpoint}" "${eval_dir}" "${result}" || \
      failures=$((failures + 1))
  else
    failures=$((failures + 1))
  fi
done

printf '%s\n' "${failures}" >"${STATUS_DIR}/${WORKER_TAG}_failures.txt"
date -Is >"${STATUS_DIR}/${WORKER_TAG}_finished.txt"
if (( failures == 0 )); then
  log "QUEUE_COMPLETE"
  exit 0
fi
log "QUEUE_FINISHED failures=${failures}; rerun is resumable"
exit 1
