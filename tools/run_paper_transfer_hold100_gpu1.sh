#!/usr/bin/env bash
set -uo pipefail

ROOT=${VPWEM_HOLD_ROOT:-/home/jrw/code_vpwem}
PY=${VPWEM_HOLD_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${VPWEM_HOLD_OUTPUT:-/home/jrw/vpwem_outputs/paper_transfer_screen_20260902}
DATA_ROOT=${VPWEM_HOLD_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}
DATA=${DATA_ROOT}/RememberColor5NeutralHoldOracle-v0
SOURCE=${DATA_ROOT}/RememberColor5-v0
GPU=${VPWEM_HOLD_GPU:-1}
SEED=${VPWEM_HOLD_EVAL_SEED:-20260823}

export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

LOG_DIR=${OUT}/logs/hold100
STATUS_DIR=${OUT}/status/hold100
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
SUPERVISOR_LOG=${LOG_DIR}/supervisor_gpu${GPU}.log

log() {
  printf '[%s] gpu=%s %s\n' "$(date -Is)" "${GPU}" "$*" | \
    tee -a "${SUPERVISOR_LOG}"
}

method_config() {
  case "$1" in
    vpwem) echo remember_color_5_ncm ;;
    gmp) echo remember_color_5_gmp_screen ;;
    aem) echo remember_color_5_aem_screen ;;
    dssp) echo remember_color_5_dssp_screen ;;
    vq) echo remember_color_5_vq_screen ;;
    cycle) echo remember_color_5_cycle_screen ;;
    chameleon) echo remember_color_5_chameleon_screen ;;
    *) return 2 ;;
  esac
}

run_one() {
  local method=$1 config checkpoint work result extra=()
  config=$(method_config "${method}") || return 2
  checkpoint=${OUT}/remember_color5/${method}/train/models/model_5000.pt
  work=${OUT}/remember_color5/${method}/hold_eval_100
  result=${work}/ncm_correct_100.npy
  if [[ -s "${result}" && -s "${result%.npy}.json" ]]; then
    log "SKIP method=${method} result_exists=${result}"
    return 0
  fi
  if [[ ! -s "${checkpoint}" ]]; then
    log "FAIL method=${method} checkpoint_missing=${checkpoint}"
    return 1
  fi
  if [[ "${method}" == vpwem ]]; then
    extra+=("+memory_read_mode=context")
  fi
  mkdir -p "${work}"
  log "START method=${method} episodes=100 seed=${SEED}"
  CUDA_VISIBLE_DEVICES=${GPU} "${PY}" \
    "${ROOT}/pipelines/ncm_shellgame_eval.py" \
    --config-name="${config}" \
    dataset_path="${DATA}" normalizer_dataset_path="${SOURCE}" \
    model_path="${checkpoint}" work_dir="${work}" result_path="${result}" \
    mem_compress_length=2 qformer_num_layers=2 \
    qformer_cache_compress=fifo qformer_cache_max_length=8 \
    diffusion_num_layers=8 inference_action_steps=1 \
    eval_episodes=100 num_envs=10 eval_seed_start="${SEED}" \
    diffusion_seed="${SEED}" seed=100 device=cuda:0 \
    save_video=false wandb_mode=offline \
    +ncm_memory_interventions='[correct]' "${extra[@]}" \
    >"${LOG_DIR}/${method}.log" 2>&1
  local rc=$?
  if (( rc == 0 )) && [[ -s "${result}" && -s "${result%.npy}.json" ]]; then
    log "DONE method=${method}"
    return 0
  fi
  log "FAIL method=${method} rc=${rc}"
  return 1
}

run_wave() {
  local methods=("$@") pids=() method pid failed=0
  log "WAVE_BEGIN methods=${methods[*]}"
  for method in "${methods[@]}"; do
    run_one "${method}" &
    pid=$!
    pids+=("${pid}")
    printf '%s\n' "${pid}" >"${STATUS_DIR}/${method}.pid"
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=$((failed + 1))
  done
  log "WAVE_END methods=${methods[*]} failures=${failed}"
  return "${failed}"
}

test -x "${PY}" || { log "FATAL python_not_executable=${PY}"; exit 2; }
test -f "${DATA}/NCM_PHYSICAL_HOLD.json" || {
  log "FATAL physical_hold_marker_missing=${DATA}/NCM_PHYSICAL_HOLD.json"
  exit 2
}

log "QUEUE_BEGIN protocol=physical_hold_10 correct_only episodes=100"
failures=0
run_wave vpwem gmp aem dssp || failures=$((failures + $?))
run_wave vq cycle chameleon || failures=$((failures + $?))
printf '%s\n' "${failures}" >"${STATUS_DIR}/failures.txt"
date -Is >"${STATUS_DIR}/finished.txt"
if (( failures == 0 )); then
  log "QUEUE_COMPLETE"
  exit 0
fi
log "QUEUE_FINISHED failures=${failures}; rerun is resumable"
exit 1
