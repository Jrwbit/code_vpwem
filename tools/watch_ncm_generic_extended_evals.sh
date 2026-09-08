#!/usr/bin/env bash
set -uo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
RUNNER=${ROOT}/tools/run_ncm_generic_checkpoint_eval.sh
GPU=${NCM_GENERIC_EVAL_GPU:-4}
POLL=${NCM_EVAL_POLL_SECONDS:-30}
STATUS=${OUT}/generic_extended_eval_status.tsv
mkdir -p "${OUT}/logs"

checkpoint_path() {
  case "$1" in
    rc3) echo "${OUT}/remember_color3/parity600/train/models/model_$2.pt" ;;
    rc5_base) echo "${OUT}/remember_color5/base400/train/models/model_$2.pt" ;;
    intercept) echo "${OUT}/intercept_grab/base400/train/models/model_$2.pt" ;;
    *) return 2 ;;
  esac
}

wait_and_eval() {
  local run=$1 step=$2 episodes=$3 model attempt rc
  model=$(checkpoint_path "${run}" "${step}") || return $?
  while [[ ! -s "${model}" ]]; do sleep "${POLL}"; done
  for attempt in 1 2; do
    printf '%s\t%s\t%s\t%s\tSTART\n' \
      "$(date -Is)" "${run}" "${step}" "${attempt}" >>"${STATUS}"
    if "${RUNNER}" "${run}" "${step}" "${episodes}" "${GPU}" 10; then
      printf '%s\t%s\t%s\t%s\tCOMPLETE\n' \
        "$(date -Is)" "${run}" "${step}" "${attempt}" >>"${STATUS}"
      return 0
    fi
    rc=$?
    printf '%s\t%s\t%s\t%s\tFAILED:%s\n' \
      "$(date -Is)" "${run}" "${step}" "${attempt}" "${rc}" >>"${STATUS}"
    sleep 60
  done
  return 1
}

watch_curve() {
  local run=$1 spec failed=0
  shift
  for spec in "$@"; do
    wait_and_eval "${run}" "${spec%%:*}" "${spec##*:}" || failed=1
  done
  return "${failed}"
}

pids=()
watch_curve rc3 40000:30 100000:100 400000:100 600000:100 & pids+=("$!")
watch_curve rc5_base 40000:30 100000:100 400000:100 & pids+=("$!")
watch_curve intercept 20000:30 40000:30 100000:100 400000:100 & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) && date -Is >"${OUT}/GENERIC_EXTENDED_EVALS_COMPLETE"
exit "${failed}"
