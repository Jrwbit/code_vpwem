#!/usr/bin/env bash
set -uo pipefail
ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
RUNNER=${ROOT}/tools/run_ncm_generic_checkpoint_eval.sh
GPU=${NCM_GENERIC_EVAL_GPU:-4}
POLL=${NCM_EVAL_POLL_SECONDS:-30}
STATUS=${OUT}/generic_eval_status.tsv
mkdir -p "${OUT}/logs"

checkpoint_path() {
  local run=$1 step=$2
  case "${run}" in
    rc3) echo "${OUT}/remember_color3/parity600/train/models/model_${step}.pt" ;;
    rc5_base) echo "${OUT}/remember_color5/base400/train/models/model_${step}.pt" ;;
    intercept) echo "${OUT}/intercept_grab/base400/train/models/model_${step}.pt" ;;
    *) return 2 ;;
  esac
}

wait_and_eval() {
  local run=$1 step=$2 episodes=$3 model rc
  model=$(checkpoint_path "${run}" "${step}") || return $?
  while ! test -s "${model}"; do sleep "${POLL}"; done
  printf '%s\t%s\t%s\tSTART\n' "$(date -Is)" "${run}" "${step}" >>"${STATUS}"
  if "${RUNNER}" "${run}" "${step}" "${episodes}" "${GPU}" 10; then
    printf '%s\t%s\t%s\tCOMPLETE\n' "$(date -Is)" "${run}" "${step}" >>"${STATUS}"
  else
    rc=$?
    printf '%s\t%s\t%s\tFAILED:%s\n' "$(date -Is)" "${run}" "${step}" "${rc}" >>"${STATUS}"
    return "${rc}"
  fi
}

watch_curve() {
  local run=$1
  shift
  local spec step episodes
  for spec in "$@"; do
    step=${spec%%:*}; episodes=${spec##*:}
    wait_and_eval "${run}" "${step}" "${episodes}" || return $?
  done
}

pids=()
watch_curve rc3 5000:30 10000:30 20000:30 400000:100 600000:100 & pids+=("$!")
watch_curve rc5_base 5000:30 10000:30 20000:30 400000:100 & pids+=("$!")
watch_curve intercept 5000:30 10000:30 20000:30 400000:100 & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
if (( failed == 0 )); then date -Is >"${OUT}/GENERIC_EVALS_COMPLETE"; fi
exit "${failed}"
