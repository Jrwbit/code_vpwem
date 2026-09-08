#!/usr/bin/env bash
set -uo pipefail
ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
RUNNER=${ROOT}/tools/run_ncm_causal_checkpoint_eval.sh
GPU=${NCM_CAUSAL_EVAL_GPU:-6}
POLL=${NCM_EVAL_POLL_SECONDS:-30}

mkdir -p "${OUT}/logs"
STATUS=${OUT}/causal_eval_status.tsv

checkpoint_path() {
  local run=$1 step=$2
  case "${run}" in
    shell_scratch) echo "${OUT}/shell/physical_scratch600/train/models/model_${step}.pt" ;;
    shell_transfer) echo "${OUT}/shell/physical_transfer100/train/models/model_${step}.pt" ;;
    rc5_*) echo "${OUT}/remember_color5/stage1/${run#rc5_}/train/models/model_${step}.pt" ;;
    *) return 2 ;;
  esac
}

wait_and_eval() {
  local run=$1 step=$2 episodes=$3 model
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
    step=${spec%%:*}
    episodes=${spec##*:}
    wait_and_eval "${run}" "${step}" "${episodes}" || return $?
  done
}

pids=()
watch_curve shell_scratch 10000:20 20000:20 400000:100 600000:100 & pids+=("$!")
watch_curve shell_transfer 10000:20 20000:20 100000:100 & pids+=("$!")
for method in vpwem random_cf unmatched_branch_cf matched_cf; do
  watch_curve "rc5_${method}" 5000:20 10000:20 20000:100 & pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
if (( failed == 0 )); then
  date -Is >"${OUT}/CAUSAL_EVALS_COMPLETE"
fi
exit "${failed}"
