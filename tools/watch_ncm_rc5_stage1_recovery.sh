#!/usr/bin/env bash
set -uo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
RUNNER=${ROOT}/tools/run_ncm_causal_checkpoint_eval.sh
GPU=${NCM_CAUSAL_EVAL_GPU:-6}
STATUS=${OUT}/causal_eval_status.tsv

watch_method() {
  local method=$1 spec step episodes model rc
  for spec in 10000:20 20000:100; do
    step=${spec%%:*}
    episodes=${spec##*:}
    model=${OUT}/remember_color5/stage1/${method}/train/models/model_${step}.pt
    while [[ ! -s "${model}" ]]; do sleep 30; done
    printf '%s\trc5_%s\t%s\tSTART\n' "$(date -Is)" "${method}" \
      "${step}" >>"${STATUS}"
    if "${RUNNER}" "rc5_${method}" "${step}" "${episodes}" "${GPU}" 10; then
      printf '%s\trc5_%s\t%s\tCOMPLETE\n' "$(date -Is)" "${method}" \
        "${step}" >>"${STATUS}"
    else
      rc=$?
      printf '%s\trc5_%s\t%s\tFAILED:%s\n' "$(date -Is)" "${method}" \
        "${step}" "${rc}" >>"${STATUS}"
      return "${rc}"
    fi
  done
}

pids=()
for method in vpwem random_cf unmatched_branch_cf matched_cf; do
  watch_method "${method}" & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
exit "${failed}"
