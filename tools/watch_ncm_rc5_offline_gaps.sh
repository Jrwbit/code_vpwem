#!/usr/bin/env bash
set -uo pipefail
ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
RUNNER=${ROOT}/tools/run_ncm_rc5_offline_gap.sh
GPU=${NCM_OFFLINE_GPU:-5}
POLL=${NCM_EVAL_POLL_SECONDS:-30}
STATUS=${OUT}/rc5_offline_gap_status.tsv

watch_method() {
  local method=$1 step model rc
  for step in 5000 10000 20000; do
    model=${OUT}/remember_color5/stage1/${method}/train/models/model_${step}.pt
    while ! test -s "${model}"; do sleep "${POLL}"; done
    printf '%s\t%s\t%s\tSTART\n' "$(date -Is)" "${method}" "${step}" >>"${STATUS}"
    if "${RUNNER}" "${method}" "${step}" "${GPU}"; then
      printf '%s\t%s\t%s\tCOMPLETE\n' "$(date -Is)" "${method}" "${step}" >>"${STATUS}"
    else
      rc=$?
      printf '%s\t%s\t%s\tFAILED:%s\n' "$(date -Is)" "${method}" "${step}" "${rc}" >>"${STATUS}"
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
if (( failed == 0 )); then date -Is >"${OUT}/RC5_OFFLINE_GAPS_COMPLETE"; fi
exit "${failed}"
