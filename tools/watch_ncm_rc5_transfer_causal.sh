#!/usr/bin/env bash
set -euo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
RUNNER=${ROOT}/tools/run_ncm_causal_checkpoint_eval.sh
GPU=${NCM_CAUSAL_EVAL_GPU:-6}
STATUS=${OUT}/causal_eval_status.tsv

for spec in 5000:20 10000:20 20000:20 100000:100; do
  step=${spec%%:*}
  episodes=${spec##*:}
  model=${OUT}/remember_color5/physical_transfer100/train/models/model_${step}.pt
  while [[ ! -s "${model}" ]]; do sleep 30; done
  printf '%s\trc5_transfer\t%s\tSTART\n' "$(date -Is)" "${step}" \
    >>"${STATUS}"
  if "${RUNNER}" rc5_transfer "${step}" "${episodes}" "${GPU}" 10; then
    printf '%s\trc5_transfer\t%s\tCOMPLETE\n' "$(date -Is)" "${step}" \
      >>"${STATUS}"
  else
    rc=$?
    printf '%s\trc5_transfer\t%s\tFAILED:%s\n' "$(date -Is)" \
      "${step}" "${rc}" >>"${STATUS}"
    exit "${rc}"
  fi
done
