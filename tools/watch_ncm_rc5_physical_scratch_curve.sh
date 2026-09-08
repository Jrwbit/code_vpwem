#!/usr/bin/env bash
set -euo pipefail

# Evaluate the clean physical-neutral RC5 baseline without interrupting its
# 400k training run. Early checkpoints are trend diagnostics; 400k is the
# formal, 100-episode base-feasibility result.

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
RUNNER=${ROOT}/tools/run_ncm_causal_checkpoint_eval.sh
GPU=${NCM_RC5_SCRATCH_EVAL_GPU:-6}
POLL=${NCM_EVAL_POLL_SECONDS:-30}
STATUS=${OUT}/remember_color5/physical_scratch400/causal_eval_status.tsv

mkdir -p "$(dirname "${STATUS}")" "${OUT}/logs"
for spec in 5000:20 10000:20 20000:20 40000:20 100000:20 400000:100; do
  step=${spec%%:*}
  episodes=${spec##*:}
  model=${OUT}/remember_color5/physical_scratch400/train/models/model_${step}.pt
  while [[ ! -s "${model}" ]]; do sleep "${POLL}"; done
  printf '%s\t%s\tSTART\n' "$(date -Is)" "${step}" >>"${STATUS}"
  "${RUNNER}" rc5_scratch "${step}" "${episodes}" "${GPU}" 10
  printf '%s\t%s\tCOMPLETE\n' "$(date -Is)" "${step}" >>"${STATUS}"
done
date -Is >"${OUT}/remember_color5/physical_scratch400/CAUSAL_EVALS_COMPLETE"
