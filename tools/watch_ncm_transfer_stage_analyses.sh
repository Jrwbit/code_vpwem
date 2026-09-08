#!/usr/bin/env bash
set -euo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}

analyze_when_ready() {
  local task=$1 stage=$2
  while [[ ! -f "${stage}/COMPLETE" ]]; do sleep 30; done
  "${ROOT}/tools/run_ncm_stage_analysis.sh" "${task}" "${stage}" 20000
}

analyze_when_ready shell "${OUT}/shell/stage1_transfer100" & p1=$!
analyze_when_ready rc5 "${OUT}/remember_color5/stage1_transfer100" & p2=$!
failed=0
wait "${p1}" || failed=1
wait "${p2}" || failed=1
exit "${failed}"
