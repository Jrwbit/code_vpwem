#!/usr/bin/env bash
set -euo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
STAGE=${OUT}/remember_color5/stage1
CHECKPOINT=20000
METHODS=(vpwem random_cf unmatched_branch_cf matched_cf)

while :; do
  ready=1
  for method in "${METHODS[@]}"; do
    [[ -s "${STAGE}/${method}/eval_${CHECKPOINT}/ncm_eval.npy" && \
       -s "${STAGE}/${method}/offline_${CHECKPOINT}/offline_gap.json" ]] \
      || ready=0
  done
  (( ready == 1 )) && break
  sleep 30
done

args=()
for method in "${METHODS[@]}"; do
  cli=${method//_/-}
  args+=(
    "--${cli}" "${STAGE}/${method}/eval_${CHECKPOINT}/ncm_eval.npy"
    "--${cli}-offline" "${STAGE}/${method}/offline_${CHECKPOINT}/offline_gap.json"
  )
done
analysis_log=${OUT}/logs/rc5_stage1_${CHECKPOINT}_analysis.log
if ! "${PY}" "${ROOT}/tools/analyze_ncm_shellgame.py" \
  "${args[@]}" --output "${STAGE}/analysis_${CHECKPOINT}" \
  >"${analysis_log}" 2>&1
then
  "${PY}" - "${STAGE}/analysis_${CHECKPOINT}" "${analysis_log}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
log = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
failure = log.read_text(encoding="utf-8", errors="replace").strip()
result = {
    "analysis_valid": False,
    "go_to_necessity_gate": False,
    "failure": failure,
    "failure_class": "unpaired_counterfactual_donor_table",
}
(output / "summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(output / "RESULTS.md").write_text(
    "# RC5 initial Stage-1: NO-GO\n\n"
    "Strict paired analysis was rejected because the four arms did not use "
    "the same matched-wrong donor table. This diagnostic run must not be used "
    "as the fair main comparison.\n", encoding="utf-8")
PY
fi
touch "${STAGE}/ANALYSIS_${CHECKPOINT}_COMPLETE"
