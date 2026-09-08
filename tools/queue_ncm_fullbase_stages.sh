#!/usr/bin/env bash
set -euo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}

run_if_base_is_memory_capable() {
  local task=$1 base=$2 base_eval=$3 final_stage=$4 runner=$5
  while [[ ! -s "${base}" || ! -s "${base_eval}" ]]; do sleep 30; done
  if ! "${PY}" - "${task}" "${base_eval}" <<'PY'
import json
import sys
task = sys.argv[1]
with open(sys.argv[2], encoding="utf-8") as stream:
    result = json.load(stream)
correct = result["modes"]["correct"]
wrong = result["modes"]["matched_wrong"]
chance = {"shell": 1 / 3, "rc5": 1 / 5}[task]
min_success = {"shell": 0.40, "rc5": 0.20}[task]
checks = {
    "branch_above_chance_by_10pp": correct["branch_acc"] >= chance + 0.10,
    "correct_success_floor": correct["mean_success"] >= min_success,
    "matched_wrong_success_gap": (
        correct["mean_success"] - wrong["mean_success"] >= 0.15),
    "wrong_history_attracts_choice": result["wrong_attraction_uplift"] > 0,
    "wrong_history_attracts_direction": (
        result["first_direction_wrong_uplift"] > 0),
}
print(json.dumps(checks, sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 1)
PY
  then
    mkdir -p "${final_stage}"
    date -Is >"${final_stage}/BASE_MEMORY_NO_GO"
    return 0
  fi

  case "${task}" in
    shell)
      NCM_SHELL_STAGE_BASE="${base}" \
      NCM_SHELL_STAGE_OUTPUT="${final_stage}" \
      NCM_SHELL_STAGE_TAG=stage1_fullbase600 \
        "${runner}"
      ;;
    rc5)
      NCM_RC5_STAGE_BASE="${base}" \
      NCM_RC5_STAGE_OUTPUT="${final_stage}" \
      NCM_RC5_STAGE_TAG=stage1_fullbase400 \
        "${runner}"
      ;;
  esac
}

run_if_base_is_memory_capable shell \
  "${OUT}/shell/physical_scratch600/train/models/model_600000.pt" \
  "${OUT}/shell/physical_scratch600/eval_600000/ncm_eval.json" \
  "${OUT}/shell/stage1_fullbase600" \
  "${ROOT}/tools/queue_ncm_shell_stage1_from_transfer100.sh" & p1=$!

run_if_base_is_memory_capable rc5 \
  "${OUT}/remember_color5/physical_scratch400/train/models/model_400000.pt" \
  "${OUT}/remember_color5/physical_scratch400/eval_400000/ncm_eval.json" \
  "${OUT}/remember_color5/stage1_fullbase400" \
  "${ROOT}/tools/queue_ncm_rc5_stage1_from_transfer100.sh" & p2=$!

failed=0
wait "${p1}" || failed=1
wait "${p2}" || failed=1
exit "${failed}"
