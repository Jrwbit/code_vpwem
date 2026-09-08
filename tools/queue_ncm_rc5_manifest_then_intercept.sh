#!/usr/bin/env bash
set -uo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
COLLECTION_PID=${NCM_COLLECTION_PID:-998091}
DATA=/home/jrw/vpwem_data/mikasa/RememberColor5NeutralHoldOracle-v0
SOURCE=/home/jrw/vpwem_data/mikasa/RememberColor5-v0
CHECKPOINT=/home/jrw/vpwem_outputs/per_vpwem_screen_seed100/fifo_controls/mikasa/RememberColor5-v0/100/models/model_400000.pt
MANIFEST_DIR=${OUT}/remember_color5/pair_manifest

export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUNBUFFERED=1
mkdir -p "${MANIFEST_DIR}" "${OUT}/logs"

while kill -0 "${COLLECTION_PID}" 2>/dev/null; do
  sleep 30
done

build_status=0
CUDA_VISIBLE_DEVICES=6 "${PY}" "${ROOT}/tools/build_ncm_shellgame_manifest.py" \
  --dataset "${DATA}" \
  --normalizer-dataset "${SOURCE}" \
  --checkpoint "${CHECKPOINT}" \
  --branch-labels "${DATA}/branch_labels.npy" \
  --output "${MANIFEST_DIR}" \
  --device cuda:0 \
  --batch-size 64 \
  --encoder-batch-size 64 \
  --pairing-seed 100 \
  --num-branches 5 \
  --decision-step 10 \
  --hold-step 9 \
  --hold-steps 0 \
  >"${OUT}/logs/remember_color5_pair_manifest.log" 2>&1 || build_status=$?

if (( build_status == 0 )); then
  "${PY}" - "${MANIFEST_DIR}/pair_quality.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
failures = []
if d["physical_success_rate"] < 0.90:
    failures.append("physical success below 90%")
if d["pair_coverage"] < 0.30:
    failures.append("pair coverage below 30%")
if d["current_branch_lobo_1nn_probe_acc"] > 0.35:
    failures.append("branch decodable from complete current input")
if d["target_action_branch_lobo_1nn_probe_acc"] < 0.80:
    failures.append("decision action is not branch-dependent")
if d["matched_cosine"]["mean"] <= d["unmatched_cosine"]["mean"]:
    failures.append("matched current is not closer than unmatched control")
gate = {
    "passed": not failures,
    "failures": failures,
    "pair_quality": d,
}
(p.parent / "PAIR_GATE.json").write_text(
    json.dumps(gate, indent=2, sort_keys=True) + "\n")
if failures:
    raise SystemExit("PAIR_QUALITY_GATE_FAILED: " + "; ".join(failures))
print("PAIR_QUALITY_GATE_PASSED")
PY
  gate_status=$?
else
  gate_status=${build_status}
fi

printf '%s build_status=%s gate_status=%s\n' \
  "$(date -Is)" "${build_status}" "${gate_status}" \
  >"${OUT}/remember_color5/pair_manifest_status.txt"

stage1_status=0
if (( gate_status == 0 )); then
  "${ROOT}/tools/run_ncm_remember_color5_stage1.sh" || stage1_status=$?
else
  stage1_status=125
fi
printf '%s stage1_status=%s\n' "$(date -Is)" "${stage1_status}" \
  >"${OUT}/remember_color5/stage1_status.txt"

# InterceptGrab is an independent negative-control queue and must still run if
# RC5 pair feasibility is rejected.
exec "${ROOT}/tools/run_ncm_crossbench_long_bases.sh" intercept
