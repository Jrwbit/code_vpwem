#!/usr/bin/env bash
set -euo pipefail

# Wait for the shared physical-neutral transfer base, then run the four Shell
# counterfactual arms from exactly the same checkpoint and data order.  The
# 5k/10k points are diagnostics; 20k is the paired-adaptation endpoint.

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
DATA_ROOT=${NCM_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}
DATA=${DATA_ROOT}/ShellGameTouchNeutralHoldOracle-v0
SOURCE=${DATA_ROOT}/ShellGameTouch-v0
BASE=${NCM_SHELL_STAGE_BASE:-${OUT}/shell/physical_transfer100/train/models/model_100000.pt}
STAGE=${NCM_SHELL_STAGE_OUTPUT:-${OUT}/shell/stage1_transfer100}
STAGE_TAG=${NCM_SHELL_STAGE_TAG:-$(basename "${STAGE}")}
MANIFEST_DIR=${STAGE}/pair_manifest
MANIFEST=${MANIFEST_DIR}/ncm_shellgame_pairs.npz

export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "${STAGE}" "${OUT}/logs"
while [[ ! -s "${BASE}" ]]; do
  sleep 30
done

if [[ ! -s "${MANIFEST}" ]]; then
  mkdir -p "${MANIFEST_DIR}"
  CUDA_VISIBLE_DEVICES=3 "${PY}" "${ROOT}/tools/build_ncm_shellgame_manifest.py" \
    --dataset "${DATA}" \
    --normalizer-dataset "${SOURCE}" \
    --checkpoint "${BASE}" \
    --branch-labels "${DATA}/branch_labels.npy" \
    --output "${MANIFEST_DIR}" \
    --device cuda:0 \
    --batch-size 16 \
    --pairing-seed 100 \
    --decision-step 6 \
    --hold-step 5 \
    --hold-steps 0 \
    >"${OUT}/logs/shell_${STAGE_TAG}_manifest.log" 2>&1
fi

"${PY}" - "${MANIFEST_DIR}/pair_quality.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
failures = []
if report["pair_coverage"] < 0.30:
    failures.append("pair coverage below 30%")
if report["current_branch_lobo_1nn_probe_acc"] > 0.45:
    failures.append("branch decodable from complete current input")
if report.get("target_action_branch_lobo_1nn_probe_acc", 0.0) < 0.80:
    failures.append("decision action not branch-dependent")
if report["matched_cosine"]["mean"] <= report["unmatched_cosine"]["mean"]:
    failures.append("matched donors no closer than unmatched controls")
if failures:
    raise SystemExit("PAIR_QUALITY_GATE_FAILED: " + "; ".join(failures))
print("PAIR_QUALITY_GATE_PASSED")
PY

methods=(vpwem random_cf unmatched_branch_cf matched_cf)
read -r -a gpus <<<"${NCM_SHELL_STAGE_GPUS:-2 3 6 5}"
if (( ${#gpus[@]} != ${#methods[@]} )); then
  echo "NCM_SHELL_STAGE_GPUS must contain four GPU ids" >&2
  exit 2
fi
modes=(none random unmatched_branch matched)
coefficients=(0.0 0.5 0.5 0.5)

train_one() {
  local method=$1 gpu=$2 mode=$3 coefficient=$4
  local work=${STAGE}/${method}/train
  if [[ -s "${work}/models/model_20000.pt" ]]; then
    return 0
  fi
  mkdir -p "${work}"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" \
    "${ROOT}/pipelines/longmemdpptp_mikasa_image.py" \
    --config-name=shell_game_touch_ncm \
    dataset_path="${DATA}" normalizer_dataset_path="${SOURCE}" \
    ncm_pair_manifest_path="${MANIFEST}" +ncm_pair_batch_fraction=0.1 \
    +dataset_preserve_episode_anchor=true \
    init_model_path="${BASE}" work_dir="${work}" \
    save_path="${OUT}" exp_name="shell/${STAGE_TAG}/${method}" \
    counterfactual_negative_mode="${mode}" \
    counterfactual_memory_coeff="${coefficient}" \
    counterfactual_memory_margin=0.1 \
    counterfactual_memory_probability=1.0 \
    gradient_steps=20000 save_freq=5000 keep_checkpoints=4 \
    log_freq=500 batch_size=16 lr=0.00001 seed=100 \
    device=cuda:0 wandb_mode=offline \
    >"${OUT}/logs/shell_${STAGE_TAG}_${method}.log" 2>&1
}

pids=()
for i in "${!methods[@]}"; do
  train_one "${methods[$i]}" "${gpus[$i]}" \
    "${modes[$i]}" "${coefficients[$i]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || exit 1

eval_one() {
  local method=$1 gpu=$2 checkpoint=$3 episodes=$4
  local work=${STAGE}/${method}/eval_${checkpoint}
  local result=${work}/ncm_eval.npy
  if [[ -s "${result}" && -s "${work}/ncm_eval.json" ]]; then
    return 0
  fi
  mkdir -p "${work}"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" \
    "${ROOT}/pipelines/ncm_shellgame_eval.py" \
    --config-name=shell_game_touch_ncm \
    dataset_path="${DATA}" normalizer_dataset_path="${SOURCE}" \
    model_path="${STAGE}/${method}/train/models/model_${checkpoint}.pt" \
    pairing_checkpoint_path="${BASE}" \
    work_dir="${work}" result_path="${result}" \
    eval_seed_start=20260807 diffusion_seed=20260807 \
    eval_episodes="${episodes}" num_envs=10 save_video=false \
    device=cuda:0 wandb_mode=offline \
    >"${OUT}/logs/shell_${STAGE_TAG}_${method}_${checkpoint}_eval.log" 2>&1
}

pids=()
for i in "${!methods[@]}"; do
  (
    eval_one "${methods[$i]}" "${gpus[$i]}" 5000 20
    eval_one "${methods[$i]}" "${gpus[$i]}" 10000 20
    eval_one "${methods[$i]}" "${gpus[$i]}" 20000 100
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || exit 1

"${ROOT}/tools/run_ncm_stage_analysis.sh" shell "${STAGE}" 20000
touch "${STAGE}/COMPLETE"
