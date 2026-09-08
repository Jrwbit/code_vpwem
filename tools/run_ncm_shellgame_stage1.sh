#!/usr/bin/env bash
set -euo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
DATA=${NCM_DATASET:-/home/jrw/vpwem_data/mikasa/ShellGameTouchNeutralHold-v0}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_vpwem_shellgame_stage1_seed100}
BASE=${NCM_BASE_CHECKPOINT:-/home/jrw/vpwem_outputs/shell_game_touch/ours/mikasa/ShellGameTouch-v0/obs_2_act_8/8layer_5_qformer_2_sin_pos_2_fifo_8_0.0001/100/models/model_600000.pt}
SOURCE_DATA=${NCM_SOURCE_DATASET:-/home/jrw/vpwem_data/mikasa/ShellGameTouch-v0}
ORACLE=${NCM_ORACLE_CHECKPOINT:-}
MANIFEST_DIR=${OUT}/pair_manifest
MANIFEST=${MANIFEST_DIR}/ncm_shellgame_pairs.npz
COLLECTION_GPU=${NCM_COLLECTION_GPU:-2}
PAIRING_GPU=${NCM_PAIRING_GPU:-2}

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}

mkdir -p "${OUT}/logs" "${MANIFEST_DIR}"

if [[ ! -s "${BASE}" ]]; then
  echo "Missing VPWEM base checkpoint: ${BASE}" >&2
  exit 2
fi

if [[ ! -s "${SOURCE_DATA}/train_data_999.npz" ]]; then
  echo "Missing source demonstrations: ${SOURCE_DATA}" >&2
  exit 2
fi

if [[ ! -s "${DATA}/NCM_PHYSICAL_HOLD.json" ]] || ! grep -q '"complete": true' "${DATA}/NCM_PHYSICAL_HOLD.json"; then
  mkdir -p "${DATA}"
  collect_args=(
    --source-dataset "${SOURCE_DATA}"
    --output "${DATA}"
    --episodes 1000
    --batch-size 250
    --neutral-through-step 5
  )
  if [[ -n "${ORACLE}" ]]; then
    if [[ ! -s "${ORACLE}" ]]; then
      echo "Missing MIKASA oracle checkpoint: ${ORACLE}" >&2
      exit 2
    fi
    collect_args+=(--oracle-checkpoint "${ORACLE}")
  fi
  CUDA_VISIBLE_DEVICES=${COLLECTION_GPU} "${PY}" "${ROOT}/tools/collect_shellgame_neutral_hold.py" \
    "${collect_args[@]}" \
    >"${OUT}/logs/collect_physical_hold.log" 2>&1
fi

if [[ ! -s "${MANIFEST}" ]]; then
  CUDA_VISIBLE_DEVICES=${PAIRING_GPU} "${PY}" "${ROOT}/tools/build_ncm_shellgame_manifest.py" \
    --dataset "${DATA}" \
    --normalizer-dataset "${SOURCE_DATA}" \
    --checkpoint "${BASE}" \
    --branch-labels "${DATA}/branch_labels.npy" \
    --output "${MANIFEST_DIR}" \
    --device cuda:0 \
    --batch-size 16 \
    --pairing-seed 100 \
    --decision-step 6 \
    --hold-step 5 \
    --hold-steps 0 \
    >"${OUT}/logs/build_pair_manifest.log" 2>&1
fi

"${PY}" - "${MANIFEST_DIR}/pair_quality.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
failures = []
if report["pair_coverage"] < 0.30:
    failures.append("pair coverage below 30%")
if report["current_branch_lobo_1nn_probe_acc"] > 0.45:
    failures.append("branch is decodable from complete current input")
if report.get("target_action_branch_lobo_1nn_probe_acc", 0.0) < 0.80:
    failures.append("expert decision action is not branch-dependent")
if report["matched_cosine"]["mean"] <= report["unmatched_cosine"]["mean"]:
    failures.append("matched pairs are not closer than unmatched control")
if failures:
    raise SystemExit("PAIR_QUALITY_GATE_FAILED: " + "; ".join(failures))
print("PAIR_QUALITY_GATE_PASSED", json.dumps(report, sort_keys=True))
PY

methods=(vpwem random_cf unmatched_branch_cf matched_cf)
gpus=(2 3 4 5)
modes=(none random unmatched_branch matched)
coefficients=(0.0 0.5 0.5 0.5)

train_one() {
  local method=$1 gpu=$2 mode=$3 coefficient=$4
  local work=${OUT}/${method}/train
  local checkpoint=${work}/models/model_10000.pt
  if [[ -s "${checkpoint}" ]]; then
    return 0
  fi
  mkdir -p "${work}"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${ROOT}/pipelines/longmemdpptp_mikasa_image.py" \
    --config-name=shell_game_touch_ncm \
    dataset_path="${DATA}" \
    normalizer_dataset_path="${SOURCE_DATA}" \
    ncm_pair_manifest_path="${MANIFEST}" \
    init_model_path="${BASE}" \
    work_dir="${work}" \
    save_path="${OUT}" \
    exp_name="ncm_shellgame/${method}" \
    counterfactual_negative_mode="${mode}" \
    counterfactual_memory_coeff="${coefficient}" \
    counterfactual_memory_margin=0.1 \
    counterfactual_memory_probability=1.0 \
    shellgame_hold_steps=0 \
    gradient_steps=10000 \
    save_freq=10000 \
    log_freq=500 \
    batch_size=16 \
    seed=100 \
    device=cuda:0 \
    wandb_mode=offline \
    >"${OUT}/logs/${method}_train.log" 2>&1
}

pids=()
for index in "${!methods[@]}"; do
  train_one "${methods[$index]}" "${gpus[$index]}" \
    "${modes[$index]}" "${coefficients[$index]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  echo "At least one NCM training arm failed" >&2
  exit 1
fi

eval_one() {
  local method=$1 gpu=$2
  local checkpoint=${OUT}/${method}/train/models/model_10000.pt
  local work=${OUT}/${method}/eval
  local result=${work}/ncm_eval.npy
  if [[ -s "${result}" ]]; then
    return 0
  fi
  mkdir -p "${work}"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${ROOT}/pipelines/ncm_shellgame_eval.py" \
    --config-name=shell_game_touch_ncm \
    dataset_path="${DATA}" \
    normalizer_dataset_path="${SOURCE_DATA}" \
    model_path="${checkpoint}" \
    pairing_checkpoint_path="${BASE}" \
    work_dir="${work}" \
    result_path="${result}" \
    exp_name="ncm_shellgame/${method}_eval" \
    eval_seed_start=20260807 \
    diffusion_seed=20260807 \
    eval_episodes=100 \
    num_envs=10 \
    inference_action_steps=1 \
    ncm_decision_step=6 \
    ncm_intervention_steps=8 \
    ncm_commit_threshold=0.12 \
    counterfactual_negative_mode=none \
    counterfactual_memory_coeff=0.0 \
    shellgame_hold_steps=0 \
    seed=100 \
    device=cuda:0 \
    wandb_mode=offline \
    >"${OUT}/logs/${method}_eval.log" 2>&1
}

pids=()
for index in "${!methods[@]}"; do
  eval_one "${methods[$index]}" "${gpus[$index]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  echo "At least one NCM rollout evaluation failed" >&2
  exit 1
fi

offline_one() {
  local method=$1 gpu=$2
  local checkpoint=${OUT}/${method}/train/models/model_10000.pt
  local work=${OUT}/${method}/offline
  local result=${work}/ncm_offline_gap.json
  if [[ -s "${result}" ]]; then
    return 0
  fi
  mkdir -p "${work}"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${ROOT}/pipelines/ncm_shellgame_offline_eval.py" \
    --config-name=shell_game_touch_ncm \
    dataset_path="${DATA}" \
    normalizer_dataset_path="${SOURCE_DATA}" \
    ncm_pair_manifest_path="${MANIFEST}" \
    model_path="${checkpoint}" \
    work_dir="${work}" \
    offline_result_path="${result}" \
    counterfactual_negative_mode=matched \
    counterfactual_memory_coeff=1.0 \
    counterfactual_memory_probability=1.0 \
    shellgame_hold_steps=0 \
    seed=100 \
    device=cuda:0 \
    wandb_mode=offline \
    >"${OUT}/logs/${method}_offline.log" 2>&1
}

pids=()
for index in "${!methods[@]}"; do
  offline_one "${methods[$index]}" "${gpus[$index]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  echo "At least one NCM offline-gap evaluation failed" >&2
  exit 1
fi

"${PY}" "${ROOT}/tools/analyze_ncm_shellgame.py" \
  --vpwem "${OUT}/vpwem/eval/ncm_eval.npy" \
  --vpwem-offline "${OUT}/vpwem/offline/ncm_offline_gap.json" \
  --random-cf "${OUT}/random_cf/eval/ncm_eval.npy" \
  --random-cf-offline "${OUT}/random_cf/offline/ncm_offline_gap.json" \
  --unmatched-branch-cf "${OUT}/unmatched_branch_cf/eval/ncm_eval.npy" \
  --unmatched-branch-cf-offline "${OUT}/unmatched_branch_cf/offline/ncm_offline_gap.json" \
  --matched-cf "${OUT}/matched_cf/eval/ncm_eval.npy" \
  --matched-cf-offline "${OUT}/matched_cf/offline/ncm_offline_gap.json" \
  --output "${OUT}/analysis" \
  >"${OUT}/logs/analyze.log" 2>&1

"${PY}" "${ROOT}/tools/write_ncm_reproducibility.py" \
  --stage "${OUT}" \
  --repo "${ROOT}" \
  --dataset "${DATA}" \
  --base-checkpoint "${BASE_CKPT}" \
  --manifest "${MANIFEST}"

touch "${OUT}/COMPLETE"
echo "NCM Stage-1 complete: ${OUT}/analysis/RESULTS.md"
