#!/usr/bin/env bash
set -euo pipefail

# Minimal mechanism test: all four arms start from the same memory-capable
# Shell transfer100 checkpoint.  The three counterfactual arms use identical
# donor-action positive supervision; only donor selection differs.

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
DATA_ROOT=${NCM_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}
DATA=${DATA_ROOT}/ShellGameTouchNeutralHoldOracle-v0
SOURCE=${DATA_ROOT}/ShellGameTouch-v0
BASE=${NCM_DIRECTIONAL_BASE:-${OUT}/shell/physical_transfer100/train/models/model_100000.pt}
MANIFEST=${NCM_DIRECTIONAL_MANIFEST:-${OUT}/shell/stage1_transfer100/pair_manifest/ncm_shellgame_pairs.npz}
STAGE=${NCM_DIRECTIONAL_STAGE:-${OUT}/shell/directional_donor_action_transfer100}
read -r -a GPUS <<<"${NCM_DIRECTIONAL_GPUS:-2 3 5 6}"

METHODS=(vpwem random_cf unmatched_branch_cf matched_cf)
MODES=(none random unmatched_branch matched)
COEFFICIENTS=(0.0 0.5 0.5 0.5)

if (( ${#GPUS[@]} != 4 )); then
  echo "NCM_DIRECTIONAL_GPUS must contain four GPU ids" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if (( gpu < 2 || gpu > 6 )); then
    echo "Refusing GPU ${gpu}; only physical GPUs 2-6 are allowed" >&2
    exit 2
  fi
done
test -s "${BASE}"
test -s "${MANIFEST}"

export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
mkdir -p "${STAGE}" "${OUT}/logs"

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
    save_path="${OUT}" exp_name="shell/directional/${method}" \
    counterfactual_negative_mode="${mode}" \
    +counterfactual_objective=donor_action \
    counterfactual_memory_coeff="${coefficient}" \
    counterfactual_memory_margin=0.1 \
    counterfactual_memory_probability=1.0 \
    gradient_steps=20000 save_freq=20000 \
    ++checkpoint_save_early_steps='[5000,10000]' keep_checkpoints=3 \
    log_freq=500 batch_size=16 lr=0.00001 seed=100 \
    device=cuda:0 wandb_mode=offline \
    >"${OUT}/logs/shell_directional_${method}_train.log" 2>&1
}

pids=()
for i in "${!METHODS[@]}"; do
  train_one "${METHODS[$i]}" "${GPUS[$i]}" \
    "${MODES[$i]}" "${COEFFICIENTS[$i]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  echo "Directional training arm failed" >&2
  exit 1
fi

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
    >"${OUT}/logs/shell_directional_${method}_${checkpoint}_eval.log" 2>&1
}

pids=()
for i in "${!METHODS[@]}"; do
  (
    eval_one "${METHODS[$i]}" "${GPUS[$i]}" 5000 20
    eval_one "${METHODS[$i]}" "${GPUS[$i]}" 10000 20
    eval_one "${METHODS[$i]}" "${GPUS[$i]}" 20000 100
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  echo "Directional rollout evaluation failed" >&2
  exit 1
fi

analysis_args=()
for method in "${METHODS[@]}"; do
  analysis_args+=(
    "--${method//_/-}"
    "${STAGE}/${method}/eval_20000/ncm_eval.npy"
  )
done
"${PY}" "${ROOT}/tools/analyze_ncm_shellgame.py" \
  "${analysis_args[@]}" --output "${STAGE}/analysis_20000" \
  >"${OUT}/logs/shell_directional_analysis_20000.log" 2>&1

touch "${STAGE}/COMPLETE"
echo "Directional Shell minimal stage complete: ${STAGE}"
