#!/usr/bin/env bash
set -euo pipefail

# Long-budget base jobs for the NCM cross-benchmark study.  These runs are
# deliberately continuous from step 0 so the cosine scheduler has the final
# horizon from the start.  Checkpoints at 5k/10k/20k are diagnostics; the
# declared final budget remains the primary comparison.

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
DATA_ROOT=${NCM_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
SHELL_BASE=${NCM_SHELL_BASE:-/home/jrw/vpwem_outputs/shell_game_touch/ours/mikasa/ShellGameTouch-v0/obs_2_act_8/8layer_5_qformer_2_sin_pos_2_fifo_8_0.0001/100/models/model_600000.pt}

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}

mkdir -p "${OUT}/logs"

run_job() {
  local name=$1 gpu=$2 config=$3 budget=$4 batch=$5 data=$6 work=$7
  shift 7
  mkdir -p "${work}"
  echo "[$(date -Is)] START ${name} gpu=${gpu} budget=${budget}" >&2
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" \
    "${ROOT}/pipelines/longmemdpptp_mikasa_image.py" \
    --config-name="${config}" \
    dataset_path="${data}" \
    work_dir="${work}" \
    save_path="${OUT}" \
    exp_name="crossbench/${name}" \
    gradient_steps="${budget}" \
    batch_size="${batch}" \
    save_freq=20000 \
    ++checkpoint_save_early_steps='[5000,10000]' \
    ++keep_checkpoints=40 \
    ++num_workers=8 \
    seed=100 \
    device=cuda:0 \
    wandb_mode=offline \
    "$@" \
    >"${OUT}/logs/${name}.log" 2>&1
  echo "[$(date -Is)] DONE ${name}" >&2
}

case "${1:-all}" in
  shell_scratch)
    run_job shell_physical_scratch600 2 shell_game_touch_ncm 600000 16 \
      "${DATA_ROOT}/ShellGameTouchNeutralHoldOracle-v0" \
      "${OUT}/shell/physical_scratch600/train" \
      lr=0.0001 +dataset_preserve_episode_anchor=true
    ;;
  shell_transfer)
    test -s "${SHELL_BASE}"
    run_job shell_physical_transfer100 3 shell_game_touch_ncm 100000 16 \
      "${DATA_ROOT}/ShellGameTouchNeutralHoldOracle-v0" \
      "${OUT}/shell/physical_transfer100/train" \
      init_model_path="${SHELL_BASE}" \
      normalizer_dataset_path="${DATA_ROOT}/ShellGameTouch-v0" \
      lr=0.00001 +dataset_preserve_episode_anchor=true
    ;;
  rc3)
    run_job remember_color3_parity600 4 remember_color_3 600000 64 \
      "${DATA_ROOT}/RememberColor3-v0" \
      "${OUT}/remember_color3/parity600/train" \
      +inference_action_steps=4 +dataset_preserve_episode_anchor=true
    ;;
  rc5)
    run_job remember_color5_base400 5 remember_color_5 400000 64 \
      "${DATA_ROOT}/RememberColor5-v0" \
      "${OUT}/remember_color5/base400/train" \
      +inference_action_steps=4 +dataset_preserve_episode_anchor=true
    ;;
  intercept)
    run_job intercept_grab_negative_control400 6 intercept_grab_medium 400000 64 \
      "${DATA_ROOT}/InterceptGrabMedium-v0" \
      "${OUT}/intercept_grab/base400/train" \
      +inference_action_steps=8
    ;;
  all)
    "$0" shell_scratch & p1=$!
    "$0" shell_transfer & p2=$!
    "$0" rc3 & p3=$!
    "$0" rc5 & p4=$!
    fail=0
    wait "${p1}" || fail=1
    wait "${p2}" || fail=1
    wait "${p3}" || fail=1
    wait "${p4}" || fail=1
    exit "${fail}"
    ;;
  *)
    echo "usage: $0 [all|shell_scratch|shell_transfer|rc3|rc5|intercept]" >&2
    exit 2
    ;;
esac
