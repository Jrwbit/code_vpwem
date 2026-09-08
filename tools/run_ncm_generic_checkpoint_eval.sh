#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 5 )); then
  echo "usage: $0 RUN CHECKPOINT EPISODES [GPU] [NUM_ENVS]" >&2
  exit 2
fi

RUN=$1
CHECKPOINT=$2
EPISODES=$3
GPU=${4:-4}
NUM_ENVS=${5:-10}

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
MIKASA=${NCM_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}

case "${RUN}" in
  rc3)
    CONFIG=remember_color_3
    DATA=${MIKASA}/RememberColor3-v0
    TRAIN=${OUT}/remember_color3/parity600/train
    INFERENCE=4
    MAX_STEPS=60
    ;;
  rc5_base)
    CONFIG=remember_color_5
    DATA=${MIKASA}/RememberColor5-v0
    TRAIN=${OUT}/remember_color5/base400/train
    INFERENCE=4
    MAX_STEPS=60
    ;;
  intercept)
    CONFIG=intercept_grab_medium
    DATA=${MIKASA}/InterceptGrabMedium-v0
    TRAIN=${OUT}/intercept_grab/base400/train
    INFERENCE=8
    MAX_STEPS=90
    ;;
  *)
    echo "unknown generic run: ${RUN}" >&2
    exit 2
    ;;
esac

MODEL=${TRAIN}/models/model_${CHECKPOINT}.pt
EVAL=${TRAIN%/train}/eval_${CHECKPOINT}_chunk${INFERENCE}
RESULT=${EVAL}/result.npy
LOG=${OUT}/logs/${RUN}_${CHECKPOINT}_generic_eval.log
test -s "${MODEL}"
if test -s "${RESULT}"; then
  echo "[$(date -Is)] SKIP ${RUN} ${CHECKPOINT}: result exists"
  exit 0
fi
if (( EPISODES % NUM_ENVS != 0 )); then
  echo "EPISODES must be divisible by NUM_ENVS" >&2
  exit 2
fi

mkdir -p "${EVAL}" "${OUT}/logs"
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

exec 9>"${OUT}/.ncm-generic-eval-gpu${GPU}.lock"
flock 9
echo "[$(date -Is)] START ${RUN} ckpt=${CHECKPOINT} episodes=${EPISODES} gpu=${GPU}" \
  >>"${LOG}"
"${PY}" "${ROOT}/pipelines/longmemdpptp_mikasa_image_eval.py" \
  --config-name="${CONFIG}" dataset_path="${DATA}" \
  model_path="${MODEL}" +result_path="${RESULT}" \
  work_dir="${EVAL}" num_envs="${NUM_ENVS}" eval_episodes="${EPISODES}" \
  ++inference_action_steps="${INFERENCE}" max_episode_steps="${MAX_STEPS}" \
  save_video=false seed=20260823 device=cuda:0 wandb_mode=offline \
  >>"${LOG}" 2>&1
test -s "${RESULT}"
echo "[$(date -Is)] DONE ${RUN} ckpt=${CHECKPOINT}" >>"${LOG}"
