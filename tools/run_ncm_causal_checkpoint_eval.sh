#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 5 )); then
  echo "usage: $0 RUN CHECKPOINT EPISODES [GPU] [NUM_ENVS]" >&2
  exit 2
fi

RUN=$1
CHECKPOINT=$2
EPISODES=$3
GPU=${4:-6}
NUM_ENVS=${5:-10}

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
MIKASA=${NCM_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}

case "${RUN}" in
  shell_scratch)
    CONFIG=shell_game_touch_ncm
    DATA=${MIKASA}/ShellGameTouchNeutralHoldOracle-v0
    TRAIN=${OUT}/shell/physical_scratch600/train
    EVAL=${OUT}/shell/physical_scratch600/eval_${CHECKPOINT}
    MODEL=${TRAIN}/models/model_${CHECKPOINT}.pt
    EXTRA=()
    ;;
  shell_transfer)
    CONFIG=shell_game_touch_ncm
    DATA=${MIKASA}/ShellGameTouchNeutralHoldOracle-v0
    TRAIN=${OUT}/shell/physical_transfer100/train
    EVAL=${OUT}/shell/physical_transfer100/eval_${CHECKPOINT}
    MODEL=${TRAIN}/models/model_${CHECKPOINT}.pt
    EXTRA=(normalizer_dataset_path=${MIKASA}/ShellGameTouch-v0)
    ;;
  rc5_transfer)
    CONFIG=remember_color_5_ncm
    DATA=${MIKASA}/RememberColor5NeutralHoldOracle-v0
    TRAIN=${OUT}/remember_color5/physical_transfer100/train
    EVAL=${OUT}/remember_color5/physical_transfer100/eval_${CHECKPOINT}
    MODEL=${TRAIN}/models/model_${CHECKPOINT}.pt
    EXTRA=(
      normalizer_dataset_path=${MIKASA}/RememberColor5-v0
      mem_compress_method=qformer mem_compress_length=2
      qformer_num_layers=2 qformer_cache_compress=fifo
      qformer_cache_max_length=8 diffusion_num_layers=8
    )
    ;;
  rc5_scratch)
    CONFIG=remember_color_5_ncm
    DATA=${MIKASA}/RememberColor5NeutralHoldOracle-v0
    TRAIN=${OUT}/remember_color5/physical_scratch400/train
    EVAL=${OUT}/remember_color5/physical_scratch400/eval_${CHECKPOINT}
    MODEL=${TRAIN}/models/model_${CHECKPOINT}.pt
    EXTRA=(
      mem_compress_method=qformer mem_compress_length=2
      qformer_num_layers=2 qformer_cache_compress=fifo
      qformer_cache_max_length=8 diffusion_num_layers=8
    )
    ;;
  rc5_vpwem|rc5_random_cf|rc5_unmatched_branch_cf|rc5_matched_cf)
    METHOD=${RUN#rc5_}
    CONFIG=remember_color_5_ncm
    DATA=${MIKASA}/RememberColor5NeutralHoldOracle-v0
    TRAIN=${OUT}/remember_color5/stage1/${METHOD}/train
    EVAL=${OUT}/remember_color5/stage1/${METHOD}/eval_${CHECKPOINT}
    MODEL=${TRAIN}/models/model_${CHECKPOINT}.pt
    EXTRA=(
      normalizer_dataset_path=${MIKASA}/RememberColor5-v0
      mem_compress_method=qformer mem_compress_length=2
      qformer_num_layers=2 qformer_cache_compress=fifo
      qformer_cache_max_length=8 diffusion_num_layers=8
    )
    ;;
  *)
    echo "unknown causal run: ${RUN}" >&2
    exit 2
    ;;
esac

RESULT=${EVAL}/ncm_eval.npy
SUMMARY=${EVAL}/ncm_eval.json
LOG=${OUT}/logs/${RUN}_${CHECKPOINT}_causal_eval.log
test -s "${MODEL}"
if test -s "${RESULT}" && test -s "${SUMMARY}"; then
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

# Multiple checkpoint watchers can become ready at once.  Serialize only the
# evaluator on the selected GPU; training remains concurrent.
exec 9>"${OUT}/.ncm-causal-eval-gpu${GPU}.lock"
flock 9
echo "[$(date -Is)] START ${RUN} ckpt=${CHECKPOINT} episodes=${EPISODES} gpu=${GPU}" \
  >>"${LOG}"
"${PY}" "${ROOT}/pipelines/ncm_shellgame_eval.py" \
  --config-name="${CONFIG}" \
  dataset_path="${DATA}" \
  model_path="${MODEL}" \
  pairing_checkpoint_path="${MODEL}" \
  work_dir="${EVAL}" result_path="${RESULT}" \
  num_envs="${NUM_ENVS}" eval_episodes="${EPISODES}" \
  save_video=false device=cuda:0 wandb_mode=offline \
  "${EXTRA[@]}" >>"${LOG}" 2>&1
test -s "${RESULT}"
test -s "${SUMMARY}"
echo "[$(date -Is)] DONE ${RUN} ckpt=${CHECKPOINT}" >>"${LOG}"
