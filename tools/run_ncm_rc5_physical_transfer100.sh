#!/usr/bin/env bash
set -euo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
DATA_ROOT=${NCM_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}
GPU=${NCM_RC5_TRANSFER_GPU:-3}
DATA=${DATA_ROOT}/RememberColor5NeutralHoldOracle-v0
SOURCE=${DATA_ROOT}/RememberColor5-v0
BASE=${NCM_RC5_BASE:-/home/jrw/vpwem_outputs/per_vpwem_screen_seed100/fifo_controls/mikasa/RememberColor5-v0/100/models/model_400000.pt}
WORK=${OUT}/remember_color5/physical_transfer100/train

test -s "${BASE}"
mkdir -p "${WORK}" "${OUT}/logs"
if [[ -s "${WORK}/models/model_100000.pt" ]]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

"${PY}" "${ROOT}/pipelines/longmemdpptp_mikasa_image.py" \
  --config-name=remember_color_5 \
  dataset_path="${DATA}" \
  +normalizer_dataset_path="${SOURCE}" \
  +init_model_path="${BASE}" \
  +counterfactual_negative_mode=none \
  +counterfactual_memory_coeff=0.0 \
  +ncm_pair_manifest_path=null \
  +ncm_pair_batch_fraction=1.0 \
  +dataset_preserve_episode_anchor=true \
  mem_compress_method=qformer mem_compress_length=2 \
  qformer_num_layers=2 qformer_cache_compress=fifo \
  qformer_cache_max_length=8 diffusion_num_layers=8 \
  gradient_steps=100000 batch_size=16 lr=0.00001 \
  save_freq=20000 ++checkpoint_save_early_steps='[5000,10000]' \
  ++keep_checkpoints=10 ++num_workers=8 \
  +inference_action_steps=1 \
  work_dir="${WORK}" save_path="${OUT}" \
  exp_name=remember_color5/physical_transfer100 \
  seed=100 device=cuda:0 wandb_mode=offline \
  >"${OUT}/logs/remember_color5_physical_transfer100.log" 2>&1
