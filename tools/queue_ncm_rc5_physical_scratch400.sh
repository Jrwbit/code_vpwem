#!/usr/bin/env bash
set -euo pipefail

# Start the clean RememberColor5 physical-neutral baseline only after the two
# shared-base Stage-1 queues have released their short-run GPU allocations.

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
DATA_ROOT=${NCM_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}
DATA=${DATA_ROOT}/RememberColor5NeutralHoldOracle-v0
WORK=${OUT}/remember_color5/physical_scratch400/train
GPU=${NCM_RC5_SCRATCH_GPU:-3}

export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "${WORK}" "${OUT}/logs"
while [[ ! -f "${OUT}/shell/stage1_transfer100/COMPLETE" || \
         ! -f "${OUT}/remember_color5/stage1_transfer100/COMPLETE" ]]; do
  sleep 30
done

if [[ -s "${WORK}/models/model_400000.pt" ]]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES=${GPU}
exec "${PY}" "${ROOT}/pipelines/longmemdpptp_mikasa_image.py" \
  --config-name=remember_color_5 \
  dataset_path="${DATA}" \
  +normalizer_dataset_path=null \
  +init_model_path=null \
  +counterfactual_negative_mode=none \
  +counterfactual_memory_coeff=0.0 \
  +ncm_pair_manifest_path=null \
  +ncm_pair_batch_fraction=1.0 \
  +dataset_preserve_episode_anchor=true \
  mem_compress_method=qformer mem_compress_length=2 \
  qformer_num_layers=2 qformer_cache_compress=fifo \
  qformer_cache_max_length=8 diffusion_num_layers=8 \
  gradient_steps=400000 batch_size=64 lr=0.0001 \
  save_freq=20000 ++checkpoint_save_early_steps='[5000,10000]' \
  ++keep_checkpoints=25 ++num_workers=8 \
  +inference_action_steps=1 \
  work_dir="${WORK}" save_path="${OUT}" \
  exp_name=remember_color5/physical_scratch400 \
  seed=100 device=cuda:0 wandb_mode=offline
