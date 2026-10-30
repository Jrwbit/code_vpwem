#!/usr/bin/env bash
set -euo pipefail
repo=/home/jrw/code_vpwem_momart_a1_bs256_250k_20261030
work=/home/jrw/vpwem_outputs/momart_a1_bs256_250k_20261030
mkdir -p "$work/seed100"
exec 9>"$work/pipeline.lock"
flock -n 9 || exit 1
cd "$repo"
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH="$repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
exec /home/jrw/anaconda3/envs/vpwem/bin/python -m momart_object_memory.prepare_train \
  --config "$repo/experiments/momart_a1_bs256_250k_20261030/config.yaml" --device cuda:0
