#!/usr/bin/env bash
set -euo pipefail
repo=/home/jrw/code_vpwem_boxed_bs64_1m_20261029
work=/home/jrw/vpwem_outputs/momart_boxed_no_history_bs64_1m_20261029
mkdir -p "$work/seed100"
exec 9>"$work/train.lock"
flock -n 9 || exit 1
cd "$repo"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$repo"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=8
exec /home/jrw/anaconda3/envs/vpwem/bin/python -m momart_boxed_no_history.train \
  --config-path "$repo/experiments/momart_boxed_nohistory_bs64_1m_20261029" \
  --config-name config hydra.run.dir="$work/hydra"
