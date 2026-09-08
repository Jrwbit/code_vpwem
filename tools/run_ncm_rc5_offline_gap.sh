#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 METHOD CHECKPOINT [GPU]" >&2
  exit 2
fi
METHOD=$1
CHECKPOINT=$2
GPU=${3:-6}
case "${METHOD}" in
  vpwem|random_cf|unmatched_branch_cf|matched_cf) ;;
  *) echo "unknown RC5 method: ${METHOD}" >&2; exit 2 ;;
esac

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
DATA=/home/jrw/vpwem_data/mikasa/RememberColor5NeutralHoldOracle-v0
SOURCE=/home/jrw/vpwem_data/mikasa/RememberColor5-v0
MANIFEST=${OUT}/remember_color5/pair_manifest/ncm_shellgame_pairs.npz
MODEL=${OUT}/remember_color5/stage1/${METHOD}/train/models/model_${CHECKPOINT}.pt
WORK=${OUT}/remember_color5/stage1/${METHOD}/offline_${CHECKPOINT}
RESULT=${WORK}/offline_gap.json
LOG=${OUT}/logs/rc5_${METHOD}_${CHECKPOINT}_offline_gap.log

test -s "${MODEL}" && test -s "${MANIFEST}"
test -s "${RESULT}" && exit 0
mkdir -p "${WORK}" "${OUT}/logs"
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=${ROOT}:${ROOT}/pipelines${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

exec 9>"${OUT}/.ncm-offline-gap-gpu${GPU}.lock"
flock 9
"${PY}" "${ROOT}/pipelines/ncm_shellgame_offline_eval.py" \
  --config-name=remember_color_5_ncm \
  dataset_path="${DATA}" normalizer_dataset_path="${SOURCE}" \
  ncm_pair_manifest_path="${MANIFEST}" model_path="${MODEL}" \
  work_dir="${WORK}" offline_result_path="${RESULT}" \
  mem_compress_method=qformer mem_compress_length=2 qformer_num_layers=2 \
  qformer_cache_compress=fifo qformer_cache_max_length=8 diffusion_num_layers=8 \
  counterfactual_negative_mode=matched counterfactual_memory_coeff=0.5 \
  counterfactual_memory_margin=0.1 counterfactual_memory_probability=1.0 \
  offline_seed=20260823 offline_mc_samples=4 offline_max_batches=-1 \
  +dataset_preserve_episode_anchor=true num_workers=8 device=cuda:0 \
  >"${LOG}" 2>&1
test -s "${RESULT}"
