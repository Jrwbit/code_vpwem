#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 {shell|rc5} STAGE_DIR [CHECKPOINT]" >&2
  exit 2
fi

TASK=$1
STAGE=$2
CHECKPOINT=${3:-20000}
ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
DATA_ROOT=${NCM_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}

case "${TASK}" in
  shell)
    CONFIG=shell_game_touch_ncm
    DATA=${DATA_ROOT}/ShellGameTouchNeutralHoldOracle-v0
    SOURCE=${DATA_ROOT}/ShellGameTouch-v0
    DEFAULT_GPUS="2 3 6 5"
    ;;
  rc5)
    CONFIG=remember_color_5_ncm
    DATA=${DATA_ROOT}/RememberColor5NeutralHoldOracle-v0
    SOURCE=${DATA_ROOT}/RememberColor5-v0
    DEFAULT_GPUS="2 3 5 6"
    ;;
  *)
    echo "unknown task: ${TASK}" >&2
    exit 2
    ;;
esac

MANIFEST=${STAGE}/pair_manifest/ncm_shellgame_pairs.npz
ANALYSIS=${STAGE}/analysis_${CHECKPOINT}
METHODS=(vpwem random_cf unmatched_branch_cf matched_cf)
read -r -a GPUS <<<"${NCM_STAGE_ANALYSIS_GPUS:-${DEFAULT_GPUS}}"
if (( ${#GPUS[@]} != ${#METHODS[@]} )); then
  echo "NCM_STAGE_ANALYSIS_GPUS must contain four GPU ids" >&2
  exit 2
fi

export PYTHONPATH=${ROOT}:${ROOT}/pipelines${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
mkdir -p "${ANALYSIS}" "${OUT}/logs"
exec 9>"${STAGE}/.analysis_${CHECKPOINT}.lock"
flock 9
[[ ! -s "${ANALYSIS}/summary.json" ]] || exit 0
test -s "${MANIFEST}"

offline_one() {
  local method=$1 gpu=$2
  local model=${STAGE}/${method}/train/models/model_${CHECKPOINT}.pt
  local work=${STAGE}/${method}/offline_${CHECKPOINT}
  local result=${work}/offline_gap.json
  test -s "${model}"
  [[ ! -s "${result}" ]] || return 0
  mkdir -p "${work}"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" \
    "${ROOT}/pipelines/ncm_shellgame_offline_eval.py" \
    --config-name="${CONFIG}" \
    dataset_path="${DATA}" normalizer_dataset_path="${SOURCE}" \
    ncm_pair_manifest_path="${MANIFEST}" model_path="${model}" \
    work_dir="${work}" offline_result_path="${result}" \
    mem_compress_method=qformer mem_compress_length=2 qformer_num_layers=2 \
    qformer_cache_compress=fifo qformer_cache_max_length=8 \
    diffusion_num_layers=8 counterfactual_negative_mode=matched \
    counterfactual_memory_coeff=0.5 counterfactual_memory_margin=0.1 \
    counterfactual_memory_probability=1.0 offline_seed=20260823 \
    offline_mc_samples=4 offline_max_batches=-1 \
    +dataset_preserve_episode_anchor=true num_workers=8 device=cuda:0 \
    >"${OUT}/logs/$(basename "${STAGE}")_${method}_${CHECKPOINT}_offline.log" 2>&1
  test -s "${result}"
}

pids=()
for index in "${!METHODS[@]}"; do
  offline_one "${METHODS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || exit 1

args=()
for method in "${METHODS[@]}"; do
  cli=${method//_/-}
  rollout=${STAGE}/${method}/eval_${CHECKPOINT}/ncm_eval.npy
  offline=${STAGE}/${method}/offline_${CHECKPOINT}/offline_gap.json
  test -s "${rollout}" && test -s "${offline}"
  args+=("--${cli}" "${rollout}" "--${cli}-offline" "${offline}")
done
"${PY}" "${ROOT}/tools/analyze_ncm_shellgame.py" \
  "${args[@]}" --output "${ANALYSIS}" \
  >"${OUT}/logs/$(basename "${STAGE}")_${CHECKPOINT}_analysis.log" 2>&1
test -s "${ANALYSIS}/summary.json"
touch "${STAGE}/ANALYSIS_${CHECKPOINT}_COMPLETE"
