#!/usr/bin/env bash
set -uo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
DATA=/home/jrw/vpwem_data/robomimic/datasets
BASE=/home/jrw/vpwem_outputs/paper_strict/robomimic_seed100
CONTEXT=/home/jrw/vpwem_outputs/robomimic/dpptp
EPISODES=${NCM_ROBOMIMIC_EVAL_EPISODES:-100}

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${OUT}/logs" "${OUT}/robomimic_controls"

eval_standard() {
  local task=$1 env=$2 type=$3 gpu=$4 raw=$5 context=$6
  local model=${BASE}/${task}/models/model_1000000.pt
  local work=${OUT}/robomimic_controls/${task}
  local result=${work}/result_1000000_memory-none.npy
  local zero_result=${work}/result_1000000_memory-zero.npy
  local expected_result=${result}
  local interventions='[none,zero]'
  test -s "${model}" && test -s "${raw}" && test -s "${context}" || return 2
  test -s "${result}" && test -s "${zero_result}" && return 0
  if test -s "${result}"; then
    interventions='[zero]'
    expected_result=${zero_result}
  fi
  mkdir -p "${work}"
  CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu} "${PY}" \
    "${ROOT}/pipelines/longmemdpptp_robomimic_image_emb_eval.py" \
    --config-name="${env}_abs_emb_eval" env_name="${env}" env_type="${type}" \
    dataset_path="${raw}" model_path="${model}" \
    +context_model_path="${context}" +ckpt=1000000 \
    diffusion_num_layers=8 sample_steps=50 action_steps=8 obs_steps=2 horizon=9 \
    mem_compress_method=qformer mem_compress_length=2 qformer_num_layers=2 \
    qformer_cache_compress=fifo qformer_cache_max_length=8 \
    dataset_memory_subsample_ratio=20 short_cond_dropout=0.3 long_cond_dropout=0.3 \
    +inference_action_steps=8 +dataset_maximum_memory_length=160 \
    +eval_normalizer_only=true "++memory_interventions=${interventions}" \
    +diffusion_seed=20260823 seed=20260823 device=cuda:0 \
    num_envs=10 eval_episodes="${EPISODES}" save_video=false \
    work_dir="${work}" wandb_mode=offline \
    >"${OUT}/logs/${task}_control_eval.log" 2>&1
  test -s "${expected_result}"
}

eval_longsquare() {
  local gpu=$1 task=longsquare_ph
  local raw=${DATA}/longhistsquare100/demos.hdf5
  local model=${BASE}/${task}/models/model_1000000.pt
  local context=${CONTEXT}/${task}/obs_2_act_8/100/models/model_1000000.pt
  local work=${OUT}/robomimic_controls/${task}
  local result=${work}/result_1000000_memory-none.npy
  local zero_result=${work}/result_1000000_memory-zero.npy
  local intervention=none
  local expected_result=${result}
  test -s "${model}" && test -s "${raw}" && test -s "${context}" || return 2
  test -s "${result}" && test -s "${zero_result}" && return 0
  if test -s "${result}"; then
    intervention=zero
    expected_result=${zero_result}
  fi
  mkdir -p "${work}"
  CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu} "${PY}" \
    "${ROOT}/pipelines/longmemdpptp_robomimic_lhsq_image_emb_eval.py" \
    --config-name=longsquare_abs_emb_eval env_name=longsquare env_type=ph \
    dataset_path="${raw}" model_path="${model}" \
    +context_model_path="${context}" +ckpt=1000000 \
    diffusion_num_layers=8 sample_steps=50 action_steps=8 obs_steps=2 horizon=9 \
    mem_compress_method=qformer mem_compress_length=2 qformer_num_layers=2 \
    qformer_cache_compress=fifo qformer_cache_max_length=8 \
    dataset_memory_subsample_ratio=20 short_cond_dropout=0.3 long_cond_dropout=0.3 \
    +inference_action_steps=8 +dataset_maximum_memory_length=160 \
    memory_intervention="${intervention}" seed=20260823 device=cuda:0 \
    num_envs=10 eval_episodes="${EPISODES}" save_video=false \
    work_dir="${work}" wandb_mode=offline \
    >"${OUT}/logs/${task}_control_eval.log" 2>&1
  test -s "${expected_result}"
}

eval_longsquare 2 & p1=$!
eval_standard square_mh square mh 3 \
  "${DATA}/square/mh/image_abs.hdf5" \
  "${CONTEXT}/square_mh/obs_2_act_8/100/models/model_1000000.pt" & p2=$!
eval_standard transport_ph transport ph 5 \
  "${DATA}/transport/ph/image_abs.hdf5" \
  "${CONTEXT}/transport_ph/obs_2_act_8/100/models/model_1000000.pt" & p3=$!

failed=0
wait "${p1}" || failed=1
wait "${p2}" || failed=1
wait "${p3}" || failed=1
if (( failed == 0 )); then date -Is >"${OUT}/ROBOMIMIC_CONTROLS_COMPLETE"; fi
exit "${failed}"
