#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/wjr/code/vpwem
PY=/home/wjr/miniconda3/envs/vpwem5090/bin/python
OUT=${ROOT}/artifacts/robomimic_all_checkpoints
SQUARE_MODELS=${ROOT}/outputs/ours/robomimic_emb/square_ph/vpwem_reproduction_run2/100/models
LONG_MODELS=${ROOT}/outputs/ours/robomimic_emb/longsquare_ph/obs_2_act_8/20_qformer_4_sin_pos_1_adjsim_8_0.0001/100/models
DATA=${ROOT}/data/robomimic/datasets
CONTEXT=${ROOT}/outputs/robomimic/dpptp
EPISODES=${ROBOMIMIC_EVAL_EPISODES:-30}
MAX_PARALLEL=${ROBOMIMIC_EVAL_PARALLEL:-2}
NUM_ENVS=${ROBOMIMIC_EVAL_NUM_ENVS:-5}

export WANDB_MODE=offline HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/wjr/miniconda3/envs/vpwem5090/lib/python3.9/site-packages/mujoco_py/binaries/linux${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
mkdir -p "${OUT}/logs" "${OUT}/legacy4"
QUEUE_LOG=${OUT}/logs/queue.log
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${QUEUE_LOG}"; }

eval_one() {
    local task=$1 step=$2 model_root env_name raw context mem_len cache
    if [[ ${task} == square_ph ]]; then
        model_root=${SQUARE_MODELS}; env_name=square
        raw=${DATA}/square/ph/image_abs.hdf5
        context=${CONTEXT}/square_ph/obs_2_act_8/100/models/model_1000000.pt
        mem_len=2; cache=fifo
    else
        model_root=${LONG_MODELS}; env_name=longsquare
        raw=${DATA}/longhistsquare100/demos.hdf5
        context=${CONTEXT}/longsquare_ph/obs_2_act_8/100/models/model_1000000.pt
        mem_len=4; cache=adjsim
    fi
    local model=${model_root}/model_${step}.pt
    local work=${OUT}/legacy4/${task}/${step}
    local result=${work}/result_${step}_memory-none.npy
    local logfile=${OUT}/logs/${task}_${step}.log
    if [[ -s ${result} ]]; then log "SKIP ${task}@${step}"; return 0; fi
    for required in "${model}" "${raw}" "${context}"; do
        [[ -s ${required} ]] || { log "FAIL ${task}@${step}: missing ${required}"; return 2; }
    done
    mkdir -p "${work}"
    log "START ${task}@${step} (legacy 4-layer, ${EPISODES} rollouts)"
    local pipeline=${ROOT}/pipelines/longmemdpptp_robomimic_image_emb_eval.py
    [[ ${task} == longsquare_ph ]] && pipeline=${ROOT}/pipelines/longmemdpptp_robomimic_lhsq_image_emb_eval.py
    local common=(
        --config-name=${env_name}_abs_emb_eval env_name=${env_name} env_type=ph
        dataset_path=${raw} model_path=${model} +context_model_path=${context} +ckpt=${step}
        diffusion_num_layers=4 sample_steps=50 action_steps=8 obs_steps=2 horizon=9
        mem_compress_method=qformer mem_compress_length=${mem_len} qformer_num_layers=1
        qformer_cache_compress=${cache} qformer_cache_max_length=8 dataset_memory_subsample_ratio=20
        short_cond_dropout=0.3 long_cond_dropout=0.3 +inference_action_steps=8
        +dataset_maximum_memory_length=16 +eval_normalizer_only=true
        seed=20260908 device=cuda:0 num_envs=${NUM_ENVS} eval_episodes=${EPISODES}
        save_video=false work_dir=${work} wandb_mode=offline
    )
    if [[ ${task} == longsquare_ph ]]; then
        "${PY}" "${pipeline}" "${common[@]}" memory_intervention=none >"${logfile}" 2>&1
    else
        "${PY}" "${pipeline}" "${common[@]}" '+memory_interventions=[none]' >"${logfile}" 2>&1
    fi
    local rc=$?
    if [[ ${rc} -eq 0 && -s ${result} ]]; then log "DONE ${task}@${step}"; return 0; fi
    log "FAIL ${task}@${step}: rc=${rc}; see ${logfile}"
    return 1
}

mapfile -t square_steps < <(find "${SQUARE_MODELS}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sed -nE 's/model_([0-9]+)\.pt/\1/p' | awk '$1 >= 100000 && $1 <= 1000000' | sort -n)
mapfile -t long_steps < <(find "${LONG_MODELS}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sed -nE 's/model_([0-9]+)\.pt/\1/p' | awk '$1 >= 100000 && $1 <= 1000000' | sort -n)
log "QUEUE BEGIN: square=${#square_steps[@]} points, longsquare=${#long_steps[@]} points, parallel=${MAX_PARALLEL}"
failed=0
pids=()
for spec in $(printf 'square_ph:%s\n' "${square_steps[@]}") $(printf 'longsquare_ph:%s\n' "${long_steps[@]}"); do
    task=${spec%%:*}; step=${spec##*:}
    eval_one "${task}" "${step}" & pids+=("$!")
    if (( ${#pids[@]} >= MAX_PARALLEL )); then
        if ! wait "${pids[0]}"; then failed=1; fi
        pids=("${pids[@]:1}")
    fi
done
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done

if ! REPORT=$("${PY}" "${ROOT}/tools/summarize_robomimic_all_checkpoints.py"); then
    log "FAIL summarization"; exit 1
fi
log "REPORT ${REPORT}"
FEISHU_PY=/home/wjr/.venvs/feishu-codex-bridge/bin/python
if "${FEISHU_PY}" "${ROOT}/tools/feishu_codex_bridge/publish_markdown.py" "${REPORT}" \
    --title 'Robomimic 全部现存 checkpoint 成功率' >"${OUT}/feishu_publish.log" 2>&1; then
    head -n 1 "${OUT}/feishu_publish.log" >"${OUT}/feishu_url.txt"
    log "FEISHU $(cat "${OUT}/feishu_url.txt")"
else
    failed=1; log "FAIL Feishu publish"
fi
if (( failed == 0 )); then date -Is >"${OUT}/COMPLETE"; log "QUEUE COMPLETE"; fi
exit "${failed}"
