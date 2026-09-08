#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/wjr/code/vpwem
PY=/home/wjr/miniconda3/envs/vpwem5090/bin/python
OUT=${ROBOMIMIC_INTERMEDIATE_OUT:-${ROOT}/artifacts/robomimic_intermediate_local}
CKPTS=${OUT}/strict_ckpts
DATA=${ROOT}/data/robomimic/datasets
CONTEXT=${ROOT}/outputs/robomimic/dpptp
EPISODES=${ROBOMIMIC_EVAL_EPISODES:-30}
MAX_PARALLEL=${ROBOMIMIC_EVAL_PARALLEL:-2}
NUM_ENVS=${ROBOMIMIC_EVAL_NUM_ENVS:-5}

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/wjr/miniconda3/envs/vpwem5090/lib/python3.9/site-packages/mujoco_py/binaries/linux${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

mkdir -p "${OUT}/logs" "${OUT}/results"
QUEUE_LOG=${OUT}/logs/queue.log
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${QUEUE_LOG}"; }

task_meta() {
    case "$1" in
        square_ph) echo 'square ph square/ph/image_abs.hdf5' ;;
        square_mh) echo 'square mh square/mh/image_abs.hdf5' ;;
        transport_ph) echo 'transport ph transport/ph/image_abs.hdf5' ;;
        transport_mh) echo 'transport mh transport/mh/image_abs.hdf5' ;;
        longsquare_ph) echo 'longsquare ph longhistsquare100/demos.hdf5' ;;
        *) return 2 ;;
    esac
}

eval_one() {
    local task=$1 step=$2 env_name env_type raw_rel
    read -r env_name env_type raw_rel < <(task_meta "${task}")
    local model=${CKPTS}/${task}/models/model_${step}.pt
    local raw=${DATA}/${raw_rel}
    local context=${CONTEXT}/${task}/obs_2_act_8/100/models/model_1000000.pt
    # Hydra, wandb and the pipeline logger all write inside work_dir. Keep each
    # checkpoint isolated so two local evaluations can run safely in parallel.
    local work=${OUT}/results/${task}/${step}
    local result=${work}/result_${step}_memory-none.npy
    local logfile=${OUT}/logs/${task}_${step}.log
    if [[ -s "${result}" ]]; then
        log "SKIP ${task}@${step}: result exists"
        return 0
    fi
    for required in "${model}" "${raw}" "${context}"; do
        if [[ ! -s "${required}" ]]; then
            log "FAIL ${task}@${step}: missing ${required}"
            return 2
        fi
    done
    mkdir -p "${work}"
    log "START ${task}@${step} (${EPISODES} rollouts)"
    local pipeline=${ROOT}/pipelines/longmemdpptp_robomimic_image_emb_eval.py
    [[ ${task} == longsquare_ph ]] && pipeline=${ROOT}/pipelines/longmemdpptp_robomimic_lhsq_image_emb_eval.py
    local common=(
        --config-name=${env_name}_abs_emb_eval env_name=${env_name} env_type=${env_type}
        dataset_path=${raw} model_path=${model} +context_model_path=${context} +ckpt=${step}
        diffusion_num_layers=8 sample_steps=50 action_steps=8 obs_steps=2 horizon=9
        mem_compress_method=qformer mem_compress_length=2 qformer_num_layers=2
        qformer_cache_compress=fifo qformer_cache_max_length=8
        dataset_memory_subsample_ratio=20 short_cond_dropout=0.3 long_cond_dropout=0.3
        +inference_action_steps=8 +dataset_maximum_memory_length=160
        +eval_normalizer_only=true seed=20260907 device=cuda:0 num_envs=${NUM_ENVS}
        eval_episodes=${EPISODES} save_video=false work_dir=${work} wandb_mode=offline
    )
    if [[ ${task} == longsquare_ph ]]; then
        "${PY}" "${pipeline}" "${common[@]}" memory_intervention=none >"${logfile}" 2>&1
    else
        "${PY}" "${pipeline}" "${common[@]}" '+memory_interventions=[none]' >"${logfile}" 2>&1
    fi
    local rc=$?
    if [[ ${rc} -eq 0 && -s "${result}" ]]; then
        log "DONE ${task}@${step}"
        return 0
    fi
    log "FAIL ${task}@${step}: rc=${rc}; see ${logfile}"
    return 1
}

log "QUEUE BEGIN: strict 8-layer, checkpoints 800k/900k/1000k, parallel=${MAX_PARALLEL}"
failed=0
pids=()
labels=()
for task in square_ph square_mh transport_ph transport_mh longsquare_ph; do
    for step in 800000 900000 1000000; do
        eval_one "${task}" "${step}" &
        pids+=("$!")
        labels+=("${task}@${step}")
        if (( ${#pids[@]} >= MAX_PARALLEL )); then
            if ! wait "${pids[0]}"; then failed=1; fi
            pids=("${pids[@]:1}")
            labels=("${labels[@]:1}")
        fi
    done
done
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
done

if ! REPORT=$("${PY}" "${ROOT}/tools/summarize_robomimic_intermediate.py" "${OUT}"); then
    log "FAIL result summarization"
    exit 1
fi
log "REPORT ${REPORT}"
FEISHU_PY=/home/wjr/.venvs/feishu-codex-bridge/bin/python
if [[ -x "${FEISHU_PY}" ]]; then
    if "${FEISHU_PY}" "${ROOT}/tools/feishu_codex_bridge/publish_markdown.py" \
        "${REPORT}" --title 'Robomimic 严格 8-layer 中间 checkpoint 成功率' \
        >"${OUT}/feishu_publish.log" 2>&1; then
        head -n 1 "${OUT}/feishu_publish.log" >"${OUT}/feishu_url.txt"
        log "FEISHU $(cat "${OUT}/feishu_url.txt")"
    else
        failed=1
        log "FAIL Feishu publish; see ${OUT}/feishu_publish.log"
    fi
else
    failed=1
    log "FAIL Feishu publisher Python missing"
fi

if (( failed == 0 )); then
    date -Is >"${OUT}/COMPLETE"
    log "QUEUE COMPLETE"
else
    log "QUEUE FINISHED WITH FAILURES; rerun resumes missing points"
fi
exit "${failed}"
