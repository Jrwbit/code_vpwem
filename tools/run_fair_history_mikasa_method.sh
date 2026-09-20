#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 GPU TASK METHOD [MAX_STEPS] [EXISTING_TRAIN_PID]" >&2
    exit 2
fi

gpu=$1
task=$2
method=$3
max_steps=${4:-90}
existing_pid=${5:-}
repo=/home/jrw/code_vpwem_fair_history
python_bin=/home/jrw/anaconda3/envs/vpwem/bin/python
output_root=/home/jrw/vpwem_outputs/fair_history_mikasa
work=${output_root}/${task}/${method}/seed100
train_log=${output_root}/logs/${task}_${method}_gpu${gpu}_train.log
eval_log=${output_root}/logs/${task}_${method}_gpu${gpu}_eval.log
status_log=${output_root}/logs/${task}_${method}_gpu${gpu}_supervisor.log

mkdir -p "${output_root}/logs"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTHONPATH=${repo}
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${status_log}"
}

if [[ ! -s ${work}/models/model_600000.pt ]]; then
    if [[ -n ${existing_pid} ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        log "WAIT existing train pid=${existing_pid}"
        while kill -0 "${existing_pid}" 2>/dev/null; do sleep 60; done
        [[ -s ${work}/models/model_600000.pt ]] || {
            log "existing process ended without final checkpoint; resume"
            existing_pid=
        }
    fi
    if [[ -z ${existing_pid} ]]; then
        log "TRAIN task=${task} method=${method} gpu=${gpu}"
        cd "${repo}"
        "${python_bin}" pipelines/fair_history_mikasa.py \
            env_name="${task}" history_method="${method}" \
            max_episode_steps="${max_steps}" mode=train \
            >>"${train_log}" 2>&1
    fi
fi

if [[ ! -s ${work}/models/model_600000.pt ]]; then
    log "FAIL missing model_600000.pt"
    exit 1
fi

if [[ ! -s ${work}/result_600000_paired.npy ]]; then
    log "EVAL task=${task} method=${method} gpu=${gpu}"
    cd "${repo}"
    "${python_bin}" pipelines/fair_history_mikasa.py \
        env_name="${task}" history_method="${method}" \
        max_episode_steps="${max_steps}" mode=eval \
        >"${eval_log}" 2>&1
fi

date -Is >"${work}/COMPLETE"
log "DONE task=${task} method=${method}"
