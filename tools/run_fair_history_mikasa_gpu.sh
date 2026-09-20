#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 GPU TASK[:MAX_STEPS]..." >&2
    exit 2
fi

gpu=$1
shift
repo=/home/jrw/code_vpwem_fair_history
python_bin=/home/jrw/anaconda3/envs/vpwem/bin/python
output_root=/home/jrw/vpwem_outputs/fair_history_mikasa
supervisor_log=${output_root}/logs/gpu${gpu}_supervisor.log
methods=(no_history vpwem direct_token2 direct)

mkdir -p "${output_root}/logs"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTHONPATH=${repo}
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${supervisor_log}"
}

for task_spec in "$@"; do
    task=${task_spec%%:*}
    max_steps=${task_spec##*:}
    if [[ ${max_steps} == "${task}" ]]; then
        max_steps=90
    fi
    for method in "${methods[@]}"; do
        work=${output_root}/${task}/${method}/seed100
        train_log=${output_root}/logs/${task}_${method}_gpu${gpu}_train.log
        eval_log=${output_root}/logs/${task}_${method}_gpu${gpu}_eval.log
        if [[ ! -s ${work}/models/model_600000.pt ]]; then
            log "TRAIN task=${task} method=${method} gpu=${gpu}"
            cd "${repo}"
            "${python_bin}" pipelines/fair_history_mikasa.py \
                env_name="${task}" history_method="${method}" \
                max_episode_steps="${max_steps}" mode=train \
                >"${train_log}" 2>&1
        else
            log "SKIP TRAIN existing checkpoint task=${task} method=${method}"
        fi
        if [[ ! -s ${work}/result_600000_paired.npy ]]; then
            log "EVAL task=${task} method=${method} gpu=${gpu}"
            cd "${repo}"
            "${python_bin}" pipelines/fair_history_mikasa.py \
                env_name="${task}" history_method="${method}" \
                max_episode_steps="${max_steps}" mode=eval \
                >"${eval_log}" 2>&1
        else
            log "SKIP EVAL existing result task=${task} method=${method}"
        fi
        log "DONE task=${task} method=${method}"
    done
done

date -Is >"${output_root}/GPU${gpu}_COMPLETE"
log "QUEUE COMPLETE gpu=${gpu}"
