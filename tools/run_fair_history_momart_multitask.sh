#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 GPU METHOD[:PHYSICAL_BATCH]..." >&2
    exit 2
fi

gpu=$1
shift
repo=/home/jrw/code_vpwem_momart_fair
python_bin=/home/jrw/anaconda3/envs/vpwem/bin/python
output_root=/home/jrw/vpwem_outputs/fair_history_momart_multitask
log_root=${output_root}/logs
supervisor_log=${log_root}/gpu${gpu}_supervisor.log

mkdir -p "${log_root}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTHONPATH=${repo}
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${supervisor_log}"
}

for method_spec in "$@"; do
    method=${method_spec%%:*}
    physical_batch=${method_spec##*:}
    if [[ ${physical_batch} == "${method}" ]]; then
        physical_batch=1
    fi
    work=${output_root}/${method}/seed100
    train_log=${log_root}/${method}_gpu${gpu}_train.log
    if [[ -s ${work}/models/model_1000000.pt ]]; then
        log "SKIP existing final checkpoint method=${method}"
        continue
    fi
    log "START method=${method} gpu=${gpu} physical_batch=${physical_batch} effective_batch=64"
    cd "${repo}"
    "${python_bin}" pipelines/fair_history_momart_multitask.py \
        history_method="${method}" physical_batch_size="${physical_batch}" \
        >"${train_log}" 2>&1
    log "DONE method=${method}"
done

date -Is >"${output_root}/GPU${gpu}_COMPLETE"
log "QUEUE COMPLETE gpu=${gpu}"
