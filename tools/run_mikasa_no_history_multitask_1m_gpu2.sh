#!/usr/bin/env bash
set -euo pipefail

repo=${MIKASA_REPO:-/home/jrw/code_vpwem_fair_history}
python_bin=${MIKASA_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
work=${MIKASA_NO_HISTORY_WORK:-/home/jrw/vpwem_outputs/fair_history_mikasa_multitask/no_history_bs256_fast_1m/seed100}
gpu=${GPU:-2}
batch_size=${BATCH_SIZE:-256}

mkdir -p "$work"
exec 9>"$work/.train.lock"
flock -n 9 || { echo "no_history training is already running: $work" >&2; exit 2; }

if [[ -s "$work/models/model_1000000.pt" ]]; then
    echo "Final checkpoint already exists: $work/models/model_1000000.pt"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
cd "$repo"

printf '[%s] START gpu=%s batch=%s steps=1000000 save_freq=100000 work=%s\n' \
    "$(date -Is)" "$gpu" "$batch_size" "$work" >>"$work/supervisor.log"
"$python_bin" pipelines/fair_history_mikasa_multitask.py \
    history_method=no_history \
    batch_size="$batch_size" \
    gradient_steps=1000000 \
    dataset_maximum_memory_length=1 \
    num_workers=4 \
    save_freq=100000 \
    keep_checkpoints=20 \
    exp_name="no_history/real_bs${batch_size}_fast_1m/seed100" \
    work_dir="$work" \
    hydra.run.dir="$work" \
    >"$work/train.log" 2>&1
printf '[%s] COMPLETE work=%s\n' "$(date -Is)" "$work" >>"$work/supervisor.log"
