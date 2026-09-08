#!/usr/bin/env bash
set -euo pipefail

repo=/home/wjr/code/vpwem
python=/home/wjr/miniconda3/envs/vpwem5090/bin/python
run_dir=$repo/outputs/rmbench/longmemdpptp_rmbench_image_emb/train/swap_blocks/paper_l2s2_full100k/100
model=$run_dir/models/model_100001.pt
log_dir=$repo/outputs/rmbench/full_local_supervisor
status=$log_dir/status.tsv

mkdir -p "$run_dir" "$log_dir"
exec 9>"$log_dir/run.lock"
if ! flock -n 9; then
    echo "RMBench full local supervisor is already running"
    exit 0
fi

exec > >(tee -a "$log_dir/supervisor.log") 2>&1
export PYTHONPATH=$repo:$repo/pipelines
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

stage() {
    printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" | tee -a "$status"
}

if [[ ! -s "$model" ]]; then
    stage train START
    "$python" "$repo/pipelines/longmemdpptp_rmbench_image_emb.py" \
        --config-name=swap_blocks_emb_agentpos_dpaligned \
        dataset_path=$repo/data/rmbench/swap_blocks-demo_clean-dpaligned-dp1200.zarr \
        mem_compress_method=qformer mem_compress_length=2 \
        qformer_num_layers=2 qformer_num_heads=6 qformer_cache_max_length=8 \
        diffusion_num_layers=8 batch_size=64 num_workers=4 \
        gradient_steps=100000 log_freq=100 save_freq=20000 save_on_exit=true \
        resume=true wandb_mode=offline \
        exp_name=swap_blocks/paper_l2s2_full100k \
        work_dir="$run_dir" hydra.run.dir="$run_dir" \
        >> "$log_dir/train.log" 2>&1
    [[ -s "$model" ]]
    stage train COMPLETE
else
    stage train SKIP_EXISTING
fi

stage offline_validation START
"$python" "$repo/tools/validate_rmbench_checkpoint.py" \
    --run-dir "$run_dir" --checkpoint "$model" --device cuda:0 \
    >> "$log_dir/offline_validation.log" 2>&1
stage offline_validation COMPLETE

stage online_eval START
"$repo/tools/run_rmbench_online_eval.sh" "$run_dir" "$model" \
    >> "$log_dir/online_eval.log" 2>&1
stage online_eval COMPLETE

date -Is > "$log_dir/ALL_COMPLETE"
stage all COMPLETE
