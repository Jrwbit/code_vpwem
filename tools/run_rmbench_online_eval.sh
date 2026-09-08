#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 RUN_DIR MODEL_PATH" >&2
    exit 2
fi

run_dir=$1
model=$2
rmbench=${RMBENCH_ROOT:-/home/wjr/code/rmbench}
vpwem=${VPWEM_ROOT:-/home/wjr/code/vpwem}
python=${RMBENCH_PYTHON:-/home/wjr/miniconda3/envs/vpwem_rmbench/bin/python}
gpu=${RMBENCH_GPU:-0}
dp_ckpt=$rmbench/policy/DP/checkpoints/swap_blocks-demo_clean-50-0/1200.ckpt

[[ -x "$python" ]]
[[ -s "$model" ]]
[[ -s "$run_dir/.hydra/config.yaml" ]]
[[ -s "$run_dir/models/rmbench_swap_blocks_normalizer.pt" ]]
[[ -s "$dp_ckpt" ]]

export CUDA_VISIBLE_DEVICES=$gpu
export PYTHONPATH=$vpwem/tools:$rmbench/policy:$rmbench:$rmbench/envs/curobo/src:$vpwem:$vpwem/pipelines
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
# Official RMBench demonstrations are rendered with OIDN.  Pin the evaluator
# explicitly so a local renderer default cannot silently invalidate parity.
export RMBENCH_RT_DENOISER=oidn

# A rollout normally completes well within this limit. Prevent renderer or
# multiprocessing failures from holding the GPU indefinitely.
eval_timeout=${RMBENCH_EVAL_TIMEOUT:-2h}

cd "$rmbench"

"$python" script/probe_dummy_policy.py \
    --task_name swap_blocks \
    --task_config demo_clean \
    --seed 100000 \
    --steps 1

# Parity-corrected official DP is the deployment gate and therefore runs first.
# A DP failure must not be hidden behind a later VPWEM-specific loading error.
timeout --signal=TERM --kill-after=30s "$eval_timeout" \
"$python" script/eval_policy.py \
    --config policy/DP/deploy_policy.yml \
    --overrides \
    --task_name swap_blocks \
    --task_config demo_clean \
    --ckpt_setting demo_clean_corrected_oidn_10ep \
    --dp_ckpt_setting demo_clean \
    --expert_data_num 50 \
    --seed 0 \
    --policy_name DPCorrected \
    --ckpt_seed 0 \
    --checkpoint_num 1200 \
    --test_num 10 \
    --eval_video_log false

timeout --signal=TERM --kill-after=30s "$eval_timeout" \
"$python" script/eval_policy.py \
    --config policy/VPWEM/deploy_policy.yml \
    --overrides \
    --task_name swap_blocks \
    --task_config demo_clean \
    --ckpt_setting paper_l2s2_full100k_10ep \
    --seed 0 \
    --policy_name VPWEMCorrected \
    --model_path "$model" \
    --dry_run false \
    --config_path "$run_dir/.hydra/config.yaml" \
    --normalizer_path "$run_dir/models/rmbench_swap_blocks_normalizer.pt" \
    --dp_ckpt_path "$dp_ckpt" \
    --test_num 10 \
    --eval_video_log false \
    --debug_print_once true
