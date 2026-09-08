#!/usr/bin/env bash
set -euo pipefail

# Wait for the existing SwapBlocks parity driver to release GPU 4, then run an
# additive fixed-observation-queue diagnostic.  No existing result is replaced.
wait_pid=${1:?usage: $0 EXISTING_PARITY_PID}
rmbench=${RMBENCH_ROOT:-/home/jrw/code_rmbench}
vpwem=${VPWEM_ROOT:-/home/jrw/code_vpwem}
python=${RMBENCH_PYTHON:-/home/jrw/envs/vpwem_rmbench/bin/python}
out_root=${OUT_ROOT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
gpu=${RMBENCH_GPU:-4}
max_idle_memory_mib=${MAX_IDLE_MEMORY_MIB:-16000}

while [[ -d /proc/$wait_pid ]] && tr '\0' ' ' <"/proc/$wait_pid/cmdline" | grep -q run_rmbench_online_eval.sh; do
    sleep 30
done

wait_for_capacity() {
    local used
    while true; do
        used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        if [[ $used =~ ^[0-9]+$ ]] && (( used < max_idle_memory_mib )); then
            return
        fi
        sleep 30
    done
}

export CUDA_VISIBLE_DEVICES=$gpu
export PYTHONPATH="$vpwem/tools:$rmbench/policy:$rmbench:$rmbench/envs/curobo/src:$vpwem:$vpwem/pipelines"
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export RMBENCH_RT_DENOISER=oidn

cd "$rmbench"

run_eval() {
    local episodes=$1
    local setting="demo_clean_fixed_queue_${episodes}ep"
    local log="$out_root/logs/swap_blocks_dp_fixed_queue_${episodes}ep.log"
    wait_for_capacity
    timeout --signal=TERM --kill-after=30s 3h \
        "$python" script/eval_policy.py \
        --config policy/DP/deploy_policy.yml \
        --overrides \
        --task_name swap_blocks \
        --task_config demo_clean \
        --ckpt_setting "$setting" \
        --dp_ckpt_setting demo_clean \
        --expert_data_num 50 \
        --seed 0 \
        --policy_name DPFixedQueue \
        --ckpt_seed 0 \
        --checkpoint_num 1200 \
        --test_num "$episodes" \
        --eval_video_log false \
        2>&1 | tee "$log"
}

result_rate() {
    local episodes=$1
    local setting="demo_clean_fixed_queue_${episodes}ep"
    local result
    result=$(find "$rmbench/eval_result/swap_blocks/DPFixedQueue/demo_clean/$setting" \
        -type f -name _result.txt -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
    awk '/Success Rate:/ {print $3; exit}' "$result"
}

run_eval 1
rate=$(result_rate 1)
if awk -v rate="$rate" 'BEGIN {exit !(rate > 0)}'; then
    run_eval 3
    run_eval 10
fi
