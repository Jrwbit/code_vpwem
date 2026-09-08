#!/usr/bin/env bash
set -euo pipefail

# Formal SwapBlocks parity-corrected OOD evaluation.  Each episode is a fresh
# process with an independent RMBench result directory, so completed episodes
# survive a renderer crash, timeout, or supervisor restart.
rmbench=${RMBENCH_ROOT:-/home/jrw/code_rmbench}
vpwem=${VPWEM_ROOT:-/home/jrw/code_vpwem}
python=${RMBENCH_PYTHON:-/home/jrw/envs/vpwem_rmbench/bin/python}
out_root=${OUT_ROOT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
gpu=${RMBENCH_GPU:-4}
max_idle_memory_mib=${MAX_IDLE_MEMORY_MIB:-12000}
episode_timeout=${RMBENCH_EPISODE_TIMEOUT:-20m}
dp_episodes=${DP_EPISODES:-10}
vpwem_episodes=${VPWEM_EPISODES:-3}
first_candidate_seed=${RMBENCH_OOD_START_SEED:-100000}

vpwem_run_dir=${VPWEM_RUN_DIR:-$vpwem/outputs/rmbench/longmemdpptp_rmbench_image_emb/train/swap_blocks/paper_l2s2_full100k/100}
vpwem_model=${VPWEM_MODEL_PATH:-$vpwem_run_dir/models/model_100001.pt}
vpwem_normalizer=${VPWEM_NORMALIZER_PATH:-$vpwem_run_dir/models/rmbench_swap_blocks_normalizer.pt}
dp_ckpt=$rmbench/policy/DP/checkpoints/swap_blocks-demo_clean-50-0/1200.ckpt

run_root=$out_root/swap_blocks/corrected_ood10
log_root=$run_root/logs
state=$run_root/episodes.tsv
supervisor_log=$run_root/supervisor.log
mkdir -p "$log_root"

exec 9>"$run_root/run.lock"
if ! flock -n 9; then
    echo "corrected SwapBlocks OOD supervisor is already running"
    exit 0
fi
exec > >(tee -a "$supervisor_log") 2>&1

if [[ $gpu != 4 ]]; then
    echo "This formal run is pinned to GPU4; got RMBENCH_GPU=$gpu" >&2
    exit 2
fi
if [[ $gpu == 0 || $gpu == 1 || $gpu == 7 ]]; then
    echo "Refusing disallowed GPU $gpu" >&2
    exit 2
fi
if ! [[ $dp_episodes =~ ^[1-9][0-9]*$ && $vpwem_episodes =~ ^[0-9]+$ ]]; then
    echo "DP_EPISODES and VPWEM_EPISODES must be non-negative integers" >&2
    exit 2
fi

for required in \
    "$python" \
    "$dp_ckpt" \
    "$vpwem/tools/DPCorrected.py" \
    "$vpwem/tools/DPFixedQueue.py" \
    "$vpwem/tools/VPWEMCorrected.py"; do
    [[ -e $required ]] || { echo "Missing required path: $required" >&2; exit 2; }
done
[[ -x $python ]] || { echo "Python is not executable: $python" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="$vpwem/tools:$rmbench/policy:$rmbench:$rmbench/envs/curobo/src:$vpwem:$vpwem/pipelines"
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
# This is deliberately pinned, not caller-overridable: official RMBench data
# collection uses OIDN and the corrected protocol must not silently drift.
export RMBENCH_RT_DENOISER=oidn

timestamp() { date -Is; }

wait_for_capacity() {
    local used
    while true; do
        used=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        if [[ $used =~ ^[0-9]+$ ]] && (( used < max_idle_memory_mib )); then
            echo "$(timestamp) GPU4 guard pass: ${used}MiB < ${max_idle_memory_mib}MiB"
            return
        fi
        echo "$(timestamp) GPU4 guard waiting: ${used:-unknown}MiB >= ${max_idle_memory_mib}MiB"
        sleep 30
    done
}

latest_result_dir() {
    local policy_name=$1
    local setting=$2
    find "$rmbench/eval_result/swap_blocks/$policy_name/demo_clean/$setting" \
        -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -n 1 | cut -d' ' -f2-
}

state_value() {
    local phase=$1
    local episode=$2
    local column=$3
    awk -F '\t' -v phase="$phase" -v episode="$episode" -v column="$column" \
        '$1 == phase && $2 == episode {value=$column} END {if (value != "") print value}' "$state"
}

append_result() {
    local phase=$1 episode=$2 requested=$3 actual=$4 result=$5 success=$6 reward=$7 setting=$8 result_dir=$9 log=${10}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$phase" "$episode" "$requested" "$actual" "$result" "$success" "$reward" \
        "$setting" "$result_dir" "$log" "$(timestamp)" >> "$state"
}

run_one() {
    local phase=$1 episode=$2 requested_seed=$3 policy_name=$4 setting=$5 log=$6
    shift 6
    wait_for_capacity
    echo "$(timestamp) START phase=$phase episode=$episode requested_seed=$requested_seed policy=$policy_name"
    set +e
    timeout --signal=TERM --kill-after=30s "$episode_timeout" \
        "$python" script/eval_policy.py "$@" 2>&1 | tee "$log"
    local command_status=${PIPESTATUS[0]}
    set -e
    if (( command_status != 0 )); then
        append_result "$phase" "$episode" "$requested_seed" -1 "ERROR_$command_status" 0 0 \
            "$setting" - "$log"
        echo "$(timestamp) episode failed with status $command_status; completed episodes remain in $state" >&2
        exit "$command_status"
    fi

    local result_dir eval_log result_file actual_seed result success reward
    result_dir=$(latest_result_dir "$policy_name" "$setting")
    [[ -n $result_dir ]] || { echo "No result directory for $policy_name/$setting" >&2; exit 3; }
    eval_log=$result_dir/eval_log.txt
    result_file=$result_dir/_result.txt
    [[ -s $eval_log && -s $result_file ]] || { echo "Incomplete result directory: $result_dir" >&2; exit 3; }
    actual_seed=$(sed -n 's/.*seed=\([0-9][0-9]*\), result=.*/\1/p' "$eval_log" | tail -n 1)
    result=$(sed -n 's/.*result=\([^,]*\),.*/\1/p' "$eval_log" | tail -n 1)
    success=$(awk '/Success Rate:/ {print ($3 > 0 ? 1 : 0); exit}' "$result_file")
    reward=$(awk '/Reward:/ {print $2; exit}' "$result_file")
    [[ $actual_seed =~ ^[0-9]+$ && $success =~ ^[01]$ ]] || {
        echo "Could not parse persisted result: $result_dir" >&2
        exit 3
    }
    append_result "$phase" "$episode" "$requested_seed" "$actual_seed" "$result" "$success" \
        "$reward" "$setting" "$result_dir" "$log"
    echo "$(timestamp) COMPLETE phase=$phase episode=$episode actual_seed=$actual_seed success=$success reward=$reward"
}

cd "$rmbench"

# Fast static/behavioral parity gate before loading the renderer or checkpoint.
"$python" -m unittest discover -s "$vpwem/tests" -p test_rmbench_corrected_dp.py -v
grep -q 'RMBENCH_RT_DENOISER' "$rmbench/envs/_base_task.py"

if [[ ! -s $state ]]; then
    printf 'phase\tepisode\trequested_seed\tactual_seed\tresult\tsuccess\treward\tsetting\tresult_dir\tlog\tfinished_at\n' > "$state"
fi

candidate_seed=$first_candidate_seed
for ((episode=0; episode<dp_episodes; episode++)); do
    completed_seed=$(state_value dp "$episode" 4)
    if [[ $completed_seed =~ ^[0-9]+$ ]]; then
        echo "$(timestamp) RESUME skip dp episode=$episode actual_seed=$completed_seed"
        candidate_seed=$((completed_seed + 1))
        continue
    fi
    setting=$(printf 'demo_clean_corrected_oidn_ood_dp_ep%02d_from%d' "$episode" "$candidate_seed")
    log=$(printf '%s/dp_ep%02d_from%d.log' "$log_root" "$episode" "$candidate_seed")
    run_one dp "$episode" "$candidate_seed" DPCorrected "$setting" "$log" \
        --config policy/DP/deploy_policy.yml \
        --overrides \
        --task_name swap_blocks \
        --task_config demo_clean \
        --ckpt_setting "$setting" \
        --dp_ckpt_setting demo_clean \
        --expert_data_num 50 \
        --seed 0 \
        --eval_start_seed "$candidate_seed" \
        --policy_name DPCorrected \
        --ckpt_seed 0 \
        --checkpoint_num 1200 \
        --test_num 1 \
        --expert_check true \
        --eval_video_log false
    candidate_seed=$(( $(state_value dp "$episode" 4) + 1 ))
done

dp_successes=$(awk -F '\t' '$1 == "dp" {sum += $6} END {print sum+0}' "$state")
dp_completed=$(awk -F '\t' '$1 == "dp" {count++} END {print count+0}' "$state")
echo "$(timestamp) corrected DP complete: successes=$dp_successes/$dp_completed"

if (( dp_successes == 0 )); then
    printf '%s\tcorrected_dp_successes=0/%s\tcoverage_or_checkpoint_generalization_blocked\n' \
        "$(timestamp)" "$dp_completed" > "$run_root/COVERAGE_PARITY_BLOCKED.tsv"
    echo "No corrected-DP OOD recovery; VPWEM comparison intentionally not launched."
    exit 0
fi

for required in "$vpwem_model" "$vpwem_run_dir/.hydra/config.yaml" "$vpwem_normalizer"; do
    [[ -s $required ]] || { echo "Corrected DP recovered, but VPWEM artifact is missing: $required" >&2; exit 4; }
done

mapfile -t comparison_seeds < <(awk -F '\t' '$1 == "dp" {print $4}' "$state" | head -n "$vpwem_episodes")
for ((episode=0; episode<${#comparison_seeds[@]}; episode++)); do
    seed=${comparison_seeds[$episode]}
    if [[ $(state_value vpwem "$episode" 4) == "$seed" ]]; then
        echo "$(timestamp) RESUME skip vpwem episode=$episode seed=$seed"
        continue
    fi
    setting=$(printf 'paper_l2s2_full100k_corrected_oidn_ood_ep%02d_seed%d' "$episode" "$seed")
    log=$(printf '%s/vpwem_ep%02d_seed%d.log' "$log_root" "$episode" "$seed")
    run_one vpwem "$episode" "$seed" VPWEMCorrected "$setting" "$log" \
        --config policy/VPWEM/deploy_policy.yml \
        --overrides \
        --task_name swap_blocks \
        --task_config demo_clean \
        --ckpt_setting "$setting" \
        --seed 0 \
        --eval_start_seed "$seed" \
        --policy_name VPWEMCorrected \
        --model_path "$vpwem_model" \
        --dry_run false \
        --config_path "$vpwem_run_dir/.hydra/config.yaml" \
        --normalizer_path "$vpwem_normalizer" \
        --dp_ckpt_path "$dp_ckpt" \
        --test_num 1 \
        --expert_check false \
        --eval_video_log false \
        --execute_horizon 1 \
        --debug_print_chunks 3
done

vpwem_successes=$(awk -F '\t' '$1 == "vpwem" {sum += $6} END {print sum+0}' "$state")
vpwem_completed=$(awk -F '\t' '$1 == "vpwem" {count++} END {print count+0}' "$state")
printf '%s\tdp=%s/%s\tvpwem=%s/%s\n' "$(timestamp)" "$dp_successes" "$dp_completed" \
    "$vpwem_successes" "$vpwem_completed" > "$run_root/COMPLETE.tsv"
echo "$(timestamp) ALL COMPLETE dp=$dp_successes/$dp_completed vpwem=$vpwem_successes/$vpwem_completed"
