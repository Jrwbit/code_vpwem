#!/usr/bin/env bash
set -uo pipefail

home=/home/jrw
repo=$home/code_vpwem
python=$home/anaconda3/envs/vpwem/bin/python
output_root=$home/vpwem_outputs
gpu=${VPWEM_EVAL_GPU:-4}
shard_index=${VPWEM_EVAL_SHARD_INDEX:-0}
shard_count=${VPWEM_EVAL_SHARD_COUNT:-1}
minimum_free_mib=${VPWEM_EVAL_MIN_FREE_MIB:-18000}
gpu_turn_owned=0

log_root=$home/vpwem_logs/checkpoint_curve_gpu${gpu}
mkdir -p "$log_root/jobs" "$output_root/checkpoint_curves"
exec 9>"$home/.vpwem-checkpoint-curve-gpu${gpu}.lock"
if ! flock -n 9; then
    echo "$(date -Is) checkpoint-curve evaluator is already running" >&2
    exit 0
fi

exec > >(tee -a "$log_root/supervisor.log") 2>&1
export HOME=$home
export XDG_CACHE_HOME=$home/.cache
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$gpu
export GIBSON_DEVICE_ID=$gpu
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

append_status() {
    printf "%s\t%s\t%s\t%s\n" "$(date -Is)" "$1" "$2" "$3" \
        >> "$log_root/status.tsv"
}

refresh_summary() {
    "$python" "$repo/tools/summarize_checkpoint_curve.py" \
        --output-root "$output_root" \
        --csv "$output_root/checkpoint_curves/checkpoint_scores.csv"
}

wait_for_training_queue() {
    [[ "$gpu" == 4 ]] || return 0
    while tmux has-session -t vpwem_mikasa_8layer_gpu4 2>/dev/null; do
        echo "$(date -Is) waiting for MIKASA GPU4 training/final-eval queue"
        sleep 60
    done
}

wait_for_gpu_capacity() {
    local free util
    if (( gpu_turn_owned )); then
        IFS=, read -r free util < <(
            nvidia-smi --id="$gpu" \
                --query-gpu=memory.free,utilization.gpu \
                --format=csv,noheader,nounits
        )
        free=${free//[[:space:]]/}
        util=${util//[[:space:]]/}
        echo "$(date -Is) GPU$gpu queue owns turn; continuing regardless of later sharing: free=${free}MiB util=${util}%"
        return 0
    fi
    while true; do
        IFS=, read -r free util < <(
            nvidia-smi --id="$gpu" \
                --query-gpu=memory.free,utilization.gpu \
                --format=csv,noheader,nounits
        )
        free=${free//[[:space:]]/}
        util=${util//[[:space:]]/}
        if (( free >= minimum_free_mib )); then
            echo "$(date -Is) GPU$gpu capacity available: free=${free}MiB util=${util}%"
            gpu_turn_owned=1
            return 0
        else
            echo "$(date -Is) GPU$gpu below reserve; waiting: free=${free}MiB util=${util}%"
        fi
        sleep 60
    done
}

task_index=0
claim_shard() {
    local selected=1
    if (( task_index % shard_count == shard_index )); then
        selected=0
    fi
    task_index=$((task_index + 1))
    return "$selected"
}

eval_mikasa_checkpoint() {
    local config=$1 seed=$2 checkpoint=$3
    local env_name save_root dataset work_dir result job log
    case "$config" in
        shell_game_touch)
            env_name=ShellGameTouch-v0
            save_root=$output_root/shell_game_touch
            ;;
        remember_color_3)
            env_name=RememberColor3-v0
            save_root=$output_root/remember_color_3
            ;;
        *)
            return 2
            ;;
    esac
    dataset=$home/vpwem_data/mikasa/$env_name
    work_dir=$save_root/ours/mikasa/$env_name/obs_2_act_8/8layer_5_qformer_2_sin_pos_2_fifo_8_0.0001/$seed
    result=$work_dir/result_${checkpoint}_8.npy
    job=${config}_seed${seed}
    log=$log_root/jobs/${job}_ckpt${checkpoint}.log

    [[ -s "$work_dir/models/model_${checkpoint}.pt" ]] || return 0
    if [[ -s "$result" ]]; then
        append_status "$job" "$checkpoint" SKIP
        return 0
    fi
    wait_for_gpu_capacity
    append_status "$job" "$checkpoint" START
    if "$python" "$repo/pipelines/longmemdpptp_mikasa_image_eval.py" \
        --config-name="$config" \
        dataset_path="$dataset" save_path="$save_root" \
        seed="$seed" gradient_steps=600000 \
        +inference_action_steps=8 +ckpt="$checkpoint" \
        num_envs=10 eval_episodes=30 save_video=false \
        diffusion_num_layers=8 device=cuda:0 \
        mem_compress_length=2 memory_pos_emb=sin_pos \
        qformer_num_layers=2 qformer_cache_compress=fifo \
        qformer_cache_max_length=8 dataset_memory_subsample_ratio=5 \
        work_dir="$work_dir" hydra.run.dir="$work_dir" \
        >> "$log" 2>&1 && [[ -s "$result" ]]; then
        append_status "$job" "$checkpoint" COMPLETE
        refresh_summary
    else
        status=$?
        append_status "$job" "$checkpoint" "FAILED:$status"
    fi
}

eval_momart_checkpoint() {
    local work_dir=$1 checkpoint=$2
    local seed name env_type env_name raw_data context_model result job log
    seed=${work_dir##*/}
    name=$(basename "$(dirname "$(dirname "$(dirname "$work_dir")")")")
    if [[ "$name" == *_suboptimal ]]; then
        env_type=suboptimal
    else
        env_type=expert
    fi
    env_name=${name%_$env_type}
    raw_data=$home/vpwem_data/momart/${name}.hdf5
    context_model=$output_root/momart/dpptp/$name/obs_2_act_8_8layer/100/models/model_1000000.pt
    result=$work_dir/result_ckpt${checkpoint}_8.npy
    job=${name}_seed${seed}
    log=$log_root/jobs/${job}_ckpt${checkpoint}.log

    [[ -s "$work_dir/models/model_${checkpoint}.pt" ]] || return 0
    if [[ -s "$result" ]]; then
        append_status "$job" "$checkpoint" SKIP
        return 0
    fi
    wait_for_gpu_capacity
    append_status "$job" "$checkpoint" START
    if "$python" "$repo/pipelines/longmemdpptp_momart_image_emb_eval.py" \
        --config-name=momart \
        dataset_path="$raw_data" save_path="$output_root" \
        env_name="$env_name" env_type="$env_type" seed="$seed" \
        obs_steps=2 action_steps=8 horizon=9 \
        num_envs=10 eval_episodes=30 save_video=false \
        diffusion_num_layers=8 device=cuda:0 \
        mem_compress_method=qformer mem_compress_length=2 \
        dataset_memory_type=random_from_full memory_pos_emb=sin_pos \
        qformer_num_layers=2 qformer_cache_compress=fifo \
        qformer_cache_max_length=8 \
        short_cond_dropout=0.3 long_cond_dropout=0.3 \
        dataset_memory_subsample_ratio=20 \
        +condition_model_path="$context_model" +ckpt="$checkpoint" \
        +inference_action_steps=8 \
        work_dir="$work_dir" hydra.run.dir="$work_dir" \
        >> "$log" 2>&1 && [[ -s "$result" ]]; then
        append_status "$job" "$checkpoint" COMPLETE
        refresh_summary
    else
        status=$?
        append_status "$job" "$checkpoint" "FAILED:$status"
    fi
}

cd "$repo" || exit 1
wait_for_training_queue
echo "$(date -Is) worker GPU$gpu shard=$shard_index/$shard_count min_free=${minimum_free_mib}MiB"

for config in shell_game_touch remember_color_3; do
    for seed in 100 200 300; do
        case "$config" in
            shell_game_touch)
                mikasa_env=ShellGameTouch-v0
                mikasa_root=$output_root/shell_game_touch
                ;;
            remember_color_3)
                mikasa_env=RememberColor3-v0
                mikasa_root=$output_root/remember_color_3
                ;;
        esac
        mikasa_work_dir=$mikasa_root/ours/mikasa/$mikasa_env/obs_2_act_8/8layer_5_qformer_2_sin_pos_2_fifo_8_0.0001/$seed
        [[ -s "$mikasa_work_dir/models/model_600000.pt" ]] || continue
        for checkpoint in $(seq 20000 20000 600000); do
            if claim_shard; then
                eval_mikasa_checkpoint "$config" "$seed" "$checkpoint"
            fi
        done
    done
done

# These are the six runs that were already fully trained when this queue was
# created. Keeping a fixed manifest prevents racing final evaluation from active
# training pools that happen to finish while this queue is running.
momart_signature=obs_2_act_8/8layer_20_qformer_2_sin_pos_2_fifo_8
completed_momart_runs=(
    "$output_root/ours/momart_emb/table_setup_from_dresser_suboptimal/$momart_signature/100"
    "$output_root/ours/momart_emb/unload_dishwasher_to_dresser_expert/$momart_signature/100"
    "$output_root/ours/momart_emb/unload_dishwasher_to_dresser_expert/$momart_signature/200"
    "$output_root/ours/momart_emb/unload_dishwasher_to_dresser_expert/$momart_signature/300"
    "$output_root/ours/momart_emb/unload_dishwasher_to_dresser_suboptimal/$momart_signature/100"
    "$output_root/ours/momart_emb/unload_dishwasher_to_dresser_suboptimal/$momart_signature/200"
)
for work_dir in "${completed_momart_runs[@]}"; do
    for checkpoint in $(seq 100000 100000 1000000); do
        if claim_shard; then
            eval_momart_checkpoint "$work_dir" "$checkpoint"
        fi
    done
done

refresh_summary
date -Is > "$log_root/ALL_COMPLETE"
echo "$(date -Is) checkpoint-curve evaluation queue complete"
