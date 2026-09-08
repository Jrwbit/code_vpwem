#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 GPU TASK ENV_TYPE QFORMER_LAYERS BASE_MODEL" >&2
    exit 2
fi

GPU="$1"; TASK="$2"; ENV_TYPE="$3"; QFORMER_LAYERS="$4"; BASE_MODEL="$5"
case ",1,2,7," in
    *",${GPU},"*) ;;
    *) echo "refusing GPU${GPU}: pinned set is 1,2,7" >&2; exit 2 ;;
esac

ROOT="/home/jrw/code_vpwem"
PY="/home/jrw/anaconda3/envs/vpwem/bin/python"
DATA_ROOT="/home/jrw/vpwem_data/robomimic/datasets"
OUT="${ROOT}/artifacts/random_retention_generalization/${TASK}_${ENV_TYPE}_seed100"
EMB="${DATA_ROOT}/${TASK}/${ENV_TYPE}_emb_seed100/image_abs_emb.hdf5"
RAW="${DATA_ROOT}/${TASK}/${ENV_TYPE}/image_abs.hdf5"
CONTEXT="/home/jrw/vpwem_outputs/robomimic/dpptp/${TASK}_${ENV_TYPE}/obs_2_act_8/100/models/model_1000000.pt"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl WANDB_MODE=offline
export GPU_POLICY_ALLOWED_GPUS=1,2,7 GPU_POLICY_START_OWNED=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/home/jrw/anaconda3/envs/vpwem/lib/python3.9/site-packages/mujoco_py/binaries/linux${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${OUT}/logs"
log() { printf '[%s] [GPU%s] %s\n' "$(date '+%F %T')" "${GPU}" "$*" | tee -a "${OUT}/logs/raw_methods_queue.log"; }

train_one() {
    local method="$1" dir="${OUT}/${1}_train_10k"
    [[ -s "${dir}/models/model_10000.pt" ]] && { log "SKIP ${method} train"; return 0; }
    log "START ${method} train"
    "${PY}" "${ROOT}/pipelines/longmemdpptp_robomimic_image_emb.py" \
        --config-name="${TASK}_abs_emb" env_type="${ENV_TYPE}" dataset_path="${EMB}" \
        work_dir="${dir}" save_path="${ROOT}/artifacts/random_retention_generalization" \
        gradient_steps=10000 save_freq=10000 log_freq=500 diffusion_num_layers=4 \
        mem_compress_method="${method}" mem_compress_length=2 \
        qformer_num_layers="${QFORMER_LAYERS}" qformer_cache_max_length=8 \
        qformer_cache_compress=fifo dataset_memory_subsample_ratio=20 \
        lr=0.00001 seed=100 wandb_mode=offline +num_workers=2 \
        +dataset_maximum_memory_length=16 +inference_action_steps=8 \
        +memory_aux_coeff=0.0 +counterfactual_memory_coeff=0.0 \
        +counterfactual_memory_margin=0.01 +counterfactual_memory_probability=0.0 \
        +init_model_path="${BASE_MODEL}" +keep_checkpoints=1 \
        >"${OUT}/logs/${method}_train.log" 2>&1
    local rc=$?
    [[ ${rc} -eq 0 && -s "${dir}/models/model_10000.pt" ]] && { log "DONE ${method} train"; return 0; }
    log "FAIL ${method} train rc=${rc}"
    return 1
}

eval_one() {
    local method="$1" checkpoint="${OUT}/${1}_train_10k/models/model_10000.pt"
    local dir="${OUT}/${1}_eval_100" result="${OUT}/${1}_eval_100/result_10000_memory-none.npy"
    [[ -s "${result}" ]] && { log "SKIP ${method} eval"; return 0; }
    log "START ${method} eval"
    "${PY}" "${ROOT}/pipelines/longmemdpptp_robomimic_image_emb_eval.py" \
        --config-name="${TASK}_abs_emb_eval" env_type="${ENV_TYPE}" dataset_path="${RAW}" \
        work_dir="${dir}" model_path="${checkpoint}" +ckpt=10000 \
        diffusion_num_layers=4 mem_compress_method="${method}" mem_compress_length=2 \
        qformer_num_layers="${QFORMER_LAYERS}" qformer_cache_max_length=8 \
        qformer_cache_compress=fifo dataset_memory_subsample_ratio=20 \
        lr=0.00001 seed=20260814 num_envs=10 eval_episodes=100 save_video=false \
        wandb_mode=offline +inference_action_steps=8 +dataset_maximum_memory_length=16 \
        +eval_normalizer_only=true '+memory_interventions=[none]' \
        +context_model_path="${CONTEXT}" >"${OUT}/logs/${method}_eval.log" 2>&1
    local rc=$?
    [[ ${rc} -eq 0 && -s "${result}" ]] && { log "DONE ${method} eval"; return 0; }
    log "FAIL ${method} eval rc=${rc}"
    return 1
}

log "APPENDED QUEUE BEGIN ${TASK}/${ENV_TYPE}: Recent-2 and First+Recent-2"
train_one recent & p1=$!
train_one anchor_recent & p2=$!
failed=0
wait "${p1}" || failed=1
wait "${p2}" || failed=1

# The original FIFO/Reservoir pair keeps priority for rollout. This queue is
# already pinned to the same card, so it waits for that process—not GPU idleness.
while pgrep -f "server_run_random_task_pair.sh ${GPU} ${TASK} ${ENV_TYPE} " >/dev/null; do
    log "WAIT original paired queue process; GPU occupancy is not re-evaluated"
    sleep 30
done

eval_one recent & p1=$!
eval_one anchor_recent & p2=$!
wait "${p1}" || failed=1
wait "${p2}" || failed=1

[[ ${failed} -eq 0 ]] && { log "APPENDED QUEUE COMPLETE"; exit 0; }
log "APPENDED QUEUE FINISHED WITH FAILURES; rerun to resume"
exit 1
