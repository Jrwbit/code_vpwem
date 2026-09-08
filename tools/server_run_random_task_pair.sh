#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 GPU TASK ENV_TYPE QFORMER_LAYERS BASE_MODEL"
    exit 2
fi

GPU="$1"
TASK="$2"
ENV_TYPE="$3"
QFORMER_LAYERS="$4"
BASE_MODEL="$5"

case ",1,2,7," in
    *",${GPU},"*) ;;
    *) echo "refusing GPU${GPU}: server1 experiments are pinned to GPUs 1,2,7" >&2; exit 2 ;;
esac

ROOT="/home/jrw/code_vpwem"
PY="/home/jrw/anaconda3/envs/vpwem/bin/python"
DATA_ROOT="/home/jrw/vpwem_data/robomimic/datasets"
OUTPUT_ROOT="${ROOT}/artifacts/random_retention_generalization/${TASK}_${ENV_TYPE}_seed100"
EMB_DATA="${DATA_ROOT}/${TASK}/${ENV_TYPE}_emb_seed100/image_abs_emb.hdf5"
RAW_DATA="${DATA_ROOT}/${TASK}/${ENV_TYPE}/image_abs.hdf5"
CONTEXT_MODEL="/home/jrw/vpwem_outputs/robomimic/dpptp/${TASK}_${ENV_TYPE}/obs_2_act_8/100/models/model_1000000.pt"
CONFIG="${TASK}_abs_emb"
EVAL_CONFIG="${TASK}_abs_emb_eval"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=egl
export WANDB_MODE=offline
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/home/jrw/anaconda3/envs/vpwem/lib/python3.9/site-packages/mujoco_py/binaries/linux${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${OUTPUT_ROOT}/logs"
QUEUE_LOG="${OUTPUT_ROOT}/logs/queue.log"
log() { printf '[%s] [GPU%s] %s\n' "$(date '+%F %T')" "${GPU}" "$*" | tee -a "${QUEUE_LOG}"; }

for required in "${EMB_DATA}" "${RAW_DATA}" "${BASE_MODEL}" "${CONTEXT_MODEL}"; do
    if [[ ! -s "${required}" ]]; then
        log "FAIL missing ${required}"
        exit 2
    fi
done

train_method() {
    local method="$1"
    local run_dir="${OUTPUT_ROOT}/${method}_train_10k"
    local checkpoint="${run_dir}/models/model_10000.pt"
    [[ -s "${checkpoint}" ]] && { log "SKIP ${method} train"; return 0; }
    log "START ${method} train"
    "${PY}" "${ROOT}/pipelines/longmemdpptp_robomimic_image_emb.py" \
        --config-name="${CONFIG}" env_type="${ENV_TYPE}" dataset_path="${EMB_DATA}" \
        work_dir="${run_dir}" save_path="${ROOT}/artifacts/random_retention_generalization" \
        gradient_steps=10000 save_freq=10000 log_freq=500 diffusion_num_layers=4 \
        mem_compress_method="${method}" mem_compress_length=2 \
        qformer_num_layers="${QFORMER_LAYERS}" qformer_cache_max_length=8 \
        qformer_cache_compress=fifo dataset_memory_subsample_ratio=20 \
        lr=0.00001 seed=100 wandb_mode=offline \
        +num_workers=2 +dataset_maximum_memory_length=16 +inference_action_steps=8 \
        +memory_aux_coeff=0.0 +counterfactual_memory_coeff=0.0 \
        +counterfactual_memory_margin=0.01 +counterfactual_memory_probability=0.0 \
        +init_model_path="${BASE_MODEL}" +keep_checkpoints=1 \
        >"${OUTPUT_ROOT}/logs/${method}_train.log" 2>&1
    local rc=$?
    [[ ${rc} -eq 0 && -s "${checkpoint}" ]] && { log "DONE ${method} train"; return 0; }
    log "FAIL ${method} train rc=${rc}"
    return 1
}

eval_method() {
    local method="$1"
    local checkpoint="${OUTPUT_ROOT}/${method}_train_10k/models/model_10000.pt"
    local run_dir="${OUTPUT_ROOT}/${method}_eval_100"
    local result="${run_dir}/result_10000_memory-none.npy"
    [[ -s "${result}" ]] && { log "SKIP ${method} eval"; return 0; }
    [[ ! -s "${checkpoint}" ]] && { log "FAIL ${method} eval: checkpoint missing"; return 1; }
    log "START ${method} eval"
    "${PY}" "${ROOT}/pipelines/longmemdpptp_robomimic_image_emb_eval.py" \
        --config-name="${EVAL_CONFIG}" env_type="${ENV_TYPE}" dataset_path="${RAW_DATA}" \
        work_dir="${run_dir}" model_path="${checkpoint}" +ckpt=10000 \
        diffusion_num_layers=4 mem_compress_method="${method}" mem_compress_length=2 \
        qformer_num_layers="${QFORMER_LAYERS}" qformer_cache_max_length=8 \
        qformer_cache_compress=fifo dataset_memory_subsample_ratio=20 \
        lr=0.00001 seed=20260814 num_envs=10 eval_episodes=100 \
        save_video=false wandb_mode=offline +inference_action_steps=8 \
        +dataset_maximum_memory_length=16 +eval_normalizer_only=true \
        '+memory_interventions=[none]' +context_model_path="${CONTEXT_MODEL}" \
        >"${OUTPUT_ROOT}/logs/${method}_eval.log" 2>&1
    local rc=$?
    [[ ${rc} -eq 0 && -s "${result}" ]] && { log "DONE ${method} eval"; return 0; }
    log "FAIL ${method} eval rc=${rc}"
    return 1
}

log "QUEUE BEGIN ${TASK}/${ENV_TYPE}: FIFO vs Random, seed100"
train_method qformer & fifo_pid=$!
train_method random & random_pid=$!
failed=0
wait "${fifo_pid}" || failed=1
wait "${random_pid}" || failed=1

eval_method qformer & fifo_pid=$!
eval_method random & random_pid=$!
wait "${fifo_pid}" || failed=1
wait "${random_pid}" || failed=1

if [[ ${failed} -eq 0 ]]; then
    log "QUEUE COMPLETE"
    exit 0
fi
log "QUEUE FINISHED WITH FAILURES; rerun to resume"
exit 1
