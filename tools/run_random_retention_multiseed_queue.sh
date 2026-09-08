#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/wjr/code/vpwem"
PYTHON_BIN="/home/wjr/miniconda3/envs/vpwem5090/bin/python"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/random_retention_multiseed"
RAW_DATASET="${REPO_ROOT}/data/robomimic/datasets/transport/mh/image_abs.hdf5"
EVAL_SEED=20260814
SEEDS=(200 300)

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/home/wjr/miniconda3/envs/vpwem5090/lib/python3.9/site-packages/mujoco_py/binaries/linux${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export WANDB_MODE=offline

mkdir -p "${ARTIFACT_ROOT}/logs"
QUEUE_LOG="${ARTIFACT_ROOT}/logs/queue.log"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${QUEUE_LOG}"
}

train_random() {
    local train_seed="$1"
    local run_dir="${ARTIFACT_ROOT}/seed${train_seed}/random_train_10k"
    local checkpoint="${run_dir}/models/model_10000.pt"
    local dataset="${REPO_ROOT}/data/robomimic/datasets/transport/mh_emb_seed${train_seed}/image_abs_emb.hdf5"
    local base_model="${REPO_ROOT}/outputs/ours/robomimic_emb/transport_mh/vpwem_reproduction/${train_seed}/models/model_1000000.pt"
    local run_log="${ARTIFACT_ROOT}/logs/random_train_seed${train_seed}.log"

    if [[ -s "${checkpoint}" ]]; then
        log "SKIP random train seed${train_seed}: checkpoint exists"
        return 0
    fi
    if [[ ! -s "${dataset}" || ! -s "${base_model}" ]]; then
        log "FAIL random train seed${train_seed}: dataset or base checkpoint missing"
        return 2
    fi

    mkdir -p "${run_dir}"
    log "START random train seed${train_seed} (10k)"
    "${PYTHON_BIN}" "${REPO_ROOT}/pipelines/longmemdpptp_robomimic_image_emb.py" \
        --config-name=transport_abs_emb \
        env_type=mh \
        dataset_path="${dataset}" \
        save_path="${REPO_ROOT}/artifacts/per_vpwem" \
        work_dir="${run_dir}" \
        gradient_steps=10000 save_freq=10000 log_freq=500 \
        diffusion_num_layers=4 \
        mem_compress_method=random mem_compress_length=2 \
        qformer_num_layers=2 qformer_cache_max_length=8 qformer_cache_compress=fifo \
        dataset_memory_subsample_ratio=20 lr=0.00001 seed="${train_seed}" wandb_mode=offline \
        +num_workers=2 +dataset_maximum_memory_length=16 +inference_action_steps=8 \
        +memory_aux_coeff=0.0 +counterfactual_memory_coeff=0.0 \
        +counterfactual_memory_margin=0.01 +counterfactual_memory_probability=0.0 \
        +init_model_path="${base_model}" +keep_checkpoints=1 \
        >"${run_log}" 2>&1
    local rc=$?
    if [[ ${rc} -eq 0 && -s "${checkpoint}" ]]; then
        log "DONE random train seed${train_seed}"
        return 0
    fi
    log "FAIL random train seed${train_seed}: rc=${rc}; see ${run_log}"
    return "${rc}"
}

eval_model() {
    local train_seed="$1"
    local method="$2"
    local run_dir model_path mem_method
    local context_model="${REPO_ROOT}/outputs/robomimic/dpptp/transport_mh/obs_2_act_8/${train_seed}/models/model_1000000.pt"
    if [[ "${method}" == "random" ]]; then
        run_dir="${ARTIFACT_ROOT}/seed${train_seed}/random_eval_100"
        model_path="${ARTIFACT_ROOT}/seed${train_seed}/random_train_10k/models/model_10000.pt"
        mem_method="random"
    else
        run_dir="${ARTIFACT_ROOT}/seed${train_seed}/fifo_eval_100"
        model_path="${REPO_ROOT}/artifacts/per_vpwem/finetune/transport_mh_seed${train_seed}_qformer_control_10k/models/model_10000.pt"
        mem_method="qformer"
    fi
    local result="${run_dir}/result_10000_memory-none.npy"
    local run_log="${ARTIFACT_ROOT}/logs/${method}_eval_seed${train_seed}.log"

    if [[ -s "${result}" ]]; then
        log "SKIP ${method} eval seed${train_seed}: result exists"
        return 0
    fi
    if [[ ! -s "${model_path}" || ! -s "${context_model}" ]]; then
        log "FAIL ${method} eval seed${train_seed}: policy or context checkpoint missing"
        return 2
    fi

    mkdir -p "${run_dir}"
    log "START ${method} eval seed${train_seed} (100 episodes, eval seed ${EVAL_SEED})"
    "${PYTHON_BIN}" "${REPO_ROOT}/pipelines/longmemdpptp_robomimic_image_emb_eval.py" \
        --config-name=transport_abs_emb_eval \
        env_type=mh dataset_path="${RAW_DATASET}" \
        work_dir="${run_dir}" model_path="${model_path}" +ckpt=10000 \
        diffusion_num_layers=4 \
        mem_compress_method="${mem_method}" mem_compress_length=2 \
        qformer_num_layers=2 qformer_cache_max_length=8 qformer_cache_compress=fifo \
        dataset_memory_subsample_ratio=20 lr=0.00001 \
        seed="${EVAL_SEED}" num_envs=10 eval_episodes=100 save_video=false wandb_mode=offline \
        +inference_action_steps=8 +dataset_maximum_memory_length=16 +eval_normalizer_only=true \
        '+memory_interventions=[none]' +context_model_path="${context_model}" \
        >"${run_log}" 2>&1
    local rc=$?
    if [[ ${rc} -eq 0 && -s "${result}" ]]; then
        log "DONE ${method} eval seed${train_seed}"
        return 0
    fi
    log "FAIL ${method} eval seed${train_seed}: rc=${rc}; see ${run_log}"
    return "${rc}"
}

log "QUEUE BEGIN: parallel random training, then serial paired evaluations"

train_pids=()
for train_seed in "${SEEDS[@]}"; do
    train_random "${train_seed}" &
    train_pids+=("$!")
done

train_fail=0
for pid in "${train_pids[@]}"; do
    if ! wait "${pid}"; then
        train_fail=1
    fi
done

if [[ ${train_fail} -ne 0 ]]; then
    log "One or more training jobs failed; successful checkpoints will still be evaluated"
fi

eval_fail=0
for train_seed in "${SEEDS[@]}"; do
    # One paired wave uses about 20 GB of the 24 GB card in practice. Keep the
    # two methods for a seed concurrent, but never overlap a third rollout.
    eval_model "${train_seed}" fifo &
    fifo_pid=$!
    eval_model "${train_seed}" random &
    random_pid=$!
    wait "${fifo_pid}" || eval_fail=1
    wait "${random_pid}" || eval_fail=1
done

"${PYTHON_BIN}" "${REPO_ROOT}/tools/analyze_random_retention_multiseed.py" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --seed100-root "${REPO_ROOT}/artifacts/learned_retention_seed100" \
    --output "${ARTIFACT_ROOT}/MULTISEED_RESULTS.md" \
    >>"${QUEUE_LOG}" 2>&1 || eval_fail=1

if [[ ${train_fail} -eq 0 && ${eval_fail} -eq 0 ]]; then
    log "QUEUE COMPLETE"
    exit 0
fi
log "QUEUE FINISHED WITH FAILURES; rerun this script to resume missing tasks"
exit 1
