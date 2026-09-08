#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/wjr/code/vpwem"
PY="/home/wjr/miniconda3/envs/vpwem5090/bin/python"
OUT="${ROOT}/artifacts/random_reservoir_crossbench_strict"
TRAIN_SEED=100
EVAL_SEED=20260821
RETENTION_SEED=730100
# Override with, for example: TASKS="longsquare_ph square_mh" ./tools/...
read -r -a TASK_LIST <<< "${TASKS:-longsquare_ph square_mh transport_ph}"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export WANDB_MODE=offline
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/home/wjr/miniconda3/envs/vpwem5090/lib/python3.9/site-packages/mujoco_py/binaries/linux${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${OUT}/logs"
QUEUE_LOG="${OUT}/logs/queue.log"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${QUEUE_LOG}"; }

task_paths() {
    local task="$1"
    case "${task}" in
        longsquare_ph)
            CONFIG=longsquare_abs_emb
            ENV_NAME=longsquare
            ENV_TYPE=ph
            EMB="${ROOT}/data/robomimic/datasets/longhistsquare100_emb_seed100/demos_emb.hdf5"
            RAW="${ROOT}/data/robomimic/datasets/longhistsquare100/demos.hdf5"
            BASE="${ROOT}/outputs/ours/robomimic_emb/longsquare_ph/vpwem_reproduction/100/models/model_1000000.pt"
            CONTEXT="${ROOT}/outputs/robomimic/dpptp/longsquare_ph/obs_2_act_8/100/models/model_1000000.pt"
            ;;
        square_mh)
            CONFIG=square_abs_emb
            ENV_NAME=square
            ENV_TYPE=mh
            EMB="${ROOT}/data/robomimic/datasets/square/mh_emb_seed100/image_abs_emb.hdf5"
            RAW="${ROOT}/data/robomimic/datasets/square/mh/image_abs.hdf5"
            BASE="${ROOT}/outputs/ours/robomimic_emb/square_mh/vpwem_reproduction/100/models/model_1000000.pt"
            CONTEXT="${ROOT}/outputs/robomimic/dpptp/square_mh/obs_2_act_8/100/models/model_1000000.pt"
            ;;
        transport_ph)
            CONFIG=transport_abs_emb
            ENV_NAME=transport
            ENV_TYPE=ph
            EMB="${ROOT}/data/robomimic/datasets/transport/ph_emb_seed100/image_abs_emb.hdf5"
            RAW="${ROOT}/data/robomimic/datasets/transport/ph/image_abs.hdf5"
            BASE="${ROOT}/outputs/ours/robomimic_emb/transport_ph/vpwem_reproduction/100/models/model_1000000.pt"
            CONTEXT="${ROOT}/outputs/robomimic/dpptp/transport_ph/obs_2_act_8/100/models/model_1000000.pt"
            ;;
        square_ph)
            CONFIG=square_abs_emb
            ENV_NAME=square
            ENV_TYPE=ph
            EMB="${ROOT}/data/robomimic/datasets/square/ph_emb_seed100/image_abs_emb.hdf5"
            RAW="${ROOT}/data/robomimic/datasets/square/ph/image_abs.hdf5"
            BASE="${ROOT}/outputs/ours/robomimic_emb/square_ph/vpwem_reproduction_qformer2/100/models/model_1000000.pt"
            CONTEXT="${ROOT}/outputs/robomimic/dpptp/square_ph/obs_2_act_8/100/models/model_1000000.pt"
            ;;
        *) log "Unknown task ${task}"; return 2 ;;
    esac
}

train_variant() {
    local task="$1" name="$2" method="$3"
    task_paths "${task}" || return $?
    local dir="${OUT}/${task}/seed${TRAIN_SEED}/${name}_train_10k"
    local checkpoint="${dir}/models/model_10000.pt"
    local run_log="${OUT}/logs/${task}_${name}_train.log"
    [[ -s "${checkpoint}" ]] && { log "SKIP train ${task} ${name}"; return 0; }
    for required in "${EMB}" "${BASE}"; do
        [[ -s "${required}" ]] || { log "FAIL missing ${required}"; return 2; }
    done
    mkdir -p "${dir}"
    log "START train ${task} ${name}"
    "${PY}" "${ROOT}/pipelines/longmemdpptp_robomimic_image_emb.py" \
        --config-name="${CONFIG}" env_name="${ENV_NAME}" env_type="${ENV_TYPE}" \
        dataset_path="${EMB}" work_dir="${dir}" \
        gradient_steps=10000 save_freq=10000 log_freq=500 diffusion_num_layers=4 \
        mem_compress_method="${method}" mem_compress_length=2 \
        dataset_memory_subsample_ratio=20 lr=0.00001 seed="${TRAIN_SEED}" \
        wandb_mode=offline +num_workers=2 +dataset_maximum_memory_length=160 \
        +inference_action_steps=8 +memory_aux_coeff=0.0 \
        +counterfactual_memory_coeff=0.0 +counterfactual_memory_margin=0.01 \
        +counterfactual_memory_probability=0.0 +init_model_path="${BASE}" \
        +retention_seed="${RETENTION_SEED}" +keep_checkpoints=1 \
        >"${run_log}" 2>&1
    local rc=$?
    [[ ${rc} -eq 0 && -s "${checkpoint}" ]] && { log "DONE train ${task} ${name}"; return 0; }
    log "FAIL train ${task} ${name} rc=${rc}; see ${run_log}"
    return 1
}

eval_variant() {
    local task="$1" name="$2" method="$3"
    task_paths "${task}" || return $?
    local base_dir="${OUT}/${task}/seed${TRAIN_SEED}"
    local checkpoint="${base_dir}/${name}_train_10k/models/model_10000.pt"
    local dir="${base_dir}/${name}_eval_100"
    local result="${dir}/result_10000_memory-none.npy"
    local run_log="${OUT}/logs/${task}_${name}_eval.log"
    [[ -s "${result}" ]] && { log "SKIP eval ${task} ${name}"; return 0; }
    for required in "${RAW}" "${CONTEXT}" "${checkpoint}"; do
        [[ -s "${required}" ]] || { log "FAIL missing ${required}"; return 2; }
    done
    mkdir -p "${dir}"
    log "START eval ${task} ${name}"
    "${PY}" "${ROOT}/pipelines/longmemdpptp_robomimic_image_emb_eval.py" \
        --config-name="${CONFIG}_eval" env_name="${ENV_NAME}" env_type="${ENV_TYPE}" \
        dataset_path="${RAW}" work_dir="${dir}" model_path="${checkpoint}" +ckpt=10000 \
        diffusion_num_layers=4 mem_compress_method="${method}" mem_compress_length=2 \
        dataset_memory_subsample_ratio=20 lr=0.00001 seed="${EVAL_SEED}" \
        num_envs=10 eval_episodes=100 save_video=false wandb_mode=offline \
        +inference_action_steps=8 +dataset_maximum_memory_length=160 \
        +eval_normalizer_only=true '+memory_interventions=[none]' \
        +context_model_path="${CONTEXT}" +retention_seed="${RETENTION_SEED}" \
        +diffusion_seed="${EVAL_SEED}" \
        >"${run_log}" 2>&1
    local rc=$?
    [[ ${rc} -eq 0 && -s "${result}" ]] && { log "DONE eval ${task} ${name}"; return 0; }
    log "FAIL eval ${task} ${name} rc=${rc}; see ${run_log}"
    return 1
}

fail=0
log "QUEUE BEGIN tasks=${TASK_LIST[*]}"
for task in "${TASK_LIST[@]}"; do
    train_variant "${task}" recent2 recent & recent_pid=$!
    train_variant "${task}" reservoir2 random & reservoir_pid=$!
    wait "${recent_pid}" || fail=1
    wait "${reservoir_pid}" || fail=1
    eval_variant "${task}" recent2 recent & recent_pid=$!
    eval_variant "${task}" reservoir2 random & reservoir_pid=$!
    wait "${recent_pid}" || fail=1
    wait "${reservoir_pid}" || fail=1
    "${PY}" "${ROOT}/tools/analyze_random_reservoir_crossbench.py" \
        --root "${OUT}" --output "${OUT}/CROSSBENCH_RESULTS.md" \
        >>"${QUEUE_LOG}" 2>&1 || fail=1
done

[[ ${fail} -eq 0 ]] && { log "QUEUE COMPLETE"; exit 0; }
log "QUEUE FINISHED WITH FAILURES; rerun to resume"
exit 1
