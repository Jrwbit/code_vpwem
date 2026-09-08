#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/wjr/code/vpwem"
PY="/home/wjr/miniconda3/envs/vpwem5090/bin/python"
OUT="${ROOT}/artifacts/random_retention_mechanism_seed100"
EMB="${ROOT}/data/robomimic/datasets/transport/mh_emb_seed100/image_abs_emb.hdf5"
RAW="${ROOT}/data/robomimic/datasets/transport/mh/image_abs.hdf5"
BASE="${ROOT}/outputs/ours/robomimic_emb/transport_mh/vpwem_reproduction/100/models/model_1000000.pt"
CONTEXT="${ROOT}/outputs/robomimic/dpptp/transport_mh/obs_2_act_8/100/models/model_1000000.pt"

export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=offline
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/home/wjr/miniconda3/envs/vpwem5090/lib/python3.9/site-packages/mujoco_py/binaries/linux${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${OUT}/logs"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUT}/logs/queue.log"; }

train_variant() {
    local name="$1" method="$2" cache="$3"
    local dir="${OUT}/${name}_train_10k" ckpt="${OUT}/${name}_train_10k/models/model_10000.pt"
    [[ -s "${ckpt}" ]] && { log "SKIP train ${name}"; return 0; }
    log "START train ${name}"
    "${PY}" "${ROOT}/pipelines/longmemdpptp_robomimic_image_emb.py" \
        --config-name=transport_abs_emb env_type=mh dataset_path="${EMB}" \
        save_path="${ROOT}/artifacts/per_vpwem" work_dir="${dir}" \
        gradient_steps=10000 save_freq=10000 log_freq=500 diffusion_num_layers=4 \
        mem_compress_method="${method}" mem_compress_length=2 qformer_num_layers=2 \
        qformer_cache_max_length=8 qformer_cache_compress="${cache}" \
        dataset_memory_subsample_ratio=20 lr=0.00001 seed=100 wandb_mode=offline \
        +num_workers=2 +dataset_maximum_memory_length=16 +inference_action_steps=8 \
        +memory_aux_coeff=0.0 +counterfactual_memory_coeff=0.0 \
        +counterfactual_memory_margin=0.01 +counterfactual_memory_probability=0.0 \
        +init_model_path="${BASE}" +keep_checkpoints=1 \
        >"${OUT}/logs/${name}_train.log" 2>&1
    local rc=$?
    [[ ${rc} -eq 0 && -s "${ckpt}" ]] && { log "DONE train ${name}"; return 0; }
    log "FAIL train ${name} rc=${rc}"
    return 1
}

eval_variant() {
    local name="$1" method="$2" cache="$3"
    local dir="${OUT}/${name}_eval_100" result="${OUT}/${name}_eval_100/result_10000_memory-none.npy"
    local ckpt="${OUT}/${name}_train_10k/models/model_10000.pt"
    [[ -s "${result}" ]] && { log "SKIP eval ${name}"; return 0; }
    [[ ! -s "${ckpt}" ]] && { log "FAIL eval ${name}: checkpoint missing"; return 1; }
    log "START eval ${name}"
    "${PY}" "${ROOT}/pipelines/longmemdpptp_robomimic_image_emb_eval.py" \
        --config-name=transport_abs_emb_eval env_type=mh dataset_path="${RAW}" \
        work_dir="${dir}" model_path="${ckpt}" +ckpt=10000 diffusion_num_layers=4 \
        mem_compress_method="${method}" mem_compress_length=2 qformer_num_layers=2 \
        qformer_cache_max_length=8 qformer_cache_compress="${cache}" \
        dataset_memory_subsample_ratio=20 lr=0.00001 seed=20260814 \
        num_envs=10 eval_episodes=100 save_video=false wandb_mode=offline \
        +inference_action_steps=8 +dataset_maximum_memory_length=16 +eval_normalizer_only=true \
        '+memory_interventions=[none]' +context_model_path="${CONTEXT}" \
        >"${OUT}/logs/${name}_eval.log" 2>&1
    local rc=$?
    [[ ${rc} -eq 0 && -s "${result}" ]] && { log "DONE eval ${name}"; return 0; }
    log "FAIL eval ${name} rc=${rc}"
    return 1
}

log "QUEUE BEGIN"
train_variant recent2 recent fifo & p1=$!
train_variant qformer_random8 qformer random & p2=$!
fail=0
wait "${p1}" || fail=1
wait "${p2}" || fail=1

eval_variant recent2 recent fifo & p1=$!
eval_variant qformer_random8 qformer random & p2=$!
wait "${p1}" || fail=1
wait "${p2}" || fail=1

"${PY}" "${ROOT}/tools/analyze_random_mechanism.py" \
    --root "${OUT}" --reference-root "${ROOT}/artifacts/learned_retention_seed100" \
    --output "${OUT}/MECHANISM_RESULTS.md" >>"${OUT}/logs/queue.log" 2>&1 || fail=1

[[ ${fail} -eq 0 ]] && { log "QUEUE COMPLETE"; exit 0; }
log "QUEUE FINISHED WITH FAILURES; rerun to resume"
exit 1
