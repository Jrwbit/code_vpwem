#!/usr/bin/env bash
# Evaluate the transferred four-method MoMaRT checkpoints without retraining.
set -uo pipefail

source /oss/sh2/vpwem/runtime/activate_momart_history4_sh2.sh

transfer=${MOMART_TRANSFER_ROOT}
output=${MOMART_CURVE_OUTPUT_ROOT:-/oss/sh2/vpwem_outputs/momart_history4_curve_800_900}
seed=${EVAL_SEED:-20260917}
episodes=${EVAL_EPISODES:-30}
num_envs=${NUM_ENVS:-10}
num_video_episodes=${NUM_VIDEO_EPISODES:-1}
gpu=${GPU:-0}
python=${VPWEM_PYTHON:-${VPWEM_ENV}/bin/python}
if [[ ${python} != "${VPWEM_ENV}/bin/python" ]]; then
    export PYTHONPYCACHEPREFIX=/tmp/vpwem-pycache-${UID}
    export XDG_CACHE_HOME=/tmp/vpwem-cache-${UID}
fi
methods=(${EVAL_METHODS:-no_history direct_token2 direct vpwem})
steps=(${EVAL_STEPS:-800000 900000})
tasks=(${EVAL_TASKS:-table_cleanup_to_dishwasher_expert table_cleanup_to_sink_expert table_setup_from_dishwasher_expert table_setup_from_dresser_expert})

if (( episodes <= 0 || num_envs <= 0 || episodes % num_envs != 0 )); then
    echo "EVAL_EPISODES must be a positive multiple of NUM_ENVS" >&2
    exit 2
fi
if (( num_video_episodes < 0 || num_video_episodes > episodes )); then
    echo "NUM_VIDEO_EPISODES must be between 0 and EVAL_EPISODES" >&2
    exit 2
fi

mkdir -p "${output}/logs"
lock_file=${MOMART_CURVE_LOCK_FILE:-${output}/.queue.lock}
exec 9>"${lock_file}"
flock -n 9 || { echo "MoMaRT curve queue already running" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${gpu}"
export GIBSON_DEVICE_ID="${gpu}"
export PYTHONPATH="${MOMART_REPO}:${MOMART_REPO}/pipelines:${PYTHONPATH:-}"
if [[ -n ${MOMART_LOCAL_GIBSON_SOURCE:-} ]]; then
    export PYTHONPATH="${MOMART_LOCAL_GIBSON_SOURCE}:${PYTHONPATH}"
fi
export WANDB_MODE=offline PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd "${MOMART_REPO}" || exit 1

# The MoMaRT policies were evaluated on BHR with this HDR environment map.
# Missing it silently disables iGibson PBR and invalidates visual rollouts.
igibson_source=${MOMART_LOCAL_GIBSON_SOURCE:-/oss/sh2/vpwem_runtime/sources/iGibson-momart}
hdr=${igibson_source}/gibson2/data/ig_dataset/scenes/background/photo_studio_01_2k.hdr
expected_hdr_sha=b6178cf04ea2ac9390b8794d3088a04c0254905f335f5f30d2b582c57c40f387
if [[ ! -s ${hdr} || $(sha256sum "${hdr}" | cut -d' ' -f1) != "${expected_hdr_sha}" ]]; then
    echo "Missing or incorrect iGibson HDR environment map: ${hdr}" >&2
    exit 2
fi

# A known-good 1M checkpoint must first succeed on SH2 before the 800k/900k
# curve is trusted. Keep the positive control in a separate output directory.
control=${MOMART_POSITIVE_CONTROL_RESULT:-/oss/sh2/vpwem_outputs/momart_sh2_pbr_positive_control/direct_1m_table_setup_from_dresser_expert/result.npy}
if [[ ! -s ${control} ]] || ! "${python}" -c '
import sys
import numpy as np

result = np.load(sys.argv[1], allow_pickle=True).item()
assert result["method"] == "direct"
assert result["checkpoint_step"] == 1000000
assert result["env_name"] == "table_setup_from_dresser"
assert result["env_type"] == "expert"
assert result["eval_seed"] == 20260917
assert len(result["episode_success"]) == 10
assert result["mean_success"] >= 0.8
' "${control}"; then
    echo "SH2 MoMaRT positive control is absent or failed: ${control}" >&2
    exit 2
fi

printf 'START time=%s seed=%s episodes=%s envs=%s videos=%s gpu=%s\n' \
    "$(date -Is)" "${seed}" "${episodes}" "${num_envs}" "${num_video_episodes}" "${gpu}"

failures=0
for method in "${methods[@]}"; do
    case "${method}" in
        vpwem) base=${transfer}/experiments/fair_history_momart_multitask_emb/vpwem_bs256/seed100 ;;
        direct|direct_token2|no_history)
            base=${transfer}/experiments/fair_history_momart_multitask/${method}/seed100 ;;
        *) echo "Unknown method: ${method}" >&2; exit 2 ;;
    esac
    config=${base}/.hydra/config.yaml
    for step in "${steps[@]}"; do
        checkpoint=${base}/models/model_${step}.pt
        for task in "${tasks[@]}"; do
            if [[ ${task} == *_expert ]]; then
                env_type=expert
            elif [[ ${task} == *_suboptimal ]]; then
                env_type=suboptimal
            else
                echo "Invalid task: ${task}" >&2
                exit 2
            fi
            env_name=${task%_${env_type}}
            data=${MOMART_RAW_DATA_ROOT}/${task}.hdf5
            work=${output}/${method}/${step}/${task}
            result=${work}/result.npy
            local_work=${MOMART_LOCAL_WORK_ROOT:-/tmp/vpwem-momart-curve-${UID}}/${method}/${step}/${task}
            local_result=${local_work}/result.npy
            log=${output}/logs/${method}_${step}_${task}.log

            if [[ -s ${result} ]]; then
                printf 'SKIP %s %s %s\n' "${method}" "${step}" "${task}"
                continue
            fi
            if [[ ! -s ${config} || ! -s ${checkpoint} || ! -s ${data} ]]; then
                printf 'MISSING method=%s step=%s task=%s\n' "${method}" "${step}" "${task}" >&2
                failures=$((failures + 1))
                continue
            fi
            context_args=()
            if [[ ${method} == vpwem ]]; then
                context=${MOMART_CONTEXT_ROOT}/${task}/obs_2_act_8_8layer/100/models/model_1000000.pt
                if [[ ! -s ${context} ]]; then
                    printf 'MISSING context=%s\n' "${context}" >&2
                    failures=$((failures + 1))
                    continue
                fi
                context_args=(--context-checkpoint "${context}")
            fi

            mkdir -p "${work}" "${local_work}"
            printf 'START %s %s %s %s\n' "$(date -Is)" "${method}" "${step}" "${task}"
            "${python}" pipelines/eval_fair_history_momart_multitask_emb.py \
                --checkpoint "${checkpoint}" --checkpoint-step "${step}" \
                --train-config "${config}" "${context_args[@]}" \
                --dataset-path "${data}" --env-name "${env_name}" --env-type "${env_type}" \
                --work-dir "${local_work}" --result-path "${local_result}" \
                --seed "${seed}" --num-envs "${num_envs}" \
                --eval-episodes "${episodes}" --num-video-episodes "${num_video_episodes}" \
                >"${log}" 2>&1
            status=$?
            if (( status == 0 )) && [[ -s ${local_result} ]]; then
                cp "${local_result}" "${result}" || status=$?
            fi
            if (( status == 0 )) && [[ -s ${result} ]]; then
                printf 'DONE %s %s %s %s\n' "$(date -Is)" "${method}" "${step}" "${task}"
            else
                printf 'FAIL %s %s %s rc=%s log=%s\n' \
                    "${method}" "${step}" "${task}" "${status}" "${log}" >&2
                failures=$((failures + 1))
            fi
        done
    done
done

printf 'FINISH time=%s failures=%s\n' "$(date -Is)" "${failures}"
exit "$((failures > 0))"
