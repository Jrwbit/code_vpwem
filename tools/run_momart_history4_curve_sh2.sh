#!/usr/bin/env bash
# Evaluate the transferred four-method MoMaRT checkpoints without retraining.
set -uo pipefail

source /oss/sh2/vpwem/runtime/activate_momart_history4_sh2.sh

transfer=${MOMART_TRANSFER_ROOT}
output=${MOMART_CURVE_OUTPUT_ROOT:-/oss/sh2/vpwem_outputs/momart_history4_curve_800_900}
seed=${EVAL_SEED:-20260917}
episodes=${EVAL_EPISODES:-30}
num_envs=${NUM_ENVS:-10}
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

mkdir -p "${output}/logs"
exec 9>"${output}/.queue.lock"
flock -n 9 || { echo "MoMaRT curve queue already running" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${gpu}"
export GIBSON_DEVICE_ID="${gpu}"
export PYTHONPATH="${MOMART_REPO}:${MOMART_REPO}/pipelines:${PYTHONPATH:-}"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd "${MOMART_REPO}" || exit 1

printf 'START time=%s seed=%s episodes=%s envs=%s gpu=%s\n' \
    "$(date -Is)" "${seed}" "${episodes}" "${num_envs}" "${gpu}"

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

            mkdir -p "${work}"
            printf 'START %s %s %s %s\n' "$(date -Is)" "${method}" "${step}" "${task}"
            "${python}" pipelines/eval_fair_history_momart_multitask_emb.py \
                --checkpoint "${checkpoint}" --checkpoint-step "${step}" \
                --train-config "${config}" "${context_args[@]}" \
                --dataset-path "${data}" --env-name "${env_name}" --env-type "${env_type}" \
                --work-dir "${work}" --result-path "${result}" \
                --seed "${seed}" --num-envs "${num_envs}" \
                --eval-episodes "${episodes}" --num-video-episodes 0 \
                >"${log}" 2>&1
            status=$?
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
