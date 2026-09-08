#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-4}
CHECKPOINT=${2:-10000}
EPISODES=${3:-30}
NUM_ENVS=${4:-10}

case "${GPU}" in
  2|3|4|5|6) ;;
  *)
    echo "Refusing GPU ${GPU}; RC3 diagnostics are pinned to GPUs 2-6" >&2
    exit 2
    ;;
esac
if (( EPISODES % NUM_ENVS != 0 )); then
  echo "EPISODES must be divisible by NUM_ENVS" >&2
  exit 2
fi

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
DATA_ROOT=${NCM_DATA_ROOT:-/home/jrw/vpwem_data/mikasa}
RUN_ROOT=${OUT}/remember_color3/parity600
MODEL=${RUN_ROOT}/train/models/model_${CHECKPOINT}.pt
DIAG_ROOT=${RUN_ROOT}/diagnostics
EXPERT_RESULT=${DIAG_ROOT}/current_env_expert_replay10.json
POLICY_DIR=${DIAG_ROOT}/policy_${CHECKPOINT}_chunk4_eval${EPISODES}
POLICY_RESULT=${POLICY_DIR}/result.npy
LOG=${OUT}/logs/rc3_parity_diagnostics_gpu${GPU}.log

mkdir -p "${DIAG_ROOT}" "${POLICY_DIR}" "${OUT}/logs"
exec 9>"${OUT}/.rc3-parity-diagnostics-gpu${GPU}.lock"
flock 9

# Preserve the requested ordering until the complete SwapBlocks fixed-policy
# queue, including its A/B children and monitor, has exited.  The character
# class prevents pgrep from matching its own regex argument.
while pgrep -f '[s]wap_blocks' >/dev/null 2>&1; do
  echo "[$(date -Is)] WAIT_ALL_SWAP_PROCESSES gpu=${GPU}" >>"${LOG}"
  sleep 60
done

export GPU_POLICY_ALLOWED_GPUS=2,3,4,5,6
export GPU_POLICY_INITIAL_MAX_USED_MIB=${GPU_POLICY_INITIAL_MAX_USED_MIB:-11000}
export GPU_POLICY_INITIAL_MAX_UTILIZATION=${GPU_POLICY_INITIAL_MAX_UTILIZATION:-20}
export GPU_POLICY_INITIAL_STABLE_SAMPLES=${GPU_POLICY_INITIAL_STABLE_SAMPLES:-3}
export GPU_POLICY_POLL_SECONDS=${GPU_POLICY_POLL_SECONDS:-60}
# shellcheck source=tools/gpu_queue_policy.sh
source "${ROOT}/tools/gpu_queue_policy.sh"
wait_for_gpu_turn "${GPU}" >>"${LOG}" 2>&1

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

GPU_ABORT_USED_MIB=${GPU_ABORT_USED_MIB:-23000}
GPU_MONITOR_SECONDS=${GPU_MONITOR_SECONDS:-10}

run_with_gpu_monitor() {
  local child used utilization status
  "$@" >>"${LOG}" 2>&1 &
  child=$!
  while kill -0 "${child}" 2>/dev/null; do
    IFS=, read -r used utilization < <(nvidia-smi -i "${GPU}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits)
    used=${used// /}
    utilization=${utilization// /}
    echo "[$(date -Is)] GPU_MONITOR gpu=${GPU} child=${child} used_mib=${used} utilization=${utilization}" \
      >>"${LOG}"
    if (( used >= GPU_ABORT_USED_MIB )); then
      echo "[$(date -Is)] DIAGNOSTIC_ABORT_OOM_GUARD gpu=${GPU} child=${child} used_mib=${used} threshold_mib=${GPU_ABORT_USED_MIB}" \
        >>"${LOG}"
      kill -TERM "${child}" 2>/dev/null || true
      wait "${child}" 2>/dev/null || true
      return 86
    fi
    sleep "${GPU_MONITOR_SECONDS}"
  done
  set +e
  wait "${child}"
  status=$?
  set -e
  return "${status}"
}

if [[ ! -s "${EXPERT_RESULT}" ]]; then
  echo "[$(date -Is)] START expert-replay gpu=${GPU}" >>"${LOG}"
  run_with_gpu_monitor "${PY}" "${ROOT}/tools/replay_rc3_expert_current_env.py" \
    --dataset-dir "${DATA_ROOT}/RememberColor3-v0" \
    --output "${EXPERT_RESULT}" \
    --episodes 10 --source-batch-size 250 --seed 0
  test -s "${EXPERT_RESULT}"
  echo "[$(date -Is)] DONE expert-replay result=${EXPERT_RESULT}" >>"${LOG}"
else
  echo "[$(date -Is)] SKIP expert-replay result-exists" >>"${LOG}"
fi

test -s "${MODEL}"
if [[ ! -s "${POLICY_RESULT}" ]]; then
  echo "[$(date -Is)] START policy-diagnostics ckpt=${CHECKPOINT} gpu=${GPU}" \
    >>"${LOG}"
  run_with_gpu_monitor "${PY}" "${ROOT}/pipelines/remembercolor3_policy_diagnostics.py" \
    --config-name=remember_color_3 \
    dataset_path="${DATA_ROOT}/RememberColor3-v0" \
    model_path="${MODEL}" +result_path="${POLICY_RESULT}" \
    work_dir="${POLICY_DIR}" num_envs="${NUM_ENVS}" \
    eval_episodes="${EPISODES}" ++inference_action_steps=4 \
    max_episode_steps=60 save_video=false seed=20260823 \
    device=cuda:0 wandb_mode=offline
  test -s "${POLICY_RESULT}"
  test -s "${POLICY_DIR}/result_episodes.json"
  echo "[$(date -Is)] DONE policy-diagnostics result=${POLICY_RESULT}" \
    >>"${LOG}"
else
  echo "[$(date -Is)] SKIP policy-diagnostics result-exists" >>"${LOG}"
fi

touch "${DIAG_ROOT}/COMPLETE_${CHECKPOINT}_eval${EPISODES}"
echo "[$(date -Is)] COMPLETE rc3-parity-diagnostics" >>"${LOG}"
