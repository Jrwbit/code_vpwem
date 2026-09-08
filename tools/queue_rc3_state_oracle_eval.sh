#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-4}
case "${GPU}" in 2|3|4|5|6) ;; *) exit 2 ;; esac

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
RESULT=${OUT}/remember_color3/parity600/diagnostics/current_env_state_oracle100.json
LOG=${OUT}/logs/rc3_state_oracle100_gpu${GPU}.log
CKPT=/home/jrw/MIKASA-Robo/oracle_checkpoints/ncm/RememberColor3-v0/final_success_ckpt.pt

mkdir -p "$(dirname "${RESULT}")" "$(dirname "${LOG}")"
[[ -s "${CKPT}" ]]
[[ ! -s "${RESULT}" ]] || exit 0

export GPU_POLICY_ALLOWED_GPUS=2,3,4,5,6
export GPU_POLICY_INITIAL_MAX_USED_MIB=11000
export GPU_POLICY_INITIAL_MAX_UTILIZATION=20
export GPU_POLICY_INITIAL_STABLE_SAMPLES=3
export GPU_POLICY_POLL_SECONDS=20
source "${ROOT}/tools/gpu_queue_policy.sh"
wait_for_gpu_turn "${GPU}" >>"${LOG}" 2>&1

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=${ROOT}:/home/jrw/MIKASA-Robo${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUNBUFFERED=1
echo "[$(date -Is)] START gpu=${GPU}" >>"${LOG}"
"${PY}" "${ROOT}/tools/eval_rc3_state_oracle.py" \
  --checkpoint "${CKPT}" --output "${RESULT}" \
  --episodes 100 --seed 20260824 --max-episode-steps 60 \
  >>"${LOG}" 2>&1
[[ -s "${RESULT}" ]]
echo "[$(date -Is)] COMPLETE result=${RESULT}" >>"${LOG}"
