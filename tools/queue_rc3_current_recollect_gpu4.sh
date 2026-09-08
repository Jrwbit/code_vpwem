#!/usr/bin/env bash
set -euo pipefail

# A short, authorized co-card collection beside the old-demo RC3 training.
# Only this collector is stopped if aggregate GPU pressure reaches 23 GiB.

GPU=4
ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
DATA=${NCM_RC3_CURRENT_DATASET:-/home/jrw/vpwem_data/mikasa/RememberColor3-v0-current-20260824}
CHECKPOINT=${NCM_RC3_ORACLE:-/home/jrw/MIKASA-Robo/oracle_checkpoints/ncm/RememberColor3-v0/final_success_ckpt.pt}
MARKER=${DATA}/RC3_CURRENT_ORACLE_DATASET.json
LOG=${OUT}/logs/rc3_current_oracle1000_collect_gpu4.log

mkdir -p "${DATA}" "${OUT}/logs"
[[ -s "${CHECKPOINT}" ]]
if [[ -s "${MARKER}" ]] && grep -q '"complete": true' "${MARKER}"; then
  echo "[$(date -Is)] SKIP complete dataset=${DATA}"
  exit 0
fi

export GPU_POLICY_ALLOWED_GPUS=2,3,4,5,6
export GPU_POLICY_INITIAL_MAX_USED_MIB=11000
# This is an explicitly authorized short co-card job; admission is based on
# memory headroom, not the old trainer's alternating utilization samples.
export GPU_POLICY_INITIAL_MAX_UTILIZATION=100
export GPU_POLICY_INITIAL_STABLE_SAMPLES=3
export GPU_POLICY_POLL_SECONDS=20
source "${ROOT}/tools/gpu_queue_policy.sh"
wait_for_gpu_turn "${GPU}"

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=${ROOT}:/home/jrw/MIKASA-Robo${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUNBUFFERED=1
echo "[$(date -Is)] START gpu=${GPU} dataset=${DATA}" >>"${LOG}"
"${PY}" "${ROOT}/tools/collect_rc3_current_oracle.py" \
  --checkpoint "${CHECKPOINT}" \
  --output "${DATA}" \
  --episodes 1000 \
  --batch-size 250 \
  --seed-start 0 \
  --max-episode-steps 60 \
  --post-success-steps 8 \
  --max-source-batches 20 \
  >>"${LOG}" 2>&1 &
collector_pid=$!

guard_triggered=0
while kill -0 "${collector_pid}" 2>/dev/null; do
  used=$(nvidia-smi -i "${GPU}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  if (( used >= 23000 )); then
    guard_triggered=1
    echo "[$(date -Is)] OOM_GUARD used_mib=${used} action=term-collector-only pid=${collector_pid}" \
      >>"${LOG}"
    kill -TERM "${collector_pid}" 2>/dev/null || true
    break
  fi
  sleep 10
done

set +e
wait "${collector_pid}"
rc=$?
set -e
if (( guard_triggered )); then
  exit 75
fi
(( rc == 0 )) || exit "${rc}"
[[ -s "${MARKER}" ]]
grep -q '"complete": true' "${MARKER}"
echo "[$(date -Is)] COMPLETE dataset=${DATA}" >>"${LOG}"
