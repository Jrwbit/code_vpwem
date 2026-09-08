#!/usr/bin/env bash
set -euo pipefail

# Fresh RC3 training on current-version demonstrations.  GPU3 admission is
# deliberately behind both full-base branches so it cannot steal the NCM main
# line's current or already-queued stage/evaluation work.

GPU=3
ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
DATA=${NCM_RC3_CURRENT_DATASET:-/home/jrw/vpwem_data/mikasa/RememberColor3-v0-current-20260824}
MARKER=${DATA}/RC3_CURRENT_ORACLE_DATASET.json
WORK=${OUT}/remember_color3/current_demo600/train
LOG=${OUT}/logs/rc3_current_demo600_gpu3.log
RC5_CAUSAL_DONE=${OUT}/remember_color5/physical_scratch400/CAUSAL_EVALS_COMPLETE
SHELL_FULLBASE=${OUT}/shell/stage1_fullbase600
RC5_FULLBASE=${OUT}/remember_color5/stage1_fullbase400

mkdir -p "${WORK}" "${OUT}/logs"

marker_is_done() {
  local directory=$1
  [[ -f "${directory}/COMPLETE" || -f "${directory}/BASE_MEMORY_NO_GO" ]]
}

echo "[$(date -Is)] WAIT collection + RC5 causal + both fullbase branches" >>"${LOG}"
while true; do
  collection_done=0
  [[ -s "${MARKER}" ]] && grep -q '"complete": true' "${MARKER}" \
    && collection_done=1
  if (( collection_done )) && [[ -f "${RC5_CAUSAL_DONE}" ]] \
      && marker_is_done "${SHELL_FULLBASE}" \
      && marker_is_done "${RC5_FULLBASE}"; then
    break
  fi
  sleep 60
done

"${PY}" - "${DATA}" "${MARKER}" <<'PY'
import json
import pathlib
import sys

data = pathlib.Path(sys.argv[1])
with open(sys.argv[2], encoding="utf-8") as stream:
    marker = json.load(stream)
assert marker["complete"] is True
assert marker["successful_episodes"] == 1000
assert len(marker["episodes"]) == 1000
assert all((data / f"train_data_{index}.npz").is_file()
           for index in range(1000))
assert not (data / "train_data_1000.npz").exists()
PY

export GPU_POLICY_ALLOWED_GPUS=2,3,4,5,6
export GPU_POLICY_INITIAL_MAX_USED_MIB=1024
export GPU_POLICY_INITIAL_MAX_UTILIZATION=20
export GPU_POLICY_INITIAL_STABLE_SAMPLES=3
export GPU_POLICY_POLL_SECONDS=60
source "${ROOT}/tools/gpu_queue_policy.sh"
wait_for_gpu_turn "${GPU}" >>"${LOG}" 2>&1

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
echo "[$(date -Is)] START gpu=${GPU} dataset=${DATA}" >>"${LOG}"
"${PY}" "${ROOT}/pipelines/longmemdpptp_mikasa_image.py" \
  --config-name=remember_color_3 \
  dataset_path="${DATA}" \
  work_dir="${WORK}" \
  save_path="${OUT}" \
  exp_name=crossbench/remember_color3_current_demo600 \
  gradient_steps=600000 \
  batch_size=64 \
  save_freq=20000 \
  ++checkpoint_save_early_steps='[5000,10000]' \
  ++keep_checkpoints=40 \
  ++num_workers=8 \
  seed=100 \
  device=cuda:0 \
  wandb_mode=offline \
  +inference_action_steps=4 \
  +dataset_preserve_episode_anchor=true \
  >>"${LOG}" 2>&1
[[ -s "${WORK}/models/model_600000.pt" ]]
echo "[$(date -Is)] COMPLETE model=${WORK}/models/model_600000.pt" >>"${LOG}"
