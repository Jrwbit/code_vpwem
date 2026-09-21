#!/usr/bin/env bash
# Check SH2 rendering and policy inference against a successful BHR MoMaRT run.
set -euo pipefail

source /oss/sh2/vpwem/runtime/activate_momart_history4_sh2.sh

python=${VPWEM_PYTHON:-${VPWEM_ENV}/bin/python}
igibson_source=${MOMART_LOCAL_GIBSON_SOURCE:-/oss/sh2/vpwem_runtime/sources/iGibson-momart}
output=${MOMART_POSITIVE_CONTROL_OUTPUT_ROOT:-/oss/sh2/vpwem_outputs/momart_sh2_pbr_positive_control}
work=${MOMART_POSITIVE_CONTROL_LOCAL_WORK_ROOT:-/tmp/vpwem-momart-positive-control-${UID}}
result=${output}/direct_1m_table_setup_from_dresser_expert/result.npy
local_result=${work}/direct_1m_table_setup_from_dresser_expert/result.npy
log=${output}/direct_1m_table_setup_from_dresser_expert.log
hdr=${igibson_source}/gibson2/data/ig_dataset/scenes/background/photo_studio_01_2k.hdr
checkpoint=${MOMART_RAW_POLICY_ROOT}/direct/seed100/models/model_1000000.pt
config=${MOMART_RAW_POLICY_ROOT}/direct/seed100/.hydra/config.yaml
dataset=${MOMART_RAW_DATA_ROOT}/table_setup_from_dresser_expert.hdf5

expected_hdr_sha=b6178cf04ea2ac9390b8794d3088a04c0254905f335f5f30d2b582c57c40f387
expected_checkpoint_sha=8280509f044aec3b610b766f987e03803e6eea618f023fa75f4b4933c0fe8bc2
for spec in "${hdr}:${expected_hdr_sha}" "${checkpoint}:${expected_checkpoint_sha}"; do
    path=${spec%:*}
    expected=${spec##*:}
    if [[ ! -s ${path} || $(sha256sum "${path}" | cut -d' ' -f1) != "${expected}" ]]; then
        echo "Missing or incorrect positive-control input: ${path}" >&2
        exit 2
    fi
done
test -s "${config}" && test -s "${dataset}"

mkdir -p "$(dirname "${result}")" "$(dirname "${local_result}")"
exec 9>"${output}/.positive_control.lock"
flock -n 9 || { echo "MoMaRT positive control already running" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=0 GIBSON_DEVICE_ID=0
export PYTHONPATH="${igibson_source}:${MOMART_REPO}:${MOMART_REPO}/pipelines:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX=/tmp/vpwem-pycache-${UID}
export XDG_CACHE_HOME=/tmp/vpwem-cache-${UID}
export WANDB_MODE=offline PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd "${MOMART_REPO}"

printf 'START positive_control time=%s checkpoint=1m task=table_setup_from_dresser_expert episodes=10 envs=10 videos=1\n' "$(date -Is)"
"${python}" pipelines/eval_fair_history_momart_multitask_emb.py \
    --checkpoint "${checkpoint}" --checkpoint-step 1000000 \
    --train-config "${config}" \
    --dataset-path "${dataset}" \
    --env-name table_setup_from_dresser --env-type expert \
    --work-dir "${work}/direct_1m_table_setup_from_dresser_expert" \
    --result-path "${local_result}" \
    --seed 20260917 --num-envs 10 --eval-episodes 10 \
    --num-video-episodes 1 >"${log}" 2>&1

test -s "${local_result}"
cp "${local_result}" "${result}"
"${python}" -c '
import sys
import numpy as np

result = np.load(sys.argv[1], allow_pickle=True).item()
success = np.asarray(result["episode_success"], dtype=np.int8)
print(f"positive_control successes={int(success.sum())}/{len(success)}")
if len(success) != 10 or success.mean() < 0.5:
    raise SystemExit("positive control failed: SH2 behavior differs from BHR")
' "${result}"
printf 'PASS positive_control time=%s\n' "$(date -Is)"
