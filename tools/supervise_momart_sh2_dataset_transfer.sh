#!/usr/bin/env bash
set -uo pipefail

BHR_HOST=${BHR_HOST:-bhr-server}
SH2_HOST=${SH2_HOST:-kai_shihao_sh2}
SHARD_COUNT=${SHARD_COUNT:-4}
MULTIPART_THREADS=${MULTIPART_THREADS:-8}
POLL_SECONDS=${POLL_SECONDS:-300}

endpoint=https://oss-cn-shanghai.aliyuncs.com
bucket=oss-pai-vo6hzhj5arm1tc6x5v-cn-shanghai
python=/home/jrw/.local/vpwem_oss_venv/bin/python
uploader=/home/jrw/.local/oss_sts_dataset_upload.py
logroot=/home/jrw/.local/momart_dataset_upload_logs
object_base=shihao/oss-sh2/vpwem_data
state_root=${XDG_STATE_HOME:-${HOME}/.local/state}/vpwem
state_log=${state_root}/momart_sh2_dataset_transfer.log
marker=/oss/sh2/vpwem_data/MOMART_DATASETS_TRANSFER_COMPLETE

mkdir -p "${state_root}"
exec >>"${state_log}" 2>&1

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

active_uploads() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${BHR_HOST}" \
        'ps -eo args | grep -c "[o]ss_sts_dataset_upload.py.*_final_" || true'
}

manifest_bhr() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${BHR_HOST}" \
        'for spec in raw:/home/jrw/vpwem_data/momart emb:/home/jrw/vpwem_data/momart_emb_8layer; do label=${spec%%:*}; dir=${spec#*:}; find "$dir" -maxdepth 1 -type f -name "*.hdf5" -printf "$label/%f\t%s\n"; done | sort'
}

manifest_sh2() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${SH2_HOST}" \
        'for spec in raw:/oss/sh2/vpwem_data/momart emb:/oss/sh2/vpwem_data/momart_emb_8layer; do label=${spec%%:*}; dir=${spec#*:}; find "$dir" -maxdepth 1 -type f -name "*.hdf5" -printf "$label/%f\t%s\n" 2>/dev/null; done | sort'
}

complete() {
    local source_manifest target_manifest
    source_manifest=$(manifest_bhr) || return 1
    target_manifest=$(manifest_sh2) || return 1
    [[ $(wc -l <<<"${source_manifest}") -eq 20 ]] || return 1
    [[ "${source_manifest}" == "${target_manifest}" ]]
}

validate_sh2() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${SH2_HOST}" 'bash -s' <<'EOS'
set -euo pipefail
source /oss/sh2/vpwem/runtime/activate_momart_history4_sh2.sh

test "$(find "${MOMART_RAW_DATA_ROOT}" -maxdepth 1 -type f -name '*.hdf5' | wc -l)" -eq 10
test "$(find "${MOMART_EMB_DATA_ROOT}" -maxdepth 1 -type f -name '*.hdf5' | wc -l)" -eq 10
test "$(find "${MOMART_CONTEXT_ROOT}" -type f -path '*/obs_2_act_8_8layer/100/models/model_1000000.pt' | wc -l)" -eq 10
test -s "${MOMART_VPWEM_ROOT}/models/model_1000000.pt"
test -s "${MOMART_RAW_POLICY_ROOT}/direct/seed100/models/model_1000000.pt"
test -s "${MOMART_RAW_POLICY_ROOT}/direct_token2/seed100/models/model_1000000.pt"
test -s "${MOMART_NO_HISTORY_ROOT}/models/model_1000000.pt"
test "$(find "${MOMART_TRANSFER_ROOT}/evaluations" -type f -name result.npy | wc -l)" -eq 40

"${VPWEM_ENV}/bin/python" - <<'PY'
from pathlib import Path
import h5py

roots = (
    Path("/oss/sh2/vpwem_data/momart"),
    Path("/oss/sh2/vpwem_data/momart_emb_8layer"),
)
for root in roots:
    for path in sorted(root.glob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            if len(handle.keys()) == 0:
                raise RuntimeError(f"empty HDF5: {path}")
PY

cd "${MOMART_REPO}"
PYTHONPATH="${MOMART_REPO}:${MOMART_REPO}/pipelines" \
    "${VPWEM_ENV}/bin/python" -m pytest -q \
    tests/test_momart_embedding_condition.py \
    tests/test_momart_nohistory_eval.py \
    tests/test_multitask.py
EOS
}

launch_shard() {
    local kind=$1 source=$2 prefix=$3 shard=$4
    log "launch kind=${kind} shard=${shard}/${SHARD_COUNT}"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${SH2_HOST}" \
        'curl -fsS http://localhost:7002/api/v1/credentials/0' |
        ssh -o BatchMode=yes -o ConnectTimeout=10 "${BHR_HOST}" \
            "${python} ${uploader} --endpoint ${endpoint} --bucket ${bucket} --source-root ${source} --object-prefix ${prefix} --shard-index ${shard} --shard-count ${SHARD_COUNT} --multipart-threads ${MULTIPART_THREADS} --part-size-mib 64 --daemon-log ${logroot}/${kind}_final_${shard}.log"
}

launch_all() {
    local kind source prefix shard
    for kind in raw emb; do
        if [[ ${kind} == raw ]]; then
            source=/home/jrw/vpwem_data/momart
        else
            source=/home/jrw/vpwem_data/momart_emb_8layer
        fi
        prefix=${object_base}/${kind/raw/momart}
        [[ ${kind} == emb ]] && prefix=${object_base}/momart_emb_8layer
        for ((shard = 0; shard < SHARD_COUNT; shard++)); do
            launch_shard "${kind}" "${source}" "${prefix}" "${shard}" || \
                log "launch_failed kind=${kind} shard=${shard}"
            sleep 10
        done
    done
}

log "supervisor_start"
while true; do
    if complete; then
        log "size_manifest_complete validating_sh2"
        if validate_sh2; then
            completed_at=$(date -Is)
            ssh -o BatchMode=yes -o ConnectTimeout=10 "${SH2_HOST}" \
                "printf 'completed_at=%s\\nraw_hdf5=10\\nemb_hdf5=10\\nverification=source_target_name_size_hdf5_and_targeted_tests_passed\\n' '${completed_at}' > '${marker}'"
            log "complete marker=${marker}"
            exit 0
        fi
        log "validation_failed; will retry"
    fi

    active=$(active_uploads 2>/dev/null || printf 'unknown')
    log "incomplete active=${active}"
    if [[ ${active} == 0 ]]; then
        launch_all
    fi
    sleep "${POLL_SECONDS}"
done
