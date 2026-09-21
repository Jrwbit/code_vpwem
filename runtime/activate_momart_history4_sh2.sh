#!/usr/bin/env bash

source /oss/sh2/vpwem_runtime/activate.sh

export MOMART_TRANSFER_ROOT=/oss/sh2/vpwem_transfer/momart_history4_20260920
export MOMART_REPO=/oss/sh2/vpwem
export MOMART_RAW_DATA_ROOT=/oss/sh2/vpwem_data/momart
export MOMART_EMB_DATA_ROOT=/oss/sh2/vpwem_data/momart_emb_8layer
export MOMART_CONTEXT_ROOT=${MOMART_TRANSFER_ROOT}/context_checkpoints
export MOMART_RAW_POLICY_ROOT=${MOMART_TRANSFER_ROOT}/experiments/fair_history_momart_multitask
export MOMART_VPWEM_ROOT=${MOMART_TRANSFER_ROOT}/experiments/fair_history_momart_multitask_emb/vpwem_bs256/seed100
export MOMART_NO_HISTORY_ROOT=${MOMART_RAW_POLICY_ROOT}/no_history/seed100
export MOMART_EVAL_OUTPUT_ROOT=/oss/sh2/vpwem_outputs/fair_history_momart_multitask_eval_1m

# The current SH2 instance exposes one GPU. Run method queues serially unless
# the caller explicitly assigns additional devices.
export DIRECT_GPU=${DIRECT_GPU:-0}
export DIRECT_TOKEN2_GPU=${DIRECT_TOKEN2_GPU:-0}
export VPWEM_GPU=${VPWEM_GPU:-0}
export GPU=${GPU:-0}
export RUN_SEQUENTIAL=${RUN_SEQUENTIAL:-1}

cd "${MOMART_REPO}"
