#!/usr/bin/env bash
# Run on BHR: one checkpoint per physical GPU, same protocol as 250k.
set -euo pipefail
repo=/home/jrw/code_vpwem_boxed_continue_20261027
export MOMART_REPO=$repo
export MOMART_RUN_ROOT=/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100
export MOMART_EVAL_ROOT=$MOMART_RUN_ROOT/eval_online_combined_blend_175k_200k_225k_20261028
export EVAL_METHOD=boxed_no_history EVAL_SEED=20260917 EVAL_EPISODES=30 NUM_ENVS=10
export PRECISION_CODE_ROOT=$repo
export PRECISION_BUNDLE=/home/jrw/vpwem_outputs/bbox_strategies_20261020/comparison_v2/detector_bundle.json
export VPWEM_PYTHON=/home/jrw/anaconda3/envs/vpwem/bin/python
export OPENBLAS_NUM_THREADS=1
case "${1:?Specify physical GPU: 0, 1, or 7}" in
    0) export GPU=0 EVAL_STEPS=175000 ;;
    1) export GPU=1 EVAL_STEPS=200000 ;;
    7) export GPU=7 EVAL_STEPS=225000 ;;
    *) exit 2 ;;
esac
exec python3 "$repo/tools/gpu_gate.py" run --gpu "$GPU" \
    --min-free 20000 --max-util 20 --stable 3 --poll 10 -- \
    bash "$repo/tools/run_momart_boxed_checkpoint_eval_bhr.sh"
