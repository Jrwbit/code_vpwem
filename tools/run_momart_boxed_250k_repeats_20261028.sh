#!/usr/bin/env bash
# Independent evaluation seeds; fixed 250k policy and combined_blend detector.
set -euo pipefail
repo=/home/jrw/code_vpwem_boxed_continue_20261027
export MOMART_REPO=$repo
export MOMART_RUN_ROOT=/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100
export EVAL_METHOD=boxed_no_history EVAL_STEPS=250000 EVAL_EPISODES=30 NUM_ENVS=10
export PRECISION_CODE_ROOT=$repo
export PRECISION_BUNDLE=/home/jrw/vpwem_outputs/bbox_strategies_20261020/comparison_v2/detector_bundle.json
export VPWEM_PYTHON=/home/jrw/anaconda3/envs/vpwem/bin/python
export OPENBLAS_NUM_THREADS=1
case "${1:?Specify physical GPU: 3 or 4}" in
    3) export GPU=3 EVAL_SEED=20260918 ;;
    4) export GPU=4 EVAL_SEED=20260919 ;;
    *) exit 2 ;;
esac
export MOMART_EVAL_ROOT=$MOMART_RUN_ROOT/eval_online_combined_blend_250k_repeat_seed${EVAL_SEED}_20261028
exec python3 "$repo/tools/gpu_gate.py" run --gpu "$GPU" \
    --min-free 20000 --max-util 20 --stable 3 --poll 10 -- \
    bash "$repo/tools/run_momart_boxed_checkpoint_eval_bhr.sh"
