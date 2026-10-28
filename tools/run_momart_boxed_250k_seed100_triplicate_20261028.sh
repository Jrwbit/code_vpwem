#!/usr/bin/env bash
# Three separate same-seed runs, then two different-seed runs after a barrier.
set -euo pipefail
repo=/home/jrw/code_vpwem_boxed_continue_20261027
export MOMART_REPO=$repo
export MOMART_RUN_ROOT=/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100
root=$MOMART_RUN_ROOT/eval_online_combined_blend_250k_seed100_triplicate_20261028
export EVAL_METHOD=boxed_no_history EVAL_STEPS=250000 EVAL_EPISODES=30 NUM_ENVS=10
export PRECISION_CODE_ROOT=$repo
export PRECISION_BUNDLE=/home/jrw/vpwem_outputs/bbox_strategies_20261020/comparison_v2/detector_bundle.json
export VPWEM_PYTHON=/home/jrw/anaconda3/envs/vpwem/bin/python
export OPENBLAS_NUM_THREADS=1
mkdir -p "$root"
exec 8>"$root/.supervisor.lock"
flock -n 8 || { echo 'Supervisor already running'; exit 2; }
run() {
    local gpu=$1 seed=$2 label=$3
    local output=$root/$label
    mkdir -p "$output"
    GPU=$gpu EVAL_SEED=$seed MOMART_EVAL_ROOT=$output \
        python3 "$repo/tools/gpu_gate.py" run --gpu "$gpu" \
        --min-free 20000 --max-util 20 --stable 3 --poll 10 -- \
        bash "$repo/tools/run_momart_boxed_checkpoint_eval_bhr.sh" \
        > "$output/queue.log" 2>&1
}
echo "$(date -Is) START phase1 seed100 triplicate"
run 3 100 seed100_repeat1 & p1=$!
run 4 100 seed100_repeat2 & p2=$!
run 6 100 seed100_repeat3 & p3=$!
failed=0
wait "$p1" || failed=1
wait "$p2" || failed=1
wait "$p3" || failed=1
if (( failed )); then
    echo "$(date -Is) FAILED phase1; phase2 withheld"
    exit 1
fi
echo "$(date -Is) START phase2 seeds200_300"
run 3 200 seed200 & p1=$!
run 4 300 seed300 & p2=$!
wait "$p1" || failed=1
wait "$p2" || failed=1
echo "$(date -Is) FINISH failures=$failed"
exit "$failed"
