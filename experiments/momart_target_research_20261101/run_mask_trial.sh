#!/usr/bin/env bash
set -euo pipefail
project=${1:?repository root}
research=${2:?research output root}
python_bin=${3:?python executable}
mode=${4:-sparse}
extra_args=()
output="$research/mask_seed100_300"
regression="$research/mask_regression1"
if [ "$mode" = dense ]; then
 extra_args=(--dense)
 output="$research/mask_dense_seed100_300"
 regression="$research/mask_dense_regression1"
fi
cd "$project"
exec 9>"$research/mask_trial.lock"
flock -n 9 || exit 0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$research/deps:."
while [ ! -f "$research/mask_old.npz" ] || [ ! -f "$research/mask_old.json" ] || [ ! -f "$research/mask_new.npz" ] || [ ! -f "$research/mask_new.json" ]; do sleep 15; done
if [ "$mode" = dense ]; then
 while [ ! -f "$research/mask_dense.npz" ] || [ ! -f "$research/mask_dense.json" ]; do sleep 15; done
fi
"$python_bin" -u experiments/momart_target_research_20261101/mask_benchmark.py --root "$(dirname "$research")" --output "$output" "${extra_args[@]}"
while [ ! -f "$research/mask_confirmation.npz" ] || [ ! -f "$research/mask_confirmation.json" ]; do sleep 15; done
# First confirmation has already been examined: this evaluation is explicitly regression.
"$python_bin" -u experiments/momart_target_research_20261101/confirm.py --root "$(dirname "$research")" --selection "$output" --output "$regression" --purpose regression
