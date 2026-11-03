#!/usr/bin/env bash
# All three experiments are launched only from a committed/pushed deployment.
set -euo pipefail
project=${1:?deployed source root}
python_bin=${2:-/home/jrw/vpwem_envs/detector_compare/bin/python}
cd "$project"
export MOMART_SOURCE_COMMIT
MOMART_SOURCE_COMMIT=$(cat "$project/.source_commit")
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="/home/jrw/vpwem_outputs/momart_target_research_20261101/deps:$project"
output=/home/jrw/vpwem_outputs/momart_new_target_three_arms_bs256_250k_20261103
mkdir -p "$output"
variants=(box recent8 stride20_8)
gpus=(5 6 7)
for i in 0 1 2; do
 variant=${variants[$i]}
 gpu=${gpus[$i]}
 # Each trainer holds its own exclusive run lock, preventing duplicate writers.
 CUDA_VISIBLE_DEVICES="$gpu" nohup "$python_bin" -u -m momart_target_history.train --config "$project/momart_target_history/config_$variant.yaml" > "$output/$variant.log" 2>&1 < /dev/null &
 echo "$variant GPU$gpu PID$!"
done
