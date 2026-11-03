#!/usr/bin/env bash
set -euo pipefail
project=/home/jrw/code_vpwem_momart_association_20261030
research=/home/jrw/vpwem_outputs/momart_target_research_20261101
cd "$project"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$research/deps:$project"
exec /home/jrw/vpwem_envs/detector_compare/bin/python -u -m momart_target_no_history.prepare_cache \
 --config momart_object_memory/config.yaml \
 --perception-root /home/jrw/vpwem_data/momart_objects_mask8_multigpu_20261101 \
 --selector-bundle "$research/unified_five_task_20261103/bundle_five_task" \
 --urdf /home/jrw/code/iGibson_momart/gibson2/data/assets/models/fetch/fetch_gripper.urdf \
 --output /home/jrw/vpwem_data/momart_target_unified_five_task_20261103 \
 --device cuda:0 --batch-size 128 --workers 4
