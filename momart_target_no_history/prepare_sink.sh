#!/usr/bin/env bash
# Reproduce the fixed sink extension without touching any existing reserve.
set -euo pipefail
project=${1:-/home/jrw/code_vpwem_momart_association_20261030}
research=${2:-/home/jrw/vpwem_outputs/momart_target_research_20261101}
python_bin=${3:-/home/jrw/vpwem_envs/detector_compare/bin/python}
gpu=${4:-1}
cd "$project"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$research/deps:$project"
corpus="$research/sink_extension_20261103_corpus"
if [[ ! -f "$corpus/manifest.json" ]]; then
 "$python_bin" -m momart_target_no_history.sample_sink --data-root /home/jrw/vpwem_data/momart --output "$corpus"
fi
"$python_bin" - "$research/sink_extension_20261103_vocabulary.json" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]);value=dict(tasks=['table_cleanup_to_dishwasher_expert','table_setup_from_dishwasher_expert','table_setup_from_dresser_expert','unload_dishwasher_to_dresser_suboptimal','table_cleanup_to_sink_expert','table_cleanup_to_sink_suboptimal'])
if path.exists():assert json.loads(path.read_text())==value
else:path.write_text(json.dumps(value,indent=2))
PY
exec 8>"$research/render.lock"
flock 8
if [[ ! -f "$corpus/annotations_all.json" ]]; then
 "$python_bin" -u experiments/detector_comparison_20261030/annotate.py --corpus "$corpus" --gpu "$gpu" --reject-unaligned
fi
if [[ ! -f "$research/sink_extension_20261103_events.json" ]]; then
 "$python_bin" -u experiments/momart_target_selector_20261101/export_targets.py --corpus "$corpus" --output "$research/sink_extension_20261103_events.json" --render-gpu "$gpu"
fi
if [[ ! -f "$research/sink_extension_20261103_phases.json" ]]; then
 "$python_bin" -u experiments/momart_stage_target_20261101/export_phases.py --corpus "$corpus" --events "$research/sink_extension_20261103_events.json" --output "$research/sink_extension_20261103_phases.json" --gpu "$gpu"
fi
flock -u 8
if [[ ! -f "$research/sink_extension_20261103_features.npz" ]]; then
 CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u experiments/momart_target_selector_20261101/export_features.py --corpus "$corpus" --config /home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101/gpu3.yaml --task-vocabulary "$research/sink_extension_20261103_vocabulary.json" --output "$research/sink_extension_20261103_features.npz" --device cuda:0 --batch-size 16
fi
if [[ ! -f "$research/geometry_sink_extension_20261103.npz" ]]; then
 "$python_bin" -u experiments/momart_target_research_20261101/robot_geometry.py --corpus "$corpus" --features "$research/sink_extension_20261103_features.npz" --urdf /home/jrw/code/iGibson_momart/gibson2/data/assets/models/fetch/fetch_gripper.urdf --output "$research/geometry_sink_extension_20261103.npz"
fi
