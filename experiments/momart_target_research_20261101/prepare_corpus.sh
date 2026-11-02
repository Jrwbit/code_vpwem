#!/usr/bin/env bash
# Rendering waits for an idle GPU; detector defaults to CPU, or an explicitly idle GPU.
set -euo pipefail
project=${1:?repository root}
research=${2:?research output root}
python_bin=${3:?python executable}
name=${4:?corpus prefix, e.g. dense or confirm2}
feature_gpu=${5:-cpu}
cd "$project"
exec 9>"$research/${name}_prepare.lock"
flock -n 9 || exit 0
corpus="$research/${name}_corpus"
status="$research/${name}_prepare_status.json"
write_status() {
 "$python_bin" - "$status" "$1" <<'PY'
import json,os,sys,time
from pathlib import Path
path=Path(sys.argv[1]);temp=path.with_suffix('.tmp')
temp.write_text(json.dumps(dict(stage=sys.argv[2],updated_unix=time.time(),pid=os.getppid()),indent=2));temp.replace(path)
PY
}
trap 'write_status failed' ERR
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
write_status waiting_for_idle_gpu
chosen_gpu=''
while [ -z "$chosen_gpu" ]; do
 chosen_gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '$1 < 7 && $2 < 2500 && $3 <= 5 {gsub(/ /,"",$1);print $1;exit}')
 if [ -z "$chosen_gpu" ]; then sleep 30; fi
 done
# Serialize research renderers even if multiple preparation queues wake together.
exec 8>"$research/render.lock"
flock 8
write_status annotating
"$python_bin" -u experiments/detector_comparison_20261030/annotate.py --corpus "$corpus" --gpu "$chosen_gpu" --reject-unaligned
write_status motion_proxies
"$python_bin" -u experiments/momart_target_selector_20261101/export_targets.py --corpus "$corpus" --output "$research/${name}_events.json" --render-gpu "$chosen_gpu"
write_status holding_proxies
"$python_bin" -u experiments/momart_stage_target_20261101/export_phases.py --corpus "$corpus" --events "$research/${name}_events.json" --output "$research/${name}_phases.json" --gpu "$chosen_gpu"
flock -u 8
if [ "$feature_gpu" = cpu ]; then
 write_status public_features_cpu
 CUDA_VISIBLE_DEVICES='' "$python_bin" -u experiments/momart_target_selector_20261101/export_features.py --corpus "$corpus" --config /home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101/gpu3.yaml --task-vocabulary /home/jrw/vpwem_outputs/momart_target_selector_20261101/features.json --output "$research/${name}_features.npz" --device cpu --batch-size 4
else
 write_status "waiting_for_feature_gpu${feature_gpu}"
 ready=''
 while [ -z "$ready" ]; do
  ready=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, -v target="$feature_gpu" '$1 == target && $2 < 2500 && $3 <= 5 {print "yes"}')
  if [ -z "$ready" ]; then sleep 15; fi
 done
 write_status "public_features_gpu${feature_gpu}_batch16"
 CUDA_VISIBLE_DEVICES="$feature_gpu" "$python_bin" -u experiments/momart_target_selector_20261101/export_features.py --corpus "$corpus" --config /home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101/gpu3.yaml --task-vocabulary /home/jrw/vpwem_outputs/momart_target_selector_20261101/features.json --output "$research/${name}_features.npz" --device cuda:0 --batch-size 16
fi
write_status wrist_depth_cpu
CUDA_VISIBLE_DEVICES='' "$python_bin" -u experiments/momart_target_research_20261101/export_extra.py --corpus "$corpus" --features "$research/${name}_features.npz" --config /home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101/gpu3.yaml --output "$research/extra_${name}.npz"
write_status robot_geometry_cpu
"$python_bin" -u experiments/momart_target_research_20261101/robot_geometry.py --corpus "$corpus" --features "$research/${name}_features.npz" --urdf /home/jrw/code/iGibson_momart/gibson2/data/assets/models/fetch/fetch_gripper.urdf --output "$research/geometry_${name}.npz"
write_status ready_cpu_features
