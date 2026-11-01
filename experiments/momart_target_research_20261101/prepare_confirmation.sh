#!/usr/bin/env bash
set -euo pipefail
project=${1:?repository root required}
research=${2:?research output root required}
python_bin=${3:?python executable required}
cd "$project"
exec 9>"$research/confirmation_prepare.lock"
flock -n 9 || exit 0
corpus="$research/confirmation_corpus"
status="$research/confirmation_prepare_status.json"
write_status() {
 "$python_bin" - "$status" "$1" "${2:-}" <<'PY'
import json,os,sys,time
from pathlib import Path
path=Path(sys.argv[1]);temp=path.with_suffix('.tmp')
temp.write_text(json.dumps(dict(stage=sys.argv[2],gpu=sys.argv[3],updated_unix=time.time(),pid=os.getppid()),indent=2));temp.replace(path)
PY
}
trap 'write_status failed "${chosen_gpu:-}"' ERR
write_status waiting_for_idle_gpu
chosen_gpu=''
while [ -z "$chosen_gpu" ]; do
 chosen_gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '$1 < 7 && $2 < 2500 && $3 <= 5 {gsub(/ /,"",$1);print $1;exit}')
 if [ -z "$chosen_gpu" ]; then sleep 30; fi
 done
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
write_status annotating "$chosen_gpu"
"$python_bin" -u experiments/detector_comparison_20261030/annotate.py --corpus "$corpus" --gpu "$chosen_gpu" --reject-unaligned
write_status motion_proxies "$chosen_gpu"
"$python_bin" -u experiments/momart_target_selector_20261101/export_targets.py --corpus "$corpus" --output "$research/confirmation_events.json" --render-gpu "$chosen_gpu"
write_status holding_proxies "$chosen_gpu"
"$python_bin" -u experiments/momart_stage_target_20261101/export_phases.py --corpus "$corpus" --events "$research/confirmation_events.json" --output "$research/confirmation_phases.json" --gpu "$chosen_gpu"
write_status public_features "$chosen_gpu"
CUDA_VISIBLE_DEVICES="$chosen_gpu" "$python_bin" -u experiments/momart_target_selector_20261101/export_features.py --corpus "$corpus" --config /home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101/gpu3.yaml --task-vocabulary /home/jrw/vpwem_outputs/momart_target_selector_20261101/features.json --output "$research/confirmation_features.npz"
write_status wrist_depth_cpu "$chosen_gpu"
CUDA_VISIBLE_DEVICES='' "$python_bin" -u experiments/momart_target_research_20261101/export_extra.py --corpus "$corpus" --features "$research/confirmation_features.npz" --config /home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101/gpu3.yaml --output "$research/extra_confirmation.npz"
write_status ready "$chosen_gpu"
