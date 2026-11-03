#!/usr/bin/env bash
# Run once only after freezing the complete recurrent selector and comparators.
set -euo pipefail
project=${1:?project}
research=${2:?research}
python_bin=${3:?python}
prefix=${4:?prefix}
selection=${5:?frozen selection directory}
history_reference=${6:?training main history npz}
assigned=${MOMART_ASSIGNED_GPU:?explicit physical GPU required}
cd "$project"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$research/deps:$project"
exec 7>"$research/${prefix}_recurrent_pipeline.lock"
flock -n 7 || exit 0
"$python_bin" - "$selection" "$research/${prefix}_corpus" "$prefix" <<'PY'
import hashlib,json,sys
from pathlib import Path
selected=Path(sys.argv[1]);corpus=Path(sys.argv[2]);value=json.loads((selected/'selected.json').read_text())
assert value['holdout']==sys.argv[3] and value['frozen_main_history_contract']
expected=value['holdout_original_manifest_sha256']
assert hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest()==expected
assert hashlib.sha256((selected/'holdout_manifest.json').read_bytes()).hexdigest()==expected
assert not any((corpus/name).exists() for name in ['annotations_all.json','candidate_manifest.json'])
for path,sha in value['frozen_weight_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
PY
write_status() {
 "$python_bin" - "$research/${prefix}_recurrent_pipeline_status.json" "$1" <<'PY'
import json,os,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(stage=sys.argv[2],pid=os.getppid(),updated_unix=time.time()),indent=2))
PY
}
trap 'write_status failed' ERR
write_status preparing_main_public_features
bash experiments/momart_target_research_20261101/prepare_corpus.sh "$project" "$research" "$python_bin" "$prefix" "$assigned" geometry
write_status exporting_fixed_clock_main_history
CUDA_VISIBLE_DEVICES="$assigned" "$python_bin" -u experiments/momart_target_research_20261101/main_history.py --root "$(dirname "$research")" --config /home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101/gpu3.yaml --urdf /home/jrw/code/iGibson_momart/gibson2/data/assets/models/fetch/fetch_gripper.urdf --output "$research/${prefix}_main_history.npz" --corpus "$research/${prefix}_corpus" --reference-history "$history_reference" --device cuda:0 --batch-size 32
write_status frozen_confirmation
"$python_bin" -u experiments/momart_target_research_20261101/confirm.py --root "$(dirname "$research")" --selection "$selection" --control-selection "$research/frozen_for_confirm4" --prefix "$prefix" --purpose confirmation --public-views geometry --main-history "$research/${prefix}_main_history.npz" --output "$research/${prefix}_recurrent_confirmation"
write_status complete
