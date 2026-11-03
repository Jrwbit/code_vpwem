#!/usr/bin/env bash
# Main-view confirmation for frozen stateless selectors; no wrist enrichment.
set -euo pipefail
project=${1:?project}
research=${2:?research}
python_bin=${3:?python}
prefix=${4:?prefix}
selection=${5:?frozen selection directory}
assigned=${MOMART_ASSIGNED_GPU:?explicit physical GPU required}
cd "$project"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$research/deps:$project"
exec 7>"$research/${prefix}_frozen_pipeline.lock"
flock -n 7 || exit 0
"$python_bin" - "$selection" "$research/${prefix}_corpus" "$prefix" <<'PY'
import hashlib,json,sys
from pathlib import Path
selected=Path(sys.argv[1]);corpus=Path(sys.argv[2]);value=json.loads((selected/'selected.json').read_text())
assert value['holdout']==sys.argv[3]
assert not value.get('frozen_main_history_contract'),'Use recurrent confirmation pipeline for stateful selectors'
expected=value['holdout_original_manifest_sha256']
assert hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest()==expected
assert hashlib.sha256((selected/'holdout_manifest.json').read_bytes()).hexdigest()==expected
assert not any((corpus/name).exists() for name in ['annotations_all.json','candidate_manifest.json'])
for path,sha in value['frozen_weight_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
PY
write_status() {
 "$python_bin" - "$research/${prefix}_frozen_pipeline_status.json" "$1" <<'PY'
import json,os,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(stage=sys.argv[2],pid=os.getppid(),updated_unix=time.time()),indent=2))
PY
}
trap 'write_status failed' ERR
write_status preparing_main_public_features
bash experiments/momart_target_research_20261101/prepare_corpus.sh "$project" "$research" "$python_bin" "$prefix" "$assigned" geometry
write_status frozen_confirmation
"$python_bin" -u experiments/momart_target_research_20261101/confirm.py --root "$(dirname "$research")" --selection "$selection" --control-selection "$research/frozen_for_confirm4" --prefix "$prefix" --purpose confirmation --public-views geometry --output "$research/${prefix}_frozen_confirmation"
write_status complete
