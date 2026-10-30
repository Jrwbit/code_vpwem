#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "${here}/.." && pwd)
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export GIBSON_DEVICE_ID="${GPU:-0}"
export PYTHONUNBUFFERED=1

exec "${VPWEM_PYTHON:-python}" -m momart_no_history_baseline.evaluate "$@"
