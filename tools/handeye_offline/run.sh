#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CALIBRATION_PYTHON="${CALIBRATION_PYTHON:-/data/rhz/conda-envs/ur5_data_collection/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
exec "$CALIBRATION_PYTHON" "$TOOL_DIR/calibrate.py" "$@"
