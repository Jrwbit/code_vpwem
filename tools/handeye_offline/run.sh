#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${CALIBRATION_PYTHON:-}" ]]; then
    if [[ -x "$HOME/miniconda3/envs/motor-inspection-1080ti/bin/python" ]]; then
        CALIBRATION_PYTHON="$HOME/miniconda3/envs/motor-inspection-1080ti/bin/python"
    elif [[ -x /data/rhz/conda-envs/ur5_data_collection/bin/python ]]; then
        CALIBRATION_PYTHON=/data/rhz/conda-envs/ur5_data_collection/bin/python
    else
        echo "Set CALIBRATION_PYTHON to an existing calibration interpreter" >&2
        exit 2
    fi
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$CALIBRATION_PYTHON" "$TOOL_DIR/calibrate.py" "$@"
