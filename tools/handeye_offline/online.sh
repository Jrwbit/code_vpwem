#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VISION_ROOT="${VISION_ROOT:-$HOME/r435-release/r435-vision-1080ti-v1}"
ROKAE_SDK="${ROKAE_SDK:-$HOME/sdk/xcore-0.7.1/Release/linux}"
export PYTHONPATH="$ROKAE_SDK:$VISION_ROOT/app:$VISION_ROOT/sdk/haikangSDK/Res/demo/python:${PYTHONPATH:-}"
HK_MVCC_LIB_DIR="${HK_MVCC_LIB_DIR:-$HOME/sdk/hik-mvcc}"
if [[ ! -d "$HK_MVCC_LIB_DIR" ]]; then
    HK_MVCC_LIB_DIR=/opt/MVS/lib/64
fi
export LD_LIBRARY_PATH="$VISION_ROOT/sdk/haikangSDK/Libs/linux64:$HK_MVCC_LIB_DIR:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
exec "${CALIBRATION_PYTHON:-$HOME/miniconda3/envs/motor-inspection-1080ti/bin/python}" "$TOOL_DIR/rokae_capture.py" "$@"
