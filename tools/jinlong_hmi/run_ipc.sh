#!/usr/bin/env bash
# bit-bhr site launcher; preserve the groups enabled on the deployed 2026-10-16 UI.
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
vision=${VISION_ROOT:-$HOME/r435-release/r435-vision-1080ti-v1}
export PATH="$HOME/.local/bin:$PATH"
export HK_PY=${HK_PY:-$vision/sdk/haikangSDK/Res/demo/python}
export HMI_ROBOT_LOCAL_IP=${HMI_ROBOT_LOCAL_IP:-192.168.71.157}
export HMI_CAMERA_LOCAL_IP=${HMI_CAMERA_LOCAL_IP:-192.168.1.68}
export HMI_CAMERA_SERIAL=${HMI_CAMERA_SERIAL:-00DA8664418}
export HELIOS_HMI_ALLOWED_GROUPS=${HELIOS_HMI_ALLOWED_GROUPS:-left_arm,waist,right_arm}
export HELIOS_HMI_DRAG_ONLY_GROUPS=${HELIOS_HMI_DRAG_ONLY_GROUPS:-}
export HELIOS_HMI_REVISION=codex-send-stop-ipc
export LD_LIBRARY_PATH="$vision/sdk/haikangSDK/Libs/linux64:${HK_MVCC_LIB_DIR:-$HOME/sdk/hik-mvcc}:${LD_LIBRARY_PATH:-}"
test -x "$app/helios_hmi/native/xcore_state_stream"
test -x "$app/helios_hmi/native/xcore_motion"
cd "$app"
exec "${HMI_PYTHON:-$app/.venv/bin/python}" -m helios_hmi --host 127.0.0.1 --port 50168 "$@"
