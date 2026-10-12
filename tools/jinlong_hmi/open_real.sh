#!/usr/bin/env bash
# Start the adapted real-device UI. Actual commands still require UI confirmation.
set -euo pipefail
app="$HOME/code/jinlong-motor-arm-control-codex-send-stop"
state_dir="$HOME/.local/state/jinlong-hmi"
mkdir -p "$state_dir"
exec 9>"$state_dir/launcher.lock"
flock 9
if ! curl --fail --silent --max-time 2 http://127.0.0.1:50168/state >/dev/null; then
    # Another SDK connection can reset the motion queue: refuse a competing owner.
    if ss -tnp | grep -E '192\.168\.71\.16[012].*(RobotAssist|rokae_capture|xcore_)' >/dev/null; then
        echo "Disconnect the other controller client before opening the real HMI." >&2
        exit 1
    fi
    cd "$app"
    nohup bash "$app/helios_hmi/run_ipc.sh" --enable-write \
        --sessions "$HOME/helios_hmi_sessions/handeye_ipc" \
        >"$state_dir/real.log" 2>&1 9>&- </dev/null &
    hmi_pid=$!
    echo "$hmi_pid" >"$state_dir/real.pid"
    ready=0
    for attempt in {1..60}; do
        if ! kill -0 "$hmi_pid" 2>/dev/null; then
            cat "$state_dir/real.log" >&2
            exit 1
        fi
        if curl --fail --silent --max-time 2 http://127.0.0.1:50168/state >/dev/null; then
            ready=1
            break
        fi
        sleep 0.25
    done
    if [ "$ready" -ne 1 ]; then
        kill -TERM "$hmi_pid"
        echo "HMI startup timed out; see $state_dir/real.log" >&2
        exit 1
    fi
fi
"$app/.venv/bin/python" - <<'PY'
import json, urllib.request
with urllib.request.urlopen('http://127.0.0.1:50168/state', timeout=3) as response:
    state = json.load(response)
if state.get('revision') != 'codex-send-stop-ipc' or state.get('write_enabled') is not True:
    raise SystemExit('Port 50168 belongs to another mode; close it before starting real HMI.')
PY
flock -u 9
exec 9>&-
exec firefox --new-window http://127.0.0.1:50168/
