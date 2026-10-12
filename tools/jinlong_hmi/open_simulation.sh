#!/usr/bin/env bash
# Open the supplied send-stop release on the bit-bhr desktop.
set -euo pipefail
app="$HOME/code/jinlong-motor-arm-control-codex-send-stop"
state_dir="$HOME/.local/state/jinlong-hmi"
mkdir -p "$state_dir"
exec 9>"$state_dir/launcher.lock"
flock 9
if ! curl --fail --silent --max-time 2 http://127.0.0.1:50168/state >/dev/null; then
    cd "$app"
    # Real hardware requires a separate x86_64 SDK adaptation.
    unset HELIOS_HMI_ALLOW_REAL
    nohup "$app/.venv/bin/python" -m helios_hmi --fake --no-camera \
        --host 127.0.0.1 --port 50168 >"$state_dir/simulation.log" 2>&1 9>&- </dev/null &
    hmi_pid=$!
    echo "$hmi_pid" >"$state_dir/simulation.pid"
    ready=0
    for attempt in {1..40}; do
        if ! kill -0 "$hmi_pid" 2>/dev/null; then
            cat "$state_dir/simulation.log" >&2
            exit 1
        fi
        if curl --fail --silent --max-time 2 http://127.0.0.1:50168/state >/dev/null; then
            ready=1
            break
        fi
        sleep 0.25
    done
    if [ "$ready" -ne 1 ]; then
        kill "$hmi_pid"
        echo "HMI startup timed out; see $state_dir/simulation.log" >&2
        exit 1
    fi
fi
flock -u 9
exec 9>&-
if command -v firefox >/dev/null; then
    exec firefox --new-window http://127.0.0.1:50168/
fi
exec xdg-open http://127.0.0.1:50168/
