#!/usr/bin/env bash
set -euo pipefail
repo=/home/wjr/code/code_vpwem
state=/home/wjr/.local/state/momart-report-20261029
codex_bin=/home/wjr/.vscode/extensions/openai.chatgpt-26.1007.21434-linux-x64/bin/linux-x86_64/codex
mkdir -p "$state"
exec 9>"$state/run.lock"
flock -n 9 || exit 1
export TZ=Asia/Shanghai
printf '%s START\n' "$(date -Is)" >> "$state/status.log"
rc=0
"$codex_bin" exec --sandbox danger-full-access -c 'approval_policy="never"' \
    -C "$repo" --color never -o "$state/final.txt" - \
    < "$repo/tools/scheduled_reports/momart_20261029_0800.prompt.md" \
    > "$state/run.log" 2>&1 || rc=$?
printf '%s EXIT %s\n' "$(date -Is)" "$rc" >> "$state/status.log"
exit "$rc"
