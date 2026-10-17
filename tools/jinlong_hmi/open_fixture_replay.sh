#!/usr/bin/env bash
# Open a recorded simulation; this launcher never invokes robot control.
set -euo pipefail
clip=${1:-full}
case "$clip" in
    full|dryrun|insert_closeup) ;;
    current) ;;
    *) echo 'Usage: open_jinlong_fixture_replay.sh [full|dryrun|insert_closeup|current]' >&2; exit 2 ;;
esac
video="$HOME/jinlong-fixture-replay/20261016-compact/$clip.mp4"
if [ "$clip" = current ]; then
    video="$HOME/jinlong-fixture-replay/20261017-current-hover/hover.mp4"
fi
test -r "$video"
exec firefox --new-window "file://$video"
