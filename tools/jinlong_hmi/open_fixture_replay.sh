#!/usr/bin/env bash
# Open a recorded simulation; this launcher never invokes robot control.
set -euo pipefail
clip=${1:-full}
case "$clip" in
    full|dryrun|insert_closeup) ;;
    current|left) ;;
    *) echo 'Usage: open_jinlong_fixture_replay.sh [full|dryrun|insert_closeup|current|left]' >&2; exit 2 ;;
esac
video="$HOME/jinlong-fixture-replay/20261016-compact/$clip.mp4"
if [ "$clip" = current ] || [ "$clip" = left ]; then
    video="$HOME/jinlong-fixture-replay/20261017-current-hover/hover.mp4"
    if [ "$clip" = left ]; then
        video="$HOME/jinlong-fixture-replay/20261017-left-layout/three_tool_cycle.mp4"
        # The camera/hover browser can become unresponsive. Give this replay
        # its own profile and instance instead of forwarding to that browser.
        profile="$HOME/snap/firefox/common/jinlong-left-layout-profile"
        mkdir -p "$profile"
        export MOZ_ENABLE_WAYLAND=1
        test -r "$video"
        exec firefox --new-instance --profile "$profile" --new-window "file://$video"
    fi
    # Keep the current study separate from an older camera-preview browser.
    profile="$HOME/snap/firefox/common/jinlong-hover-profile"
    mkdir -p "$profile"
    export MOZ_ENABLE_WAYLAND=1
    test -r "$video"
    exec firefox --profile "$profile" --new-window "file://$video"
fi
test -r "$video"
exec firefox --new-window "file://$video"
