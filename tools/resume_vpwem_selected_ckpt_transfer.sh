#!/usr/bin/env bash
set -uo pipefail

MANIFEST=${MANIFEST:-/tmp/vpwem_selected_ckpt_missing.tsv}
FULL_MANIFEST=${FULL_MANIFEST:-/tmp/vpwem_selected_ckpt_manifest.tsv}
TARGET_HOST=${TARGET_HOST:-kai_shihao_sh2}
TARGET_ROOT=${TARGET_ROOT:-/oss/sh2/vpwem_checkpoints}
SOURCE_ROOT=${SOURCE_ROOT:-/home/jrw}
WORKERS=${WORKERS:-6}
LOG_DIR=${LOG_DIR:-/tmp/vpwem_selected_ckpt_resume2_logs}

mkdir -p "$LOG_DIR"

source_ssh() {
    ssh -o BatchMode=yes \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -o UserKnownHostsFile=/tmp/bhr_socks_known_hosts \
        -o StrictHostKeyChecking=accept-new \
        -o 'ProxyCommand=nc -x 127.0.0.1:40659 %h %p' \
        -p 22022 jrw@127.0.0.1 "$@"
}

target_ssh() {
    ssh -o BatchMode=yes \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        "$TARGET_HOST" "$@"
}

release_partial_writers() {
    local partial=$1
    target_ssh "for fd in /proc/[0-9]*/fd/1; do
        link=\$(readlink \"\$fd\" 2>/dev/null || true)
        if [[ \"\$link\" == '$partial' ]]; then
            pid=\${fd#/proc/}; pid=\${pid%%/*}
            kill -TERM \"\$pid\" 2>/dev/null || true
        fi
    done"
}

mapfile -t entries < "$MANIFEST"

transfer_worker() {
    local worker=$1 index=$2
    local entry expected rel source target partial target_dir have offset status
    for ((; index<${#entries[@]}; index+=WORKERS)); do
        entry=${entries[index]}
        expected=${entry%%$'\t'*}
        rel=${entry#*$'\t'}
        source=$SOURCE_ROOT/$rel
        target=$TARGET_ROOT/$rel
        partial=$target.partial
        target_dir=${target%/*}
        while true; do
            have=$(target_ssh "stat -c %s '$target' 2>/dev/null || echo 0")
            if [[ "$have" == "$expected" ]]; then
                printf '%s SKIP %s %s\n' "$(date -Is)" "$expected" "$rel" \
                    >>"$LOG_DIR/worker${worker}.log"
                break
            fi

            have=$(target_ssh \
                "mkdir -p '$target_dir'; stat -c %s '$partial' 2>/dev/null || echo 0")
            if (( have > expected )); then
                target_ssh "rm -f '$partial'"
                have=0
            fi
            if (( have == expected )); then
                target_ssh "mv '$partial' '$target'"
                continue
            fi

            # A source-side SSH timeout can leave the target-side `cat`
            # holding the OSS object open. This worker is the sole owner of
            # the path, so clear any stale writer before resuming it.
            release_partial_writers "$partial"
            offset=$((have + 1))
            printf '%s RESUME have=%s total=%s %s\n' \
                "$(date -Is)" "$have" "$expected" "$rel" \
                >>"$LOG_DIR/worker${worker}.log"
            set -o pipefail
            source_ssh "tail -c +$offset '$source'" | \
                timeout --signal=TERM 3700 \
                ssh -o BatchMode=yes \
                    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
                    "$TARGET_HOST" "timeout --signal=TERM 3600 cat >> '$partial'"
            status=$?
            set +o pipefail
            if (( status != 0 )); then
                release_partial_writers "$partial"
                printf '%s RETRY status=%s %s\n' \
                    "$(date -Is)" "$status" "$rel" \
                    >>"$LOG_DIR/worker${worker}.log"
                sleep 15
                continue
            fi

            have=$(target_ssh "stat -c %s '$partial' 2>/dev/null || echo 0")
            if [[ "$have" == "$expected" ]]; then
                target_ssh "mv '$partial' '$target'"
                printf '%s DONE %s %s\n' "$(date -Is)" "$expected" "$rel" \
                    >>"$LOG_DIR/worker${worker}.log"
                break
            fi
            sleep 5
        done
    done
}

pids=()
for ((worker=0; worker<WORKERS; worker++)); do
    transfer_worker "$((worker + 1))" "$worker" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"

printf '%s transfer complete; generating source checksums\n' "$(date -Is)" \
    >>"$LOG_DIR/supervisor.log"
source_ssh "cd '$SOURCE_ROOT'; while IFS=\$'\\t' read -r size rel; do sha256sum \"\$rel\"; done" \
    < "$FULL_MANIFEST" > /tmp/vpwem_selected_ckpt_SHA256SUMS

scp -q /tmp/vpwem_selected_ckpt_SHA256SUMS \
    "$TARGET_HOST:$TARGET_ROOT/SHA256SUMS.new"
target_ssh \
    "test -s '$TARGET_ROOT/SHA256SUMS.new' && mv '$TARGET_ROOT/SHA256SUMS.new' '$TARGET_ROOT/SHA256SUMS' && cd '$TARGET_ROOT' && sha256sum -c SHA256SUMS > VERIFY.log 2>&1 && touch VERIFIED"

printf '%s all files verified\n' "$(date -Is)" >>"$LOG_DIR/supervisor.log"
