#!/usr/bin/env python3
"""Resumably upload a deterministic shard of dataset files to OSS.

Short-lived STS credentials are read from stdin so they are never written to
the command line or a file. Existing objects with the expected size are
skipped, which makes restarting a partially completed transfer safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import oss2


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--object-prefix", required=True)
    parser.add_argument("--pattern", default="*.hdf5")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--multipart-threads", type=int, default=4)
    parser.add_argument("--part-size-mib", type=int, default=64)
    parser.add_argument(
        "--daemon-log",
        type=Path,
        help="Fork after reading credentials and append output to this log.",
    )
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")

    credentials = json.load(sys.stdin)
    if credentials.get("Code") != "Success":
        raise RuntimeError("credential service did not return Success")
    if args.daemon_log is not None:
        pid = os.fork()
        if pid:
            print(json.dumps({"event": "daemonized", "pid": pid}), flush=True)
            return
        os.setsid()
        args.daemon_log.parent.mkdir(parents=True, exist_ok=True)
        log_fd = os.open(
            args.daemon_log,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, sys.stdin.fileno())
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())
        os.close(null_fd)
        os.close(log_fd)
        # Do not keep the SSH transport alive after the launcher exits. The
        # upload has not opened any network connections at this point.
        os.closerange(3, int(os.sysconf("SC_OPEN_MAX")))
    auth = oss2.StsAuth(
        credentials["AccessKeyId"],
        credentials["AccessKeySecret"],
        credentials["SecurityToken"],
    )
    bucket = oss2.Bucket(auth, args.endpoint, args.bucket)

    candidates = sorted(
        path for path in args.source_root.glob(args.pattern) if path.is_file()
    )
    selected = candidates[args.shard_index :: args.shard_count]
    print(
        json.dumps(
            {
                "event": "start",
                "all_files": len(candidates),
                "shard_files": len(selected),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    uploaded = skipped = uploaded_bytes = 0
    started = time.monotonic()
    for number, source in enumerate(selected, 1):
        relative = source.relative_to(args.source_root).as_posix()
        object_key = f"{args.object_prefix.rstrip('/')}/{relative}"
        size = source.stat().st_size
        try:
            meta = bucket.get_object_meta(object_key)
        except oss2.exceptions.NoSuchKey:
            meta = None
        if meta is not None and meta.content_length == size:
            skipped += 1
            print(
                f"SKIP {number}/{len(selected)} bytes={size} {relative}",
                flush=True,
            )
            continue

        source_md5 = file_md5(source)
        result = oss2.resumable_upload(
            bucket,
            object_key,
            str(source),
            headers={"x-oss-meta-source-md5": source_md5},
            multipart_threshold=100 * 1024 * 1024,
            part_size=args.part_size_mib * 1024 * 1024,
            num_threads=args.multipart_threads,
        )
        meta = bucket.get_object_meta(object_key)
        verified = (
            result.status == 200
            and meta.content_length == size
            and meta.headers.get("x-oss-meta-source-md5") == source_md5
        )
        if not verified:
            raise RuntimeError(f"verification failed for {source}")
        uploaded += 1
        uploaded_bytes += size
        elapsed = max(time.monotonic() - started, 1e-6)
        print(
            f"OK {number}/{len(selected)} bytes={size} "
            f"uploaded_gib={uploaded_bytes / 2**30:.3f} "
            f"rate_mib_s={uploaded_bytes / elapsed / 2**20:.2f} {relative}",
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "complete",
                "uploaded": uploaded,
                "skipped": skipped,
                "uploaded_bytes": uploaded_bytes,
                "shard_index": args.shard_index,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
