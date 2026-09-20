#!/usr/bin/env python3
"""Upload a deterministic shard of checkpoint files using STS credentials on stdin."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import oss2


CHECKPOINT_RE = re.compile(r"^(?:model|scheduler)_(\d+)\.pt$")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--object-prefix", required=True)
    parser.add_argument("--min-step", type=int, default=400_000)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--multipart-threads", type=int, default=1)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--end-position", type=int)
    args = parser.parse_args()

    credentials = json.load(sys.stdin)
    if credentials.get("Code") != "Success":
        raise RuntimeError("credential service did not return Success")
    auth = oss2.StsAuth(
        credentials["AccessKeyId"],
        credentials["AccessKeySecret"],
        credentials["SecurityToken"],
    )
    bucket = oss2.Bucket(auth, args.endpoint, args.bucket)

    candidates: list[Path] = []
    for path in args.source_root.rglob("*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if match and int(match.group(1)) >= args.min_step:
            candidates.append(path)
    candidates.sort()
    bounded = candidates[args.start_position : args.end_position]
    selected = bounded[args.shard_index :: args.shard_count]
    print(
        json.dumps(
            {
                "event": "start",
                "all_files": len(candidates),
                "bounded_files": len(bounded),
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
        if args.multipart_threads > 1 and size >= 100 * 1024 * 1024:
            result = oss2.resumable_upload(
                bucket,
                object_key,
                str(source),
                headers={"x-oss-meta-source-md5": source_md5},
                multipart_threshold=100 * 1024 * 1024,
                part_size=32 * 1024 * 1024,
                num_threads=args.multipart_threads,
            )
            meta = bucket.get_object_meta(object_key)
            verified = (
                result.status == 200
                and meta.content_length == size
                and meta.headers.get("x-oss-meta-source-md5") == source_md5
            )
        else:
            result = bucket.put_object_from_file(object_key, str(source))
            etag = result.etag.strip('"').lower()
            verified = result.status == 200 and etag == source_md5
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
