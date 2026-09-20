#!/usr/bin/env python3
"""Upload one file to OSS using short-lived STS credentials read from stdin."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import oss2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--object-key", required=True)
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
    started = time.monotonic()
    last_report = started

    def progress(consumed: int, total: int) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 10 or consumed == total:
            rate = consumed / max(now - started, 1e-6) / (1024 * 1024)
            print(
                f"progress={consumed}/{total} rate_mib_s={rate:.2f}",
                flush=True,
            )
            last_report = now

    result = bucket.put_object_from_file(
        args.object_key, args.source, progress_callback=progress
    )
    md5 = hashlib.md5()
    with open(args.source, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(block)
    print(
        json.dumps(
            {
                "status": result.status,
                "request_id": result.request_id,
                "etag": result.etag.strip('"').lower(),
                "source_md5": md5.hexdigest(),
                "object_key": args.object_key,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
