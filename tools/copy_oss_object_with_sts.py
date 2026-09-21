#!/usr/bin/env python3
"""Copy a verified large OSS object using short-lived credentials from stdin.

The caller must independently hash the source object before invoking this tool.
Only metadata and server-side multipart copy requests pass through this process;
the object payload is not downloaded to the client.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import oss2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--target-key", required=True)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--part-size-mib", type=int, default=128)
    args = parser.parse_args()

    if args.source_key == args.target_key:
        parser.error("source and target keys must differ")
    if args.expected_size <= 0 or args.part_size_mib <= 0:
        parser.error("expected size and part size must be positive")
    if len(args.source_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.source_sha256
    ):
        parser.error("source SHA256 must be 64 lowercase hexadecimal characters")

    credentials = json.load(sys.stdin)
    if credentials.get("Code") != "Success":
        raise RuntimeError("credential service did not return Success")
    auth = oss2.StsAuth(
        credentials["AccessKeyId"],
        credentials["AccessKeySecret"],
        credentials["SecurityToken"],
    )
    bucket = oss2.Bucket(auth, args.endpoint, args.bucket)
    source = bucket.get_object_meta(args.source_key)
    target = bucket.get_object_meta(args.target_key)
    if source.content_length != args.expected_size:
        raise RuntimeError("source size does not match expected size")
    if target.content_length != args.expected_size:
        raise RuntimeError("existing target size does not match expected size")

    part_size = args.part_size_mib * 1024 * 1024
    upload = bucket.init_multipart_upload(
        args.target_key,
        headers={"x-oss-meta-vpwem-source-sha256": args.source_sha256},
    )
    parts = []
    try:
        for number, start in enumerate(range(0, args.expected_size, part_size), 1):
            end = min(start + part_size, args.expected_size) - 1
            for attempt in range(4):
                try:
                    result = bucket.upload_part_copy(
                        args.bucket,
                        args.source_key,
                        (start, end),
                        args.target_key,
                        upload.upload_id,
                        number,
                    )
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            parts.append(oss2.models.PartInfo(number, result.etag))
            if number % 16 == 0:
                print(f"copied_parts={number}", flush=True)
        bucket.complete_multipart_upload(args.target_key, upload.upload_id, parts)
    except BaseException:
        bucket.abort_multipart_upload(args.target_key, upload.upload_id)
        raise

    copied = bucket.get_object_meta(args.target_key)
    if copied.content_length != args.expected_size:
        raise RuntimeError("copied target size does not match expected size")
    print(f"copied_bytes={copied.content_length} parts={len(parts)}", flush=True)


if __name__ == "__main__":
    main()
