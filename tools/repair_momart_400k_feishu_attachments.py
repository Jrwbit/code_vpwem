#!/usr/bin/env python3
"""Bind videos to the empty Feishu file blocks from the first publish attempt."""

from __future__ import annotations

import argparse
import pathlib
import re
import time

from feishu_codex_bridge.bridge import Config, FeishuClient


PATTERN = re.compile(
    r"WARNING: 附件 (?P<name>\S+) 上传失败：.*?"
    r"'block_type': 33, 'children': \['(?P<block_id>[^']+)'\]"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--failure-log", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    args = parser.parse_args()

    text = args.failure_log.read_text(encoding="utf-8")
    mappings = PATTERN.findall(text)
    if not mappings:
        raise RuntimeError("no failed attachment mappings found")
    videos = {path.name: path for path in args.data_root.rglob("*.mp4")}
    client = FeishuClient(Config(require_webhook=False))
    completed = 0
    for name, block_id in mappings:
        path = videos.get(name)
        if path is None:
            raise FileNotFoundError(name)
        token = client.upload_file(path, block_id)
        client.replace_file(args.document_id, block_id, token)
        completed += 1
        print(f"attached {completed}/{len(mappings)} {name}", flush=True)
        time.sleep(0.4)
    if completed != len(videos):
        raise RuntimeError(
            f"attached {completed} files but local video count is {len(videos)}"
        )


if __name__ == "__main__":
    main()
