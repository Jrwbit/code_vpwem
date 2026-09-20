#!/usr/bin/env python3
"""Publish a local Markdown file as an editable Feishu document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from bridge import Bridge, Config, FeishuClient
except ImportError:  # pragma: no cover
    from .bridge import Bridge, Config, FeishuClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--folder-token", default="")
    parser.add_argument("--editor-id", default="")
    parser.add_argument("--editor-type", default="")
    parser.add_argument(
        "--attachment", action="append", default=[], type=Path,
        help="local file to append as a downloadable Feishu document attachment",
    )
    parser.add_argument("--inspect-conversion", action="store_true")
    args = parser.parse_args()

    path = args.markdown.expanduser().resolve()
    if not path.is_file():
        parser.error(f"file does not exist: {path}")
    if path.suffix.lower() not in {".md", ".markdown"}:
        parser.error("only Markdown files are supported")

    config = Config(require_webhook=False)
    client = FeishuClient(config)
    if args.inspect_conversion:
        converted = client.convert_markdown(path.read_text(encoding="utf-8"))
        print(json.dumps(converted, ensure_ascii=False, indent=2))
        return
    title = args.title.strip() or Bridge._markdown_title(path)
    editor_id = args.editor_id or config.default_manager_id
    editor_type = args.editor_type or config.default_manager_type
    attachments = [item.expanduser().resolve() for item in args.attachment]
    missing = [str(item) for item in attachments if not item.is_file()]
    if missing:
        parser.error(f"attachment does not exist: {', '.join(missing)}")
    url, warnings = client.publish_markdown(
        path,
        title,
        folder_token=args.folder_token,
        editor_id=editor_id,
        editor_type=editor_type,
        asset_root=Path.cwd(),
        attachments=attachments,
    )
    print(url)
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
