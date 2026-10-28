#!/usr/bin/env python3
"""Attach this figure bundle and verify its Feishu document and image bytes.

Uses the already connected Feishu Docs plugin's credential client. This is
document work only; no bot messaging configuration or permission is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/momart_main_figures_20261028"
REPORT = ROOT / "reports/momart_main_figures_20261028.md"
DEFAULT_PLUGIN = Path.home() / ".codex/plugins/cache/personal/feishu-docs/0.1.0+codex.20260804031831/scripts/feishu_mcp.py"
NOTE = ("C 为高层视觉风格示意，精确连接与张量结构以 A/B 为准。C2 的框坐标标签是通用示意，实际实现使用归一化 (x1, y1, x2, y2)；"
        "图中条件箭头经过采样组的噪声区域，实际 3 个条件 token 直接供 Transformer cross-attention，9×10 噪声动作走独立动作输入，两者不拼接。图中场景均不是实际 rollout 截图。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document_id")
    parser.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--insert-note", action="store_true")
    parser.add_argument("--attach", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("connected_feishu_docs", args.plugin)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    client = plugin.FeishuClient()
    doc = args.document_id
    base = plugin.API_BASE

    def api(method, endpoint, **kwargs):
        response = requests.request(method, base + endpoint, headers=client.headers(),
                                    timeout=60, **kwargs)
        return plugin.check_response(response).get("data", {})

    children = client.list_children(doc)
    raw = api("GET", f"/docx/v1/documents/{doc}/raw_content")["content"]
    if args.insert_note and NOTE not in raw:
        image_positions = [i for i, b in enumerate(children) if "image" in b]
        assert len(image_positions) == 8
        # The seventh image is the boxed imagegen figure; insert its exact caveat next.
        api("POST", f"/docx/v1/documents/{doc}/blocks/{doc}/children",
            params={"document_revision_id": -1},
            json={"children": [plugin.text_block("text", 2, NOTE)],
                  "index": image_positions[6] + 1})
        print("Inserted C2 figure note", flush=True)

    archive = OUT / "MoMaRT_main_figures_20261028.zip"
    if args.attach:
        children = client.list_children(doc)
        # Deliberately one attachment upload; subsequent runs should omit --attach.
        created = client.create_children(doc, [{"block_type": 23, "file": {}}])
        block = created[0]
        file_block_id = block["block_id"] if block["block_type"] == 23 else block["children"][0]
        with archive.open("rb") as file:
            upload = api("POST", "/drive/v1/medias/upload_all",
                         data={"file_name": archive.name, "parent_type": "docx_file",
                               "parent_node": file_block_id, "size": str(archive.stat().st_size)},
                         files={"file": (archive.name, file, "application/zip")})
        api("PATCH", f"/docx/v1/documents/{doc}/blocks/{file_block_id}",
            params={"document_revision_id": -1},
            json={"replace_file": {"token": upload["file_token"]}})
        print("Uploaded actual ZIP attachment", flush=True)

    raw = api("GET", f"/docx/v1/documents/{doc}/raw_content")["content"]
    for phrase in ["64.67%", "194/300", "72.33%", "C 为高层视觉风格示意", "两者不拼接"]:
        assert phrase in raw, f"Missing published content: {phrase}"
    children = client.list_children(doc)
    images = [b for b in children if "image" in b]
    expected = [(REPORT.parent / path).resolve() for path in
                re.findall(r"!\[[^\]]*\]\(([^)]+)\)", REPORT.read_text())]
    assert len(images) == len(expected) == 8
    readback = OUT / "feishu_readback"
    readback.mkdir(exist_ok=True)

    def download(token, destination):
        response = requests.get(base + f"/drive/v1/medias/{token}/download",
                                headers=client.headers(), timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return hashlib.sha256(response.content).hexdigest()

    verified = []
    for block, local in zip(images, expected):
        remote_hash = download(block["image"]["token"], readback / local.name)
        local_hash = hashlib.sha256(local.read_bytes()).hexdigest()
        assert remote_hash == local_hash, f"Image readback mismatch: {local.name}"
        verified.append({"name": local.name, "sha256": local_hash, "matched": True})
        print("Image matched: " + local.name, flush=True)

    # File blocks may be nested inside a view block, so enumerate the document.
    all_blocks = []
    cursor = ""
    while True:
        params = {"page_size": 500}
        if cursor:
            params["page_token"] = cursor
        page = api("GET", f"/docx/v1/documents/{doc}/blocks", params=params)
        all_blocks.extend(page["items"])
        if not page.get("has_more"):
            break
        cursor = page["page_token"]
    attachments = [b for b in all_blocks if b.get("file", {}).get("name") == archive.name]
    assert len(attachments) == 1, f"Expected one named archive, got {len(attachments)}"
    archive_hash = download(attachments[0]["file"]["token"], readback / archive.name)
    assert archive_hash == hashlib.sha256(archive.read_bytes()).hexdigest()
    result = {"url": f"https://feishu.cn/docx/{doc}", "body_verified": True,
              "images_verified": verified, "attachment_verified": True,
              "attachment_sha256": archive_hash}
    (OUT / "feishu_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"body_verified": True, "images_verified": len(verified),
                      "attachment_verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
