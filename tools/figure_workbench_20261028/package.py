#!/usr/bin/env python3
"""Create a deliberate source + figure bundle, excluding credentials and caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/assets/momart_main_figures_20261028"
OUT = ROOT / "artifacts/momart_main_figures_20261028"


def main():
    from PIL import Image
    files = {}
    for folder, pattern in [("native", "*.svg"), ("drawio", "*.drawio"), ("prompts", "*.txt")]:
        for path in sorted((SOURCE / folder).glob(pattern)):
            files[f"{folder}/{path.name}"] = path
    for name in ["README.md", "evidence.md", "tool_runs.json"]:
        files[name] = SOURCE / name
    report = (ROOT / "reports/momart_main_figures_20261028.md").read_text()
    bundle_report = OUT / "bundle_report.md"
    bundle_report.write_text(report.replace("../artifacts/momart_main_figures_20261028/", ""))
    files["report.md"] = bundle_report
    for name in ["render_native.py", "next_drawio_bridge.py", "build_gallery.py", "package.py", "verify_feishu.py"]:
        files[f"scripts/{name}"] = Path(__file__).parent / name
    for scheme in ["native", "drawio", "imagegen"]:
        for method in ["no_history", "boxed"]:
            name = f"{scheme}_{method}.png"
            files[name] = OUT / name
    for name in ["native_no_history.pdf", "native_boxed.pdf", "drawio_no_history.svg",
                 "drawio_boxed.svg", "boxed_input_reference.png", "comparison.png", "preview.html"]:
        files[name] = OUT / name
    for name in ["imagegen_no_history_v1.png", "imagegen_boxed_v1.png",
                 "imagegen_no_history_v2.png", "imagegen_boxed_v2.png"]:
        files[f"iterations/{name}"] = OUT / name
    manifest = []
    for name, path in files.items():
        raw = path.read_bytes()
        item = {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if path.suffix.lower() == ".png":
            with Image.open(path) as im:
                item["dimensions"] = list(im.size)
        manifest.append(item)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    archive = OUT / "MoMaRT_main_figures_20261028.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in files.items():
            z.write(path, name)
        z.write(OUT / "manifest.json", "manifest.json")
        z.writestr("iterations/README.txt", "These are superseded imagegen drafts for revision review. Use the six PNGs at archive root as the selected versions. Drafts contain the issues discussed in the prompts and report.\n")
    with ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "files": len(files), "zip_integrity": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
