"""Render a private deployment without committing its financial dataset.

The template and this builder are source files. The supplied JSON and generated
dist directory are local build inputs/output and must stay outside Git.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--revision", type=Path, help="Tracked SHA-256 of the private JSON build input")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = args.template.read_text(encoding="utf-8")
    if args.data:
        if args.revision and hashlib.sha256(args.data.read_bytes()).hexdigest() != args.revision.read_text().strip():
            raise ValueError("Private data does not match the tracked build revision")
        data = json.loads(args.data.read_text(encoding="utf-8"))
        encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        encoded = encoded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        marker = "/*__FINANCE_DATA__*/ null"
        if template.count(marker) != 1:
            raise ValueError("Expected one data injection marker")
        template = template.replace(marker, encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template, encoding="utf-8")


if __name__ == "__main__":
    main()
