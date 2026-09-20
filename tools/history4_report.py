#!/usr/bin/env python3
"""Generate the Direct/Direct2/NoHistory/VPWEM audit dashboard.

The registry is intentionally explicit: experiment validity is a scientific
judgement and must not be inferred solely from the existence of result.npy.
Run with --check in CI to detect stale generated reports.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/history4/registry.json"
OUT = ROOT / "reports/history4"
METHODS = {"direct", "direct2", "nohistory", "vpwem"}


def load_registry() -> dict:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if set(data["scope"]) != METHODS:
        raise ValueError(f"scope must be exactly {sorted(METHODS)}")
    ids = [row["id"] for row in data["results"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate result ids")
    plan_ids = [row["id"] for row in data["plan"]]
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("duplicate plan ids")
    for row in data["results"]:
        if row["method"] not in METHODS:
            raise ValueError(f"unknown method in {row['id']}: {row['method']}")
        if row["validity"] not in data["validity_levels"]:
            raise ValueError(f"unknown validity in {row['id']}")
    return data


def csv_text(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def dashboard(data: dict) -> str:
    counts = {level: 0 for level in data["validity_levels"]}
    for row in data["results"]:
        counts[row["validity"]] += 1
    lines = [
        "# History-4 experiment dashboard",
        "",
        f"> Snapshot: {data['snapshot_time']}",
        "",
        "This dashboard is generated from `experiments/history4/registry.json`.",
        "`direct2` is the registry name for code-level `direct_token2`.",
        "",
        "## Validity counts",
        "",
        "| Level | Rows | Meaning |",
        "|---|---:|---|",
    ]
    for level, meaning in data["validity_levels"].items():
        lines.append(f"| {level} | {counts[level]} | {meaning} |")
    lines.extend([
        "",
        "## Results",
        "",
        "| Benchmark | Task | Method | Checkpoint | Episodes | Score | Delta vs VPWEM | Level | Status |",
        "|---|---|---|---:|---:|---:|---:|:---:|---|",
    ])
    for row in data["results"]:
        score = "—" if row["score_percent"] is None else f"{row['score_percent']:.2f}%"
        delta = "—" if row["delta_vs_vpwem_pp"] is None else f"{row['delta_vs_vpwem_pp']:+.2f}pp"
        lines.append(
            f"| {row['benchmark']} | {row['task']} | {row['method']} | "
            f"{row['checkpoint']} | {row['episodes']} | {score} | {delta} | "
            f"{row['validity']} | {row['status']} |"
        )
    lines.extend([
        "",
        "## Execution plan",
        "",
        "| Priority | ID | Action | Existing checkpoint | Acceptance gate |",
        "|:---:|---|---|:---:|---|",
    ])
    for row in data["plan"]:
        existing = "yes" if row["uses_existing_checkpoint"] else "no"
        lines.append(
            f"| {row['priority']} | {row['id']} | {row['action']} | "
            f"{existing} | {row['gate']} |"
        )
    lines.extend([
        "",
        "## Update workflow",
        "",
        "1. Add or update one registry row only after checking its config, checkpoint and per-episode result.",
        "2. Assign A/B/C/D using the declared validity definitions; a result file alone is not evidence of validity.",
        "3. Run `python3 tools/history4_report.py` and commit the registry with its generated reports.",
        "4. Run `python3 tools/history4_report.py --check` before using a table in a paper or slide.",
        "",
    ])
    return "\n".join(lines)


def render(data: dict) -> dict[Path, str]:
    result_fields = [
        "id", "benchmark", "task", "method", "checkpoint", "train_seed",
        "eval_seed", "episodes", "score_percent", "delta_vs_vpwem_pp",
        "validity", "status", "evidence", "note",
    ]
    plan_fields = [
        "priority", "id", "action", "uses_existing_checkpoint", "gate", "why",
    ]
    return {
        OUT / "status.csv": csv_text(data["results"], result_fields),
        OUT / "plan.csv": csv_text(data["plan"], plan_fields),
        OUT / "README.md": dashboard(data),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = render(load_registry())
    stale = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if stale:
        print("stale generated files:")
        for path in stale:
            print(path.relative_to(ROOT))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
