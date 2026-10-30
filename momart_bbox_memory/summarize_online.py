"""Aggregate the ten MoMaRT detector-only live evaluations, including misses."""
import argparse
import json
from pathlib import Path

from .evaluate_online import summarize

TASKS = ("table_cleanup_to_dishwasher", "table_cleanup_to_sink",
         "table_setup_from_dishwasher", "table_setup_from_dresser",
         "unload_dishwasher_to_dresser")


def aggregate(root):
    root = Path(root)
    tasks, missing, rows = {}, [], []
    specification = None
    for task in TASKS:
        for split in ("expert", "suboptimal"):
            name = task + "_" + split
            path = root / name / "result.json"
            if not path.exists():
                missing.append(name)
                continue
            result = json.loads(path.read_text())
            current = result["detector"]
            if specification is not None and current != specification:
                raise ValueError("Tasks used different detector specifications")
            specification = current
            records = [json.loads(line) for line in (path.parent / "frames.jsonl").read_text().splitlines()]
            if len(records) != result["gated"]["frames"]:
                raise ValueError(f"Incomplete frame records: {name}")
            # Segment identifiers must remain distinct across tasks.
            for record in records:
                record["demo"] = name + "/" + record["demo"]
            rows.extend(records)
            tasks[name] = result
    return dict(complete=not missing, missing_tasks=missing, tasks=tasks,
                micro_raw=summarize(rows, "raw"), micro_gated=summarize(rows),
                micro_gated_min3pixels=summarize(rows, min_pixels=3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    args = p.parse_args()
    result = aggregate(args.root)
    (Path(args.root) / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "tasks"}, indent=2))


if __name__ == "__main__": main()
