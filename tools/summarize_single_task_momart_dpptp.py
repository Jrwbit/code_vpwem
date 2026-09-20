#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


TASKS = [
    "table_cleanup_to_dishwasher_expert",
    "table_cleanup_to_dishwasher_suboptimal",
    "table_cleanup_to_sink_expert",
    "table_cleanup_to_sink_suboptimal",
    "table_setup_from_dishwasher_expert",
    "table_setup_from_dishwasher_suboptimal",
    "table_setup_from_dresser_expert",
    "table_setup_from_dresser_suboptimal",
    "unload_dishwasher_to_dresser_expert",
    "unload_dishwasher_to_dresser_suboptimal",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=30)
    parser.add_argument("--checkpoint", type=int, default=1_000_000)
    args = parser.parse_args()

    rows = []
    all_success = []
    for task in TASKS:
        path = args.root / task / f"result_{args.checkpoint}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = np.load(path, allow_pickle=True).item()
        success = np.asarray(result["episode_success"], dtype=bool).reshape(-1)
        if success.size != args.episodes_per_task:
            raise ValueError(f"{task}: expected {args.episodes_per_task} episodes, got {success.size}")
        count = int(success.sum())
        rows.append({
            "task": task,
            "episodes": int(success.size),
            "successes": count,
            "success_rate": float(success.mean()),
        })
        all_success.append(success)

    combined = np.concatenate(all_success)
    expected = len(TASKS) * args.episodes_per_task
    if combined.size != expected:
        raise ValueError(f"expected {expected} total episodes, got {combined.size}")
    summary = {
        "method": "single-task DP-PTP",
        "checkpoint_step": args.checkpoint,
        "episodes_per_task": args.episodes_per_task,
        "total_episodes": int(combined.size),
        "total_successes": int(combined.sum()),
        "mean_success_rate": float(combined.mean()),
        "tasks": rows,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md = [
        "# Single-task DP-PTP 1M evaluation",
        "",
        "| Task | Successes / 30 | Success rate |",
        "|---|---:|---:|",
    ]
    md.extend(
        f"| {row['task']} | {row['successes']} / {row['episodes']} | {row['success_rate']:.2%} |"
        for row in rows
    )
    md.extend([
        f"| **300-episode aggregate** | **{summary['total_successes']} / {summary['total_episodes']}** | **{summary['mean_success_rate']:.2%}** |",
        "",
    ])
    (args.root / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
