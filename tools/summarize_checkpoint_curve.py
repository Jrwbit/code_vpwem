#!/usr/bin/env python3
"""Collect VPWEM checkpoint evaluations and compute adjacent score changes."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
from pathlib import Path

import numpy as np


FIELDS = (
    "framework",
    "task",
    "data_type",
    "seed",
    "checkpoint",
    "mean_success",
    "mean_reward",
    "success_delta",
    "success_delta_per_100k",
    "success_relative_change",
    "result_path",
)


def scalar(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_result(path: Path) -> dict[str, object]:
    value = np.load(path, allow_pickle=True)
    if value.shape == ():
        value = value.item()
    if not isinstance(value, dict):
        raise ValueError(f"expected a dict result, got {type(value).__name__}")
    return value


def collect_mikasa(output_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in output_root.glob(
        "*/ours/mikasa/*/obs_2_act_8/"
        "8layer_5_qformer_2_sin_pos_2_fifo_8_0.0001/*/result_*_8.npy"
    ):
        parts = path.parts
        task = parts[parts.index("mikasa") + 1]
        seed = int(path.parent.name)
        checkpoint = int(path.stem.split("_")[1])
        result = load_result(path)
        rows.append(
            {
                "framework": "mikasa",
                "task": task,
                "data_type": "",
                "seed": seed,
                "checkpoint": checkpoint,
                "mean_success": scalar(result.get("mean_success")),
                "mean_reward": scalar(result.get("mean_reward")),
                "result_path": str(path),
            }
        )
    return rows


def collect_momart(output_root: Path) -> list[dict[str, object]]:
    rows = []
    pattern = (
        "ours/momart_emb/*/obs_2_act_8/"
        "8layer_20_qformer_2_sin_pos_2_fifo_8/*/result_ckpt*_8.npy"
    )
    for path in output_root.glob(pattern):
        name = path.parts[path.parts.index("momart_emb") + 1]
        data_type = "suboptimal" if name.endswith("_suboptimal") else "expert"
        task = name.removesuffix(f"_{data_type}")
        seed = int(path.parent.name)
        checkpoint = int(path.stem.split("ckpt", 1)[1].split("_", 1)[0])
        result = load_result(path)
        rows.append(
            {
                "framework": "momart",
                "task": task,
                "data_type": data_type,
                "seed": seed,
                "checkpoint": checkpoint,
                "mean_success": scalar(result.get("mean_success")),
                "mean_reward": scalar(result.get("mean_reward")),
                "result_path": str(path),
            }
        )
    return rows


def add_changes(rows: list[dict[str, object]]) -> None:
    rows.sort(
        key=lambda row: (
            str(row["framework"]),
            str(row["task"]),
            str(row["data_type"]),
            int(row["seed"]),
            int(row["checkpoint"]),
        )
    )
    previous: dict[tuple[str, str, str, int], tuple[int, float]] = {}
    for row in rows:
        key = (
            str(row["framework"]),
            str(row["task"]),
            str(row["data_type"]),
            int(row["seed"]),
        )
        checkpoint = int(row["checkpoint"])
        success = float(row["mean_success"])
        if key not in previous:
            delta = per_100k = relative = ""
        else:
            previous_checkpoint, previous_success = previous[key]
            delta_value = success - previous_success
            delta = delta_value
            per_100k = delta_value * 100_000 / (checkpoint - previous_checkpoint)
            relative = (
                delta_value / previous_success if previous_success != 0 else ""
            )
        row["success_delta"] = delta
        row["success_delta_per_100k"] = per_100k
        row["success_relative_change"] = relative
        previous[key] = (checkpoint, success)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=Path("/home/jrw/vpwem_outputs")
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "/home/jrw/vpwem_outputs/checkpoint_curves/checkpoint_scores.csv"
        ),
    )
    args = parser.parse_args()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.csv.with_suffix(".csv.lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = collect_mikasa(args.output_root) + collect_momart(args.output_root)
        add_changes(rows)
        temporary = args.csv.with_suffix(f".csv.{os.getpid()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(args.csv)
    print(f"wrote {len(rows)} checkpoint scores to {args.csv}")


if __name__ == "__main__":
    main()
