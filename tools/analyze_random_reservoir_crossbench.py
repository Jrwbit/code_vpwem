#!/usr/bin/env python3
"""Paired audit for corrected Reservoir-2 raw versus Recent-2 raw."""

import argparse
import math
from pathlib import Path

import numpy as np


def mcnemar_exact(reservoir_only: int, recent_only: int) -> float:
    discordant = reservoir_only + recent_only
    if discordant == 0:
        return 1.0
    lower = min(reservoir_only, recent_only)
    tail = sum(math.comb(discordant, i) for i in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def paired_ci(delta: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]) * 100.0)


def load_success(path: Path) -> np.ndarray:
    result = np.load(path, allow_pickle=True).item()
    success = np.asarray(result["episode_success"], dtype=np.int8).reshape(-1)
    if len(success) != 100:
        raise ValueError(f"expected 100 episodes in {path}, got {len(success)}")
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20260821)

    rows = []
    for task_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        for seed_dir in sorted(task_dir.glob("seed*")):
            seed = seed_dir.name.removeprefix("seed")
            reservoir_path = seed_dir / "reservoir2_eval_100/result_10000_memory-none.npy"
            recent_path = seed_dir / "recent2_eval_100/result_10000_memory-none.npy"
            if not reservoir_path.is_file() or not recent_path.is_file():
                continue
            reservoir = load_success(reservoir_path)
            recent = load_success(recent_path)
            delta = reservoir - recent
            reservoir_only = int(np.sum((reservoir == 1) & (recent == 0)))
            recent_only = int(np.sum((reservoir == 0) & (recent == 1)))
            low, high = paired_ci(delta, rng)
            rows.append((
                task_dir.name,
                seed,
                int(recent.sum()),
                int(reservoir.sum()),
                float(delta.mean() * 100),
                reservoir_only,
                recent_only,
                low,
                high,
                mcnemar_exact(reservoir_only, recent_only),
            ))

    lines = [
        "# Corrected Reservoir-2 cross-benchmark audit",
        "",
        "Reservoir and Recent both emit two raw embeddings and use the same policy architecture. "
        "Evaluation uses 100 paired episodes per training seed.",
        "",
        "| Benchmark | Train seed | Recent | Reservoir | Difference | Reservoir only | Recent only | Paired 95% CI | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, seed, recent, reservoir, delta, r_only, q_only, low, high, p in rows:
        lines.append(
            f"| {task} | {seed} | {recent}/100 | {reservoir}/100 | {delta:+.0f}pp | "
            f"{r_only} | {q_only} | [{low:+.1f}, {high:+.1f}]pp | {p:.4f} |")
    if not rows:
        lines.append("| _No complete paired evaluations yet_ | | | | | | | | |")
    lines.extend([
        "",
        "Screening rule: expand a benchmark to train seeds 200 and 300 only when seed100 "
        "is positive and its paired interval/pattern warrants confirmation. Cross-seed inference "
        "must treat training seed (not pooled episodes) as the replication unit.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
