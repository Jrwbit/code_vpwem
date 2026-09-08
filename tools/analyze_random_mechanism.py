#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def load(path):
    return np.load(path, allow_pickle=True).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "FIFO QFormer bank8": args.reference_root / "fifo_eval_100/result_10000_memory-none.npy",
        "Recent-2 raw": args.root / "recent2_eval_100/result_10000_memory-none.npy",
        "QFormer random-evict bank8": args.root / "qformer_random8_eval_100/result_10000_memory-none.npy",
        "Reservoir-2 raw": args.reference_root / "random_eval_100/result_10000_memory-none.npy",
    }
    rows = []
    for name, path in paths.items():
        if path.is_file():
            result = load(path)
            rows.append((name, int(result["episode_success"].sum()), float(result["mean_reward"])))
    lines = [
        "# Random retention 机制消融（seed100）", "",
        "| 方法 | 成功数/100 | 平均奖励 |", "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {success}/100 | {reward:.2f} |" for name, success, reward in rows)
    lines.extend([
        "", "解释口径：Recent-2 对比 Reservoir-2 隔离长历史覆盖；QFormer random-evict 对比 FIFO 隔离 bank 淘汰规则。",
    ])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
