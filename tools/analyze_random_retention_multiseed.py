#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def load_result(path: Path):
    return np.load(path, allow_pickle=True).item()


def paired_stats(fifo, random, bootstrap_seed):
    a = fifo["episode_success"].astype(np.int8)
    b = random["episode_success"].astype(np.int8)
    wins = int(((b == 1) & (a == 0)).sum())
    losses = int(((b == 0) & (a == 1)).sum())
    ties = int(len(a) - wins - losses)
    p_value = float(binomtest(min(wins, losses), wins + losses, 0.5).pvalue) if wins + losses else 1.0
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(a), size=(100_000, len(a)))
    differences = (b[indices] - a[indices]).mean(axis=1)
    low, high = np.quantile(differences, [0.025, 0.975])
    return wins, losses, ties, float(low), float(high), p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed100-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    all_fifo = []
    all_random = []
    for train_seed in (100, 200, 300):
        if train_seed == 100:
            fifo_path = args.seed100_root / "fifo_eval_100/result_10000_memory-none.npy"
            random_path = args.seed100_root / "random_eval_100/result_10000_memory-none.npy"
        else:
            fifo_path = args.artifact_root / f"seed{train_seed}/fifo_eval_100/result_10000_memory-none.npy"
            random_path = args.artifact_root / f"seed{train_seed}/random_eval_100/result_10000_memory-none.npy"
        if not fifo_path.is_file() or not random_path.is_file():
            continue
        fifo = load_result(fifo_path)
        random = load_result(random_path)
        wins, losses, ties, low, high, p_value = paired_stats(fifo, random, 20260814 + train_seed)
        fs = fifo["episode_success"].astype(np.int8)
        rs = random["episode_success"].astype(np.int8)
        rows.append((train_seed, int(fs.sum()), int(rs.sum()), int(rs.sum() - fs.sum()), wins, losses, ties, low, high, p_value))
        all_fifo.append(fs)
        all_random.append(rs)

    lines = [
        "# Random retention 多 seed 筛选结果",
        "",
        "统一评测 seed：20260814；每个训练 seed、每种方法 100 episodes。",
        "",
        "| 训练 seed | FIFO | Random | 差值 | Random独赢 | FIFO独赢 | 同结果 | 配对95% CI | McNemar p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, fifo_n, random_n, delta, wins, losses, ties, low, high, p_value in rows:
        lines.append(
            f"| {seed} | {fifo_n}/100 | {random_n}/100 | {delta:+d}pp | {wins} | {losses} | {ties} | [{low*100:+.1f}, {high*100:+.1f}]pp | {p_value:.4f} |"
        )

    if all_fifo:
        fifo = {"episode_success": np.concatenate(all_fifo)}
        random = {"episode_success": np.concatenate(all_random)}
        wins, losses, ties, low, high, p_value = paired_stats(fifo, random, 20260814)
        fifo_n = int(fifo["episode_success"].sum())
        random_n = int(random["episode_success"].sum())
        total = len(fifo["episode_success"])
        lines.extend([
            f"| 汇总 | {fifo_n}/{total} | {random_n}/{total} | {100*(random_n-fifo_n)/total:+.1f}pp | {wins} | {losses} | {ties} | [{low*100:+.1f}, {high*100:+.1f}]pp | {p_value:.4f} |",
            "",
            "## 预注册判断",
            "",
            "只有当 Random 在多数训练 seed 上方向一致，且汇总配对区间不再跨 0，才将其视为稳定候选；否则停止扩展 Random，并回到保留目标设计。",
        ])
    else:
        lines.extend(["", "尚无完整配对结果。"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
