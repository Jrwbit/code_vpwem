#!/usr/bin/env python3
"""Summarize strict 8-layer Robomimic intermediate-checkpoint rollouts."""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np


PAPER = {
    "square_ph": 0.90,
    "square_mh": 0.72,
    "transport_ph": 0.80,
    "transport_mh": 0.42,
    "longsquare_ph": None,
}
TASKS = tuple(PAPER)
STEPS = (800_000, 900_000, 1_000_000)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return mid - half, mid + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows: list[dict[str, object]] = []
    missing: list[str] = []

    for task in TASKS:
        for step in STEPS:
            path = root / "results" / task / str(step) / f"result_{step}_memory-none.npy"
            if not path.is_file():
                missing.append(f"{task}@{step // 1000}k")
                continue
            result = np.load(path, allow_pickle=True).item()
            if "episode_success" in result:
                episode_success = np.asarray(result["episode_success"], dtype=int)
                successes = int(episode_success.sum())
                episodes = int(episode_success.size)
                rate = successes / episodes
            else:
                # The dedicated LongSquare evaluator stores only aggregate
                # metrics. This evaluation queue always requests 30 rollouts.
                episodes = 30
                rate = float(result["mean_success"])
                successes = int(round(rate * episodes))
            low, high = wilson(successes, episodes)
            target = PAPER[task]
            rows.append({
                "task": task,
                "step": step,
                "successes": successes,
                "episodes": episodes,
                "success_rate": rate,
                "ci_low": low,
                "ci_high": high,
                "paper_rate": target,
                "delta_pp": None if target is None else 100 * (rate - target),
            })

    csv_path = root / "robomimic_intermediate_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["task", "step"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Robomimic 严格 8-layer 中间 checkpoint 成功率",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 口径",
        "",
        "- 模型：论文一致的 8-layer diffusion transformer；QFormer 2 层、FIFO bank=8。",
        "- checkpoint：本次能找到的严格版本为 800k、900k、1000k。",
        "- 评测：每个点 30 个 rollout，固定评测 seed；表中区间是成功率的 Wilson 95% CI。",
        "- “接近论文”：仅作便于阅读的 ±5 个百分点标记；30 次 rollout 的统计误差仍然较大。",
        "",
        "## 结果",
        "",
        "| Task | Step | 成功数 | 成功率 | 95% CI | 论文值 | 差值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        target = row["paper_rate"]
        paper = "—" if target is None else f"{100 * float(target):.1f}%"
        delta = "—" if row["delta_pp"] is None else f"{float(row['delta_pp']):+.1f} pp"
        lines.append(
            f"| {row['task']} | {int(row['step']) // 1000}k | "
            f"{row['successes']}/{row['episodes']} | {100 * float(row['success_rate']):.1f}% | "
            f"[{100 * float(row['ci_low']):.1f}%, {100 * float(row['ci_high']):.1f}%] | {paper} | {delta} |"
        )

    lines += ["", "## 结论", ""]
    if missing:
        lines.append("评测未完整结束，缺少：" + "、".join(missing) + "。以下结论仅基于已完成点。")
        lines.append("")
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        target = PAPER[task]
        if not task_rows:
            lines.append(f"- {task}：暂无成功完成的评测点。")
            continue
        if target is None:
            best = max(task_rows, key=lambda row: float(row["success_rate"]))
            lines.append(
                f"- {task}：这组不参与标准四任务论文值对齐；本次最好为 "
                f"{int(best['step']) // 1000}k 的 {100 * float(best['success_rate']):.1f}%。"
            )
            continue
        close = [row for row in task_rows if abs(float(row["success_rate"]) - target) <= 0.05 + 1e-12]
        if close:
            earliest = min(close, key=lambda row: int(row["step"]))
            lines.append(
                f"- {task}：在现有严格 checkpoint 中，最早于 {int(earliest['step']) // 1000}k "
                f"进入论文值 ±5pp（{100 * float(earliest['success_rate']):.1f}% vs {100 * target:.1f}%）。"
            )
        else:
            nearest = min(task_rows, key=lambda row: abs(float(row["success_rate"]) - target))
            lines.append(
                f"- {task}：800k–1000k 尚无点进入论文值 ±5pp；最近为 "
                f"{int(nearest['step']) // 1000}k 的 {100 * float(nearest['success_rate']):.1f}% "
                f"（论文 {100 * target:.1f}%）。"
            )

    lines += [
        "",
        "## 限制与下一步",
        "",
        "严格 8-layer 的 100k–700k checkpoint 当前未保留，所以本轮不能据此判断更早的收敛点。"
        "本机旧存档虽覆盖这些步数，但其 diffusion transformer 约为 4-layer，不能混作论文一致实验。"
        "若需要精确判断 100k 是否够，应让当前 8-layer 训练从 100k 起按 100k 保存，并用相同 rollout 口径复测。",
    ]
    report_path = root / "robomimic_intermediate_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
