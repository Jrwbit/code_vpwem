#!/usr/bin/env python3
"""Build one report from every locally available Robomimic VPWEM checkpoint."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path("/home/wjr/code/vpwem")
OUT = ROOT / "artifacts/robomimic_all_checkpoints"
STRICT = ROOT / "artifacts/robomimic_intermediate_local/results"
PAPER = {"square_ph": 0.90, "square_mh": 0.72, "transport_ph": 0.80, "transport_mh": 0.42}


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def read_result(path: Path, episodes: int = 30) -> tuple[int, int, float, float, float]:
    result = np.load(path, allow_pickle=True).item()
    if "episode_success" in result:
        values = np.asarray(result["episode_success"], dtype=int)
        episodes = int(values.size)
        successes = int(values.sum())
        rate = successes / episodes
    else:
        rate = float(result["mean_success"])
        successes = int(round(rate * episodes))
    low, high = wilson(successes, episodes)
    return successes, episodes, rate, low, high


def add_rows(rows: list[dict[str, object]], series: str, layers: int, task: str, paths: list[Path]) -> None:
    for path in sorted(paths, key=lambda p: int(p.stem.split("_")[1])):
        step = int(path.stem.split("_")[1])
        successes, episodes, rate, low, high = read_result(path)
        target = PAPER.get(task)
        rows.append({
            "series": series, "layers": layers, "task": task, "step": step,
            "successes": successes, "episodes": episodes, "success_rate": rate,
            "ci_low": low, "ci_high": high, "paper_rate": target,
            "delta_pp": None if target is None else 100 * (rate - target),
        })


def table(lines: list[str], rows: list[dict[str, object]]) -> None:
    lines += [
        "| Task | Step | 成功数 | 成功率 | 95% CI | 论文值 | 差值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        target = row["paper_rate"]
        paper = "—" if target is None else f"{100 * float(target):.1f}%"
        delta = "—" if row["delta_pp"] is None else f"{float(row['delta_pp']):+.1f} pp"
        lines.append(
            f"| {row['task']} | {int(row['step']) // 1000}k | {row['successes']}/{row['episodes']} | "
            f"{100 * float(row['success_rate']):.1f}% | "
            f"[{100 * float(row['ci_low']):.1f}%, {100 * float(row['ci_high']):.1f}%] | {paper} | {delta} |"
        )


def main() -> None:
    rows: list[dict[str, object]] = []
    add_rows(rows, "legacy_full_curve", 4, "square_ph", list((OUT / "legacy4/square_ph").glob("*/result_*_memory-none.npy")))
    add_rows(rows, "legacy_full_curve", 4, "longsquare_ph", list((OUT / "legacy4/longsquare_ph").glob("*/result_*_memory-none.npy")))
    for task in (*PAPER, "longsquare_ph"):
        add_rows(rows, "paper_strict", 8, task, list((STRICT / task).glob("*/result_*_memory-none.npy")))

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "robomimic_all_checkpoint_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    legacy = [r for r in rows if r["series"] == "legacy_full_curve"]
    strict = [r for r in rows if r["series"] == "paper_strict"]
    lines = [
        "# Robomimic 全部现存 checkpoint 成功率",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 评测范围与口径",
        "",
        "- 已评测本机和服务器盘点出的所有不重复 VPWEM checkpoint，范围为 100k–1000k。",
        "- Square/ph 旧曲线每 50k 一个点（100k、150k……1000k）；LongSquare/ph 旧曲线每 100k 一个点。",
        "- 两个 Square 完整目录的 checkpoint 内容重复，抽样 SHA-256 完全一致，因此只评测一份。",
        "- 论文严格 8-layer 只保留了 800k、900k、1000k；这 15 个结果沿用同一固定 seed 的 30-rollout 本机评测。",
        "- 每点 30 个 rollout；区间为 Wilson 95% CI。“接近论文”定义为绝对差不超过 5 个百分点。",
        "",
        "## 旧 4-layer 完整曲线（仅用于观察早期训练趋势）",
        "",
    ]
    table(lines, legacy)
    lines += ["", "## 论文严格 8-layer 可用 checkpoint", ""]
    table(lines, strict)

    square = [r for r in legacy if r["task"] == "square_ph"]
    target = PAPER["square_ph"]
    hits = [r for r in square if abs(float(r["success_rate"]) - target) <= 0.05 + 1e-12]
    best = max(square, key=lambda r: float(r["success_rate"])) if square else None
    lines += ["", "## 结论", ""]
    if hits:
        first = min(hits, key=lambda r: int(r["step"]))
        lines.append(
            f"- 旧 4-layer Square/ph 最早在 {int(first['step']) // 1000}k 命中论文 90% 的 ±5pp 区间，"
            f"当点为 {100 * float(first['success_rate']):.1f}%；但单点命中不等于稳定收敛。"
        )
    if best:
        lines.append(
            f"- 旧 4-layer Square/ph 全曲线最高为 {int(best['step']) // 1000}k 的 "
            f"{100 * float(best['success_rate']):.1f}%。"
        )
    long_rows = [r for r in legacy if r["task"] == "longsquare_ph"]
    if long_rows:
        long_best = max(long_rows, key=lambda r: float(r["success_rate"]))
        lines.append(
            f"- 旧 4-layer LongSquare/ph 最高为 {int(long_best['step']) // 1000}k 的 "
            f"{100 * float(long_best['success_rate']):.1f}%，不能代表标准四任务的论文对齐。"
        )
    lines += [
        "- 严格 8-layer 仍只能判断 800k–1000k：缺失的 100k–700k 权重无法通过评测补回。",
        "- 因此，关于“100k 是否够”的直接证据只能来自旧 4-layer 曲线；不能外推为严格 8-layer 的结论。",
        "",
        "## 数据完整性",
        "",
        f"本报告包含 {len(legacy)} 个旧完整曲线点和 {len(strict)} 个严格 8-layer 点，共 {len(rows)} 个不重复评测结果。",
    ]
    report = OUT / "robomimic_all_checkpoint_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
