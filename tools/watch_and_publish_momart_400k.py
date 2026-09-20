#!/usr/bin/env python3
"""Wait for the complete MoMaRT 400k evaluation and publish it to Feishu."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import subprocess
import time

import numpy as np


REMOTE_ROOT = "/home/jrw/vpwem_outputs/fair_history_momart_multitask_eval_400k"
EXPECTED_RESULTS = 30


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def remote_result_count(host: str) -> int:
    result = run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
            f"find {REMOTE_ROOT} -name result.npy -type f | wc -l",
        ]
    )
    return int(result.stdout.strip())


def sync_outputs(host: str, destination: pathlib.Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            "rsync", "-av",
            "--include=*/", "--include=result.npy", "--include=*.mp4",
            "--exclude=*", f"{host}:{REMOTE_ROOT}/", f"{destination}/",
        ]
    )


def load_rows(data_root: pathlib.Path) -> list[dict]:
    rows = []
    for path in sorted(data_root.glob("*/*/result.npy")):
        value = np.load(path, allow_pickle=True).item()
        rows.append(
            {
                "method": path.parents[1].name,
                "task": path.parent.name,
                "success": float(value["mean_success"]),
                "reward": float(value["mean_reward"]),
                "infer": float(value["mean_infer_time"]),
                "episodes": len(value["episode_success"]),
            }
        )
    return rows


def write_report(path: pathlib.Path, rows: list[dict], videos: list[pathlib.Path]) -> None:
    methods = ["vpwem", "direct", "direct_token2"]
    lines = [
        "# MoMaRT 多任务 400k checkpoint 评测",
        "",
        f"完成时间：{dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "评测设置：seed=20260917；每个任务配置 30 episodes；每个 episode 最长 "
        "1504 steps；共 3 个方法 × 10 个任务配置 = 900 episodes。",
        "",
        "注意：VPWEM 使用 embedding policy 与任务对应的 context encoder；direct 和 "
        "direct_token2 使用 raw visual policy，因此这是相同训练步数与任务协议下的结果，"
        "并非完全相同 observation encoder 架构的严格消融。",
        "",
        "## 汇总",
        "",
        "| 方法 | 配置数 | Episodes | 平均成功率 | 平均 reward | 平均推理时间/次 (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        lines.append(
            f"| {method} | {len(selected)} | {sum(r['episodes'] for r in selected)} | "
            f"{np.mean([r['success'] for r in selected]):.4f} | "
            f"{np.mean([r['reward'] for r in selected]):.4f} | "
            f"{np.mean([r['infer'] for r in selected]):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 分任务结果",
            "",
            "| 方法 | 任务配置 | Episodes | 成功率 | Reward | 推理时间/次 (s) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['task']} | {row['episodes']} | "
            f"{row['success']:.4f} | {row['reward']:.4f} | {row['infer']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 视频",
            "",
            f"共 {len(videos)} 个代表性 MP4，已作为飞书文档附件上传；每个方法、每个任务配置 1 个。",
            "",
            f"BHR 原始结果目录：`{REMOTE_ROOT}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def publish(repo: pathlib.Path, report: pathlib.Path, videos: list[pathlib.Path]) -> str:
    command = [
        "/home/wjr/.venvs/feishu-codex-bridge/bin/python",
        str(repo / "tools/feishu_codex_bridge/publish_markdown.py"),
        str(report),
        "--title", "MoMaRT 多任务 400k checkpoint 评测结果",
    ]
    for video in videos:
        command.extend(["--attachment", str(video)])
    result = run(command)
    if "WARNING: 附件" in result.stdout:
        raise RuntimeError(result.stdout)
    return result.stdout.splitlines()[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="10.1.77.25")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = repo / "reports/momart_400k_eval"
    log_path = output / "watcher.log"
    output.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"[{timestamp}] {message}\n")

    while True:
        try:
            count = remote_result_count(args.host)
            log(f"results={count}/{EXPECTED_RESULTS}")
            if count >= EXPECTED_RESULTS:
                break
        except Exception as exc:
            log(f"poll_error={exc}")
        time.sleep(args.poll_seconds)

    data_root = output / "data"
    sync_outputs(args.host, data_root)
    rows = load_rows(data_root)
    if len(rows) != EXPECTED_RESULTS:
        raise RuntimeError(f"expected {EXPECTED_RESULTS} results after sync, got {len(rows)}")
    videos = sorted(data_root.rglob("*.mp4"))
    report = output / "report.md"
    write_report(report, rows, videos)
    url = publish(repo, report, videos)
    (output / "feishu_url.txt").write_text(url + "\n", encoding="utf-8")
    log(f"published={url} videos={len(videos)}")


if __name__ == "__main__":
    main()
