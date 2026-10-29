"""Render aggregate MoMaRT success rates from the read-only collector snapshot."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/momart_total_success_20261029.md"))
    args = parser.parse_args()
    snapshot = json.loads(args.evidence.read_text())
    groups = {}
    for name, root in snapshot["roots"].items():
        for row in root["results"]:
            if "error" in row:
                raise ValueError(row)
            data = row["data"]
            flags = data["episode_success"]
            assert len(flags) == data["verified_n"] == 30
            assert all(flag in (0, 1) for flag in flags)
            assert sum(flags) == data["verified_success"]
            prefix = "step_250000" if name == "nohistory_250k" else row["path"].split("/")[0]
            key = (name, prefix)
            if name == "nohistory_250k":
                assert data["checkpoint_step"] == 250000
                assert data["method"] == "no_history"
            task = (data["env_name"], data["env_type"])
            group = groups.setdefault(key, {})
            assert task not in group
            group[task] = data

    def total(key, seed, complete=True):
        group = groups.get(key, {})
        assert all(data["eval_seed"] == seed for data in group.values())
        if complete:
            assert len(group) == 10, key
        return sum(data["verified_success"] for data in group.values()), len(group) * 30

    specs = [
        ("旧 No-History\n250k / 8步", ("nohistory_250k", "step_250000")),
        ("带框\n250k / 8步", ("250k", "step_250000")),
        ("旧 No-History\n1M / 8步", ("1m", "no_history")),
        ("带框\n150k / 8步", ("150k", "boxed_no_history")),
        ("带框\n175k / 8步", ("intermediate", "step_175000")),
        ("带框\n200k / 8步", ("intermediate", "step_200000")),
        ("带框\n225k / 8步", ("intermediate", "step_225000")),
        ("带框\n150k / 4步", ("150k_action4", "step_150000")),
        ("带框+ROI\n150k / 8步", ("150k", "boxed_current_object_tokens")),
    ]
    baseline_tasks = set(groups[specs[0][1]])
    values = []
    for _, key in specs:
        assert set(groups[key]) == baseline_tasks
        values.append(total(key, 20260917))
    rates = [100 * successes / n for successes, n in values]
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 11})
    fig, ax = plt.subplots(figsize=(15, 6.6))
    colors = ["#D68732", "#267D70", "#C4A475"] + ["#4D86AD"] * 4 + ["#7799B4", "#9AA6B2"]
    bars = ax.bar(range(len(specs)), rates, color=colors, width=0.65, zorder=3)
    ax.axhline(rates[0], color=colors[0], linestyle="--", linewidth=1.2, zorder=2)
    for bar, rate, (successes, n) in zip(bars, rates, values):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 1.3,
                f"{rate:.2f}%\n{successes}/{n}", ha="center", va="bottom", fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.set_xticks(range(len(specs)), [label for label, _ in specs])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylabel("总成功率")
    ax.set_title("MoMaRT 总成功率：与旧 No-History 对照", loc="left", fontsize=18, pad=30)
    ax.text(0, 1.025, "相同评测 seed 20260917 · 每组 10 配置 × 30 回合 · 训练预算不同",
            transform=ax.transAxes, fontsize=11, color="#596574")
    ax.grid(axis="y", color="#E3E7EB", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.08, 0.035, f"虚线：旧 No-History 250k {rates[0]:.2f}%；带框 250k {rates[1]:.2f}%，提高 {rates[1]-rates[0]:.2f} 个百分点。", fontsize=11)
    fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.82)
    asset_dir = args.output.parent / "assets" / args.output.stem
    asset_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(asset_dir / f"comparison.{suffix}", dpi=180)
    svg = asset_dir / "comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    lines = [
        "# MoMaRT 总成功率对照｜2026-10-29", "",
        "飞书图文报告：https://feishu.cn/docx/RXj4dQ7gHoPAJvxzKB1cTwwMnZd", "",
        "按最新要求，仅汇报总成功率，并以旧 No-History 为对照；不展开任务分项、逐回合一致性或额外方法排名。",
        f"数据快照：{snapshot['time']}；各项均由原始 result.npy 的成功标志复算。", "",
        f"![MoMaRT 总成功率柱状图](assets/{args.output.stem}/comparison.png)", "",
        f"旧 No-History 250k：**{values[0][0]}/300，{rates[0]:.2f}%**；带框 250k 八步：**{values[1][0]}/300，{rates[1]:.2f}%**，提高 **{rates[1]-rates[0]:.2f} 个百分点**（多成功{values[1][0]-values[0][0]}回合）。旧 No-History 1M 的72.33%保留为较长训练预算参考。", "",
        "| 版本 | 总成功数/回合 | 总成功率 | 相对旧 No-History 250k |",
        "|---|---|---|---|",
    ]
    for (label, _), (successes, n), rate in zip(specs, values, rates):
        lines.append(f"| {label.replace(chr(10), ' ')} | {successes}/{n} | {rate:.2f}% | {rate-rates[0]:+.2f} 个百分点 |")
    lines += ["", "主图各组使用相同评测 seed 20260917；250k无框与带框组有效 batch 均为256，旧 No-History 1M 为64。250k训练步数一致，但带框组由150k续训并调整学习率计划，不能仅凭这一次对照把增益全部归因于画框。总成功率 = 所有配置成功回合数之和 ÷ 所有配置回合数之和。", "",
              "## 250k 其他评测 seed：仅列总成功率", "",
              "| 版本 | 评测 seed | 总成功数/回合 | 总成功率 |", "|---|---|---|---|"]
    for seed in (100, 200, 300):
        successes, n = total(("250k_action4", f"seed{seed}"), seed)
        lines.append(f"| 带框250k / 4步 | {seed} | {successes}/{n} | {100*successes/n:.2f}% |")
    for prefix, seed in [("seed100_repeat1", 100), ("seed100_repeat2", 100), ("seed100_repeat3", 100), ("seed200", 200), ("seed300", 300)]:
        successes, n = total(("repeats", prefix), seed, complete=False)
        rate = f"{100*successes/n:.2f}%" if n == 300 else "未完成，不列最终成功率"
        lines.append(f"| 带框250k / 8步 {prefix} | {seed} | {successes}/{n} | {rate} |")
    lines += ["", "不同评测 seed 与主图分开呈现；同 seed100 重复执行不计作独立 seed。补跑完成后按完整300回合更新总成功率。", "",
              "历史详细记录：https://feishu.cn/docx/LJD4dbDkTo8jwsxuPVTch9RunSz", "",
              f"旧 No-History 250k 原始结果：{snapshot['roots']['nohistory_250k']['path']}。已核验10个配置、各30回合、checkpoint_step=250000、eval_seed=20260917。", "",
              f"复算：tools/collect_momart_evaluation_20261029.py --include-nohistory-250k；制图：tools/report_momart_total_success.py。原始快照：{args.evidence}（Git忽略）。", ""]
    args.output.write_text("\n".join(lines))
    print(f"Verified {len(specs)} main comparisons; wrote {args.output}")


if __name__ == "__main__":
    main()
