#!/usr/bin/env python3
"""Create a paper-style MoMaRT 1M checkpoint comparison figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "reports/momart_1m_eval/momart_1m_scores.csv"
DEFAULT_OUTPUT = REPO_ROOT / "reports/momart_1m_eval/momart_1m_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without extension; PNG, PDF, and SVG are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    labels = [row["label"].replace("\\n", "\n") for row in rows]
    paper_dp = np.array([float(row["paper_dp_vpwem_approx"]) for row in rows])
    paper_mail = np.array([float(row["paper_mail_vpwem_approx"]) for row in rows])
    vpwem = np.array([float(row["vpwem"]) for row in rows])
    direct_token2 = np.array([float(row["direct_token2"]) for row in rows])
    episodes = sum(int(row["episodes_per_method"]) for row in rows)

    labels.append("Average")
    # Figure 4 explicitly labels the paper averages as 69 and 71. Its
    # per-task bars are digitized/rounded estimates, so do not recompute the
    # published aggregates from those approximate values.
    paper_dp = np.append(paper_dp, 69.0)
    paper_mail = np.append(paper_mail, 71.0)
    vpwem = np.append(vpwem, vpwem.mean())
    direct_token2 = np.append(direct_token2, direct_token2.mean())

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(labels), dtype=float)
    width = 0.20
    fig, ax = plt.subplots(figsize=(14.4, 4.5), constrained_layout=False)

    bars_paper_dp = ax.bar(
        x - 1.5 * width,
        paper_dp,
        width,
        color="#777777",
        edgecolor="none",
        label="Paper DP-VPWEM (L2S2)",
    )
    bars_paper_mail = ax.bar(
        x - 0.5 * width,
        paper_mail,
        width,
        color="#BDBDBD",
        edgecolor="none",
        label="Paper MaIL-VPWEM (L2S2)",
    )
    bars_vpwem = ax.bar(
        x + 0.5 * width,
        vpwem,
        width,
        color="#8FCB88",
        edgecolor="none",
        label="Our multitask VPWEM (1M)",
    )
    bars_direct = ax.bar(
        x + 1.5 * width,
        direct_token2,
        width,
        color="#F3A35C",
        edgecolor="none",
        label="Our multitask Direct-token2 (1M)",
    )

    ax.axvline(len(labels) - 1.5, color="#999999", linestyle="--", linewidth=1.2)
    ax.set_title("MoMaRT", pad=8)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(-0.65, len(labels) - 0.35)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", direction="out", length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    # Match the reference figure: emphasize the aggregate values above the
    # final pair, while keeping the ten task groups visually uncluttered.
    for bar, value, extra_height in (
        (bars_paper_dp[-1], paper_dp[-1], 0.0),
        (bars_paper_mail[-1], paper_mail[-1], 5.0),
        (bars_vpwem[-1], vpwem[-1], 0.0),
        (bars_direct[-1], direct_token2[-1], 5.0),
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.8 + extra_height,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.2,
    )
    fig.text(
        0.5,
        0.015,
        "Paper task bars are approximate Figure 4 readings; paper averages are printed values. "
        f"Our results: one multitask training seed, 30 episodes/task ({episodes}/method).",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.30)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".png", {"dpi": 300}),
        (".pdf", {}),
        (".svg", {}),
    ):
        fig.savefig(args.output.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
