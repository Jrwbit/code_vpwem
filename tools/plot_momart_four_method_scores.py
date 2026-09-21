#!/usr/bin/env python3
"""Render the audited MoMaRT 1M task scores as two documentation charts."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/momart_1m_four_methods/report_feishu.md"
OUTPUT = ROOT / "docs/assets/momart_four_methods_1m"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
METHODS = ("VPWEM", "Direct-token2", "Direct", "No-History")
COLORS = ("#7C3AED", "#F59E0B", "#16A34A", "#475569")
EXPECTED_TOTALS = (196, 215, 216, 217)


def read_scores() -> tuple[list[str], list[list[int]]]:
    report = SOURCE.read_text(encoding="utf-8")
    table = report.split("## 7. 四种方法的逐任务结果", 1)[1].split(
        "### 7.1", 1
    )[0]
    names, counts = [], []
    for line in table.splitlines():
        if not line.startswith("| ") or " / " not in line:
            continue
        cells = [cell.strip() for cell in line.strip("| ").split("|")]
        if len(cells) != 5 or cells[0].startswith("**"):
            continue
        row = []
        for cell in cells[1:]:
            match = re.search(r"（(\d+)/30）", cell)
            if not match:
                raise ValueError(f"Missing success count: {cell}")
            row.append(int(match.group(1)))
        names.append(cells[0])
        counts.append(row)
    if len(counts) != 10 or any(len(row) != 4 for row in counts):
        raise ValueError("Expected 10 task configurations × 4 methods")
    totals = tuple(sum(row[col] for row in counts) for col in range(4))
    if totals != EXPECTED_TOTALS:
        raise ValueError(f"Totals differ from audited report: {totals}")
    return names, counts


def fonts() -> tuple[ImageFont.FreeTypeFont, ...]:
    return tuple(ImageFont.truetype(FONT, size) for size in (41, 25, 21, 18, 17))


def chart_base(width: int, height: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw, tuple]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = fonts()
    draw.text((45, 29), title, font=font[0], fill="#172033")
    draw.text(
        (46, 88), "1M checkpoint  ·  training seed 100  ·  30 episodes per task configuration",
        font=font[3], fill="#64748B",
    )
    return image, draw, font


def draw_legend(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, top: int) -> None:
    x_positions = (340, 620, 975, 1190)
    for x, name, color in zip(x_positions, METHODS, COLORS):
        draw.rounded_rectangle((x, top, x + 23, top + 23), radius=4, fill=color)
        draw.text((x + 32, top - 4), name, font=font, fill="#334155")


def plot_tasks(names: list[str], counts: list[list[int]]) -> None:
    width, height = 1600, 1410
    image, draw, font = chart_base(
        width, height, "MoMaRT 四方法：10 项任务成功率"
    )
    draw_legend(draw, font[2], 145)
    left, right = 345, 1275
    top, group_gap, bar_h = 228, 108, 17
    for tick in range(0, 101, 20):
        x = left + round((right - left) * tick / 100)
        draw.line((x, top - 8, x, top + 10 * group_gap - 10), fill="#E2E8F0", width=2)
        draw.text((x - 14, top + 10 * group_gap), str(tick), font=font[4], fill="#64748B")
    for row_index, (name, row) in enumerate(zip(names, counts)):
        group_y = top + row_index * group_gap
        draw.text((48, group_y + 21), name, font=font[2], fill="#172033")
        for col, (count, color) in enumerate(zip(row, COLORS)):
            y = group_y + col * 22
            percent = count / 30 * 100
            end = left + round((right - left) * percent / 100)
            draw.rounded_rectangle((left, y, end, y + bar_h), radius=4, fill=color)
            draw.text(
                (end + 9, y - 5), f"{percent:.1f}%",
                font=font[4], fill="#334155",
            )
        if row_index < 9:
            draw.line((46, group_y + 98, width - 46, group_y + 98), fill="#F1F5F9", width=2)
    draw.text((45, height - 45), "数值 = 成功次数 ÷ 30。expert / suboptimal 为两种数据配置。", font=font[3], fill="#64748B")
    image.save(OUTPUT / "task_scores.png", optimize=True)


def plot_averages(counts: list[list[int]]) -> None:
    totals = [sum(row[col] for row in counts) for col in range(4)]
    image, draw, font = chart_base(1500, 615, "MoMaRT 四方法：平均成功率")
    left, right = 295, 1160
    top = 185
    for tick in range(0, 101, 20):
        x = left + round((right - left) * tick / 100)
        draw.line((x, top - 20, x, 506), fill="#E2E8F0", width=2)
        draw.text((x - 14, 515), str(tick), font=font[4], fill="#64748B")
    for row, (name, color, total) in enumerate(zip(METHODS, COLORS, totals)):
        y = top + row * 80
        percent = total / 300 * 100
        end = left + round((right - left) * percent / 100)
        draw.text((45, y + 5), name, font=font[2], fill="#172033")
        draw.rounded_rectangle((left, y, end, y + 45), radius=7, fill=color)
        draw.text(
            (end + 13, y + 5), f"{percent:.2f}%  ({total}/300)",
            font=font[2], fill="#172033",
        )
    draw.text((45, 561), "10 项任务各评测 30 次；平均值也等于 300 次评测的总体成功比例。", font=font[3], fill="#64748B")
    image.save(OUTPUT / "mean_scores.png", optimize=True)


def main() -> None:
    names, counts = read_scores()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_tasks(names, counts)
    plot_averages(counts)
    print(f"Verified 10×4 scores; totals={list(EXPECTED_TOTALS)}")


if __name__ == "__main__":
    main()
