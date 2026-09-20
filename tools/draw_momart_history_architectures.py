#!/usr/bin/env python3
"""Draw publication-style architecture figures for the MoMaRT history baselines.

The figures deliberately mirror the visual grammar and three-part logic of the
VPWEM paper's Fig. 2 while keeping every connection faithful to this repository's
fair-history implementation.
"""

from __future__ import annotations

import html
from pathlib import Path


WIDTH, HEIGHT = 1800, 900
BLACK = "#111111"
GREEN = "#DCEFCB"
GREEN_DARK = "#BCD99D"
ORANGE = "#F29A5A"
ORANGE_LIGHT = "#F8D5B8"
BLUE = "#77AEDD"
BLUE_LIGHT = "#C9DFF1"
GRAY = "#D8D8D8"
LIGHT = "#F5F5F5"


class Canvas:
    def __init__(self, title: str):
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{BLACK}"/></marker>
  <marker id="arrow-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{ORANGE}"/></marker>
  <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/></marker>
  <style>
    text {{ font-family: 'Times New Roman', 'DejaVu Serif', serif; fill: #111; }}
    .panel {{ font-size: 27px; font-weight: 700; }}
    .label {{ font-size: 20px; }}
    .small {{ font-size: 16px; }}
    .tiny {{ font-size: 14px; }}
    .method {{ font-size: 31px; font-weight: 700; }}
  </style>
</defs>
<rect width="1800" height="900" fill="white"/>
<text x="900" y="38" text-anchor="middle" class="method">{html.escape(title)}</text>
<line x1="25" y1="55" x2="1775" y2="55" stroke="#BDBDBD" stroke-width="1"/>
''']

    def add(self, value: str):
        self.parts.append(value)

    def rect(self, x, y, w, h, fill="white", stroke=BLACK, sw=2, rx=0, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>')

    def line(self, x1, y1, x2, y2, color=BLACK, sw=2, arrow=True, dash=None):
        marker = ""
        if arrow:
            marker_id = "arrow-orange" if color == ORANGE else "arrow-blue" if color == BLUE else "arrow"
            marker = f' marker-end="url(#{marker_id})"'
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{marker}{dash_attr}/>')

    def path(self, d, color=BLACK, sw=2, arrow=True, dash=None, fill="none"):
        marker = ""
        if arrow:
            marker_id = "arrow-orange" if color == ORANGE else "arrow-blue" if color == BLUE else "arrow"
            marker = f' marker-end="url(#{marker_id})"'
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{marker}{dash_attr}/>')

    def text(self, x, y, value, size=20, anchor="middle", weight="normal", color=BLACK, italic=False, cls=None):
        style = f'font-size:{size}px;font-weight:{weight};fill:{color};' + ("font-style:italic;" if italic else "")
        cls_attr = f' class="{cls}"' if cls else ""
        self.add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{style}"{cls_attr}>{html.escape(value)}</text>')

    def multiline(self, x, y, lines, size=18, anchor="middle", weight="normal", color=BLACK, gap=23):
        self.add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="font-size:{size}px;font-weight:{weight};fill:{color}">')
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else gap
            self.add(f'<tspan x="{x}" dy="{dy}">{html.escape(line)}</tspan>')
        self.add('</text>')

    def module(self, x, y, w, h, lines, learned=True, border=BLACK, layers=1, font=18):
        fill = GREEN if learned else "white"
        for layer in range(layers - 1, 0, -1):
            self.rect(x + layer * 6, y - layer * 5, w, h, fill=fill, stroke=border, sw=1.8, rx=8)
        self.rect(x, y, w, h, fill=fill, stroke=border, sw=2.2, rx=8)
        start = y + h / 2 - (len(lines) - 1) * 11 + 7
        self.multiline(x + w / 2, start, lines, size=font, gap=22)

    def tokens(self, x, y, count, color, w=28, h=38, gap=3, label=None):
        for i in range(count):
            self.rect(x + i * (w + gap), y, w, h, fill=color, stroke=BLACK, sw=1.5)
        if label:
            self.text(x + (count * (w + gap) - gap) / 2, y - 8, label, size=15)

    def frame_stack(self, x, y, count=4, color=ORANGE, scale=1.0, fade=False):
        w, h = 62 * scale, 48 * scale
        for i in range(count):
            xx, yy = x + i * 18 * scale, y - i * 6 * scale
            opacity = 0.30 + 0.18 * i if fade else 1
            self.add(f'<g opacity="{min(opacity, 1):.2f}">')
            self.rect(xx, yy, w, h, fill="#F7F7F7", stroke=color, sw=2)
            # Tiny, deterministic robot/table glyph inside each observation frame.
            self.add(f'<line x1="{xx+8*scale}" y1="{yy+36*scale}" x2="{xx+54*scale}" y2="{yy+36*scale}" stroke="#8A8A8A" stroke-width="{2*scale}"/>')
            self.add(f'<circle cx="{xx+25*scale}" cy="{yy+18*scale}" r="{5*scale}" fill="#9A9A9A"/>')
            self.add(f'<line x1="{xx+25*scale}" y1="{yy+23*scale}" x2="{xx+40*scale}" y2="{yy+31*scale}" stroke="#777" stroke-width="{3*scale}"/>')
            self.add('</g>')

    def action(self, x, y, n=4, label="Action Chunk"):
        self.tokens(x, y, n, GRAY, w=34, h=38, gap=0)
        self.text(x + n * 17, y + 58, label, size=17)

    def finish(self):
        self.add('</svg>')
        return "".join(self.parts)


def panel_headers(c: Canvas):
    c.text(45, 92, "(a) Architecture", size=27, anchor="start", weight="700")
    c.text(835, 92, "(b) Training", size=27, anchor="start", weight="700")
    c.text(835, 484, "(c) Inference", size=27, anchor="start", weight="700")
    c.line(805, 70, 805, 870, color="#BDBDBD", sw=1, arrow=False)
    c.line(820, 458, 1770, 458, color="#BDBDBD", sw=1, arrow=False)


def top_timeline(c: Canvas, x1=105, x2=745, y=135):
    c.line(x1, y, x2, y, sw=1.8)
    for x in [195, 285, 375, 520, 610, 685]:
        c.line(x, y - 8, x, y + 8, sw=1.4, arrow=False)
    c.text(x2 - 5, y - 14, "Time", size=18, anchor="end")
    c.text(690, y - 16, "t", size=20, italic=True)


def predictor(c: Canvas, x, y, w=205, h=95):
    c.module(x, y, w, h, ["Diffusion / Flow Policy", "for K Denoising Steps"], learned=True, font=18)


def noise_training(c: Canvas, x, y, cond_x, cond_y):
    c.action(x, y, n=4, label="clean actions  a₀")
    c.path(f"M {x+135} {y+18} C {x+175} {y+20}, {x+170} {y+92}, {x+205} {y+92}", sw=1.8)
    c.text(x+255, y+62, "Noising", size=15)
    c.tokens(x+205, y+72, 4, GRAY, w=28, h=34, gap=0)
    c.text(x+255, y+125, "aₖ", size=18, italic=True)
    c.module(x+600, y+48, 155, 72, ["Noise", "Predictor"], learned=True, font=18)
    c.line(x+317, y+89, x+600, y+89, sw=1.8)
    c.line(cond_x, cond_y, x+600, y+72, sw=2)
    c.line(x+755, y+84, x+790, y+84, sw=2)
    c.text(x+805, y+91, "ε", size=22, italic=True)


def legend(c: Canvas, y=865):
    c.rect(58, y-17, 24, 17, fill=GREEN, stroke=BLACK, sw=1)
    c.text(90, y-3, "learnable", size=14, anchor="start")
    c.rect(205, y-17, 24, 17, fill="white", stroke=BLACK, sw=1)
    c.text(237, y-3, "fixed / non-learnable", size=14, anchor="start")
    c.line(420, y-8, 458, y-8, color=ORANGE, sw=3)
    c.text(468, y-3, "long history", size=14, anchor="start", color=ORANGE)
    c.line(590, y-8, 628, y-8, color=BLUE, sw=3)
    c.text(638, y-3, "short context", size=14, anchor="start", color=BLUE)


def draw_direct_token2() -> str:
    c = Canvas("Direct-token2 — Fixed Two-Token Compression of Spatial History")
    panel_headers(c)
    top_timeline(c)
    c.frame_stack(130, 192, count=5, color=ORANGE, fade=True)
    c.text(230, 175, "Long History", size=18, color=ORANGE)
    c.frame_stack(575, 192, count=2, color=BLUE)
    c.text(640, 175, "Context Window (2)", size=18, color=BLUE)
    c.module(565, 280, 165, 72, ["Shared Multi-modal", "Spatial Encoder"], learned=True, font=17)
    c.module(255, 320, 185, 72, ["Direct Spatial Cache", "≤16 frames × 64"], learned=False, border=ORANGE, layers=3, font=17)
    c.line(300, 240, 347, 320, color=ORANGE)
    c.line(635, 240, 648, 280, color=BLUE)
    c.path("M 565 316 C 515 316, 495 356, 442 356", color=ORANGE)
    c.module(255, 435, 185, 68, ["Adaptive Avg Pool", "1024 → 2 tokens"], learned=False, border=ORANGE, font=17)
    c.line(347, 392, 347, 435, color=ORANGE)
    c.tokens(500, 449, 2, ORANGE_LIGHT, label="Long Memory Emb.")
    c.line(440, 469, 500, 469, color=ORANGE)
    c.tokens(545, 385, 2, BLUE_LIGHT, label="Working Memory Emb.")
    c.line(648, 352, 580, 385, color=BLUE)
    predictor(c, 540, 545, 210, 95)
    c.line(532, 487, 610, 545, color=ORANGE)
    c.line(575, 423, 650, 545, color=BLUE)
    c.action(575, 690)
    c.line(645, 640, 645, 690)
    c.multiline(102, 345, ["Historical RGB-D", "only; 4 views", "4×4 grid/view"], size=16, anchor="start", color=ORANGE)
    c.multiline(80, 575, ["No learned memory module", "Fixed output: 2 tokens"], size=17, anchor="start", weight="700")

    # Training
    c.text(860, 135, "Episode Start", size=18, anchor="start")
    c.frame_stack(875, 180, count=5, color=ORANGE, scale=.82)
    c.frame_stack(1040, 180, count=2, color=BLUE, scale=.82)
    c.text(965, 158, "Long History", size=17, color=ORANGE)
    c.text(1115, 158, "Short Context", size=17, color=BLUE)
    c.module(1180, 155, 155, 62, ["Shared Spatial", "Encoder"], learned=True, font=16)
    c.line(1015, 205, 1180, 182, color=ORANGE)
    c.line(1120, 205, 1180, 195, color=BLUE)
    c.module(980, 255, 175, 60, ["Random Subsample", "max 16 frames"], learned=False, border=ORANGE, font=16)
    c.path("M 1245 217 C 1210 235, 1160 270, 1155 280", color=ORANGE)
    c.module(1200, 255, 180, 60, ["Adaptive Avg Pool", "→ 2 history tokens"], learned=False, border=ORANGE, font=16)
    c.line(1155, 285, 1200, 285, color=ORANGE)
    c.tokens(1400, 270, 2, ORANGE_LIGHT, w=26, h=34)
    c.line(1380, 285, 1400, 285, color=ORANGE)
    c.tokens(1320, 350, 2, BLUE_LIGHT, w=26, h=34, label="context")
    c.path("M 1280 217 C 1280 320, 1300 330, 1320 350", color=BLUE)
    c.path("M 1452 287 C 1460 305, 1430 315, 1420 330", color=ORANGE)
    c.line(1372, 367, 1420, 330, color=BLUE)
    noise_training(c, 850, 330, 1420, 330)

    # Inference
    c.line(865, 535, 1710, 535, sw=1.7)
    for x, lab in [(930, "t−1"), (1050, "t"), (1170, "t+1"), (1290, "t+2"), (1410, "t+3")]:
        c.line(x, 527, x, 543, sw=1.2, arrow=False)
        c.text(x, 518, lab, size=17)
    c.frame_stack(900, 580, count=4, color=ORANGE, scale=.78)
    c.frame_stack(1110, 580, count=2, color=BLUE, scale=.78)
    c.text(1010, 655, "1. Encode new observations", size=17)
    c.module(1240, 570, 175, 62, ["Update Spatial Cache", "anchor + recent ≤16"], learned=False, border=ORANGE, layers=3, font=16)
    c.line(1045, 595, 1240, 600, color=ORANGE)
    c.module(1240, 675, 175, 62, ["Adaptive Avg Pool", "→ 2 tokens"], learned=False, border=ORANGE, font=16)
    c.line(1327, 632, 1327, 675, color=ORANGE)
    c.module(1490, 660, 150, 70, ["Noise", "Predictor"], learned=True, font=17)
    c.line(1415, 706, 1490, 695, color=ORANGE)
    c.path("M 1180 610 C 1370 610, 1400 675, 1490 680", color=BLUE)
    c.action(1488, 770, label="Next Action Chunk")
    c.line(1565, 730, 1565, 770)
    c.text(1260, 785, "2. Fixed compression", size=17)
    c.text(1515, 637, "3. Predict", size=17)
    legend(c)
    return c.finish()


def draw_direct() -> str:
    c = Canvas("Direct — Full Spatial History Tokens without Compression")
    panel_headers(c)
    top_timeline(c)
    c.frame_stack(130, 192, count=5, color=ORANGE, fade=True)
    c.text(230, 175, "Long History", size=18, color=ORANGE)
    c.frame_stack(575, 192, count=2, color=BLUE)
    c.text(640, 175, "Context Window (2)", size=18, color=BLUE)
    c.module(565, 280, 165, 72, ["Shared Multi-modal", "Spatial Encoder"], learned=True, font=17)
    c.module(250, 330, 200, 78, ["Direct Spatial Cache", "≤16 frames × 64", "= ≤1024 tokens"], learned=False, border=ORANGE, layers=3, font=16)
    c.line(300, 240, 350, 330, color=ORANGE)
    c.line(635, 240, 648, 280, color=BLUE)
    c.path("M 565 316 C 510 316, 500 367, 452 367", color=ORANGE)
    c.tokens(480, 455, 7, ORANGE_LIGHT, w=22, h=35, gap=2, label="Full Long-Memory Token Sequence")
    c.line(350, 408, 480, 472, color=ORANGE)
    c.text(640, 510, "… up to 1024", size=15, italic=True, color=ORANGE)
    c.tokens(550, 385, 2, BLUE_LIGHT, label="Working Memory Emb.")
    c.line(648, 352, 585, 385, color=BLUE)
    predictor(c, 540, 560, 210, 95)
    c.line(575, 490, 615, 560, color=ORANGE)
    c.line(585, 423, 655, 560, color=BLUE)
    c.action(575, 700)
    c.line(645, 655, 645, 700)
    c.multiline(95, 520, ["No memory compressor", "No token reduction"], size=18, anchor="start", weight="700")

    # Training
    c.text(860, 135, "Episode Start", size=18, anchor="start")
    c.frame_stack(875, 180, count=5, color=ORANGE, scale=.82)
    c.frame_stack(1040, 180, count=2, color=BLUE, scale=.82)
    c.text(965, 158, "Long History", size=17, color=ORANGE)
    c.text(1115, 158, "Short Context", size=17, color=BLUE)
    c.module(1180, 155, 155, 62, ["Shared Spatial", "Encoder"], learned=True, font=16)
    c.line(1015, 205, 1180, 182, color=ORANGE)
    c.line(1120, 205, 1180, 195, color=BLUE)
    c.module(970, 255, 190, 60, ["Random Subsample", "max 16 frames"], learned=False, border=ORANGE, font=16)
    c.path("M 1245 217 C 1210 235, 1165 270, 1160 280", color=ORANGE)
    c.module(1200, 255, 170, 60, ["Flatten + Concatenate", "≤1024 tokens"], learned=False, border=ORANGE, font=16)
    c.line(1160, 285, 1200, 285, color=ORANGE)
    c.tokens(1390, 270, 5, ORANGE_LIGHT, w=20, h=34, gap=1)
    c.line(1370, 285, 1390, 285, color=ORANGE)
    c.tokens(1320, 350, 2, BLUE_LIGHT, w=26, h=34, label="context")
    c.path("M 1280 217 C 1280 320, 1300 330, 1320 350", color=BLUE)
    c.path("M 1495 287 C 1495 310, 1450 320, 1430 330", color=ORANGE)
    c.line(1372, 367, 1430, 330, color=BLUE)
    noise_training(c, 850, 330, 1430, 330)

    # Inference
    c.line(865, 535, 1710, 535, sw=1.7)
    for x, lab in [(930, "t−1"), (1050, "t"), (1170, "t+1"), (1290, "t+2"), (1410, "t+3")]:
        c.line(x, 527, x, 543, sw=1.2, arrow=False)
        c.text(x, 518, lab, size=17)
    c.frame_stack(900, 580, count=4, color=ORANGE, scale=.78)
    c.frame_stack(1110, 580, count=2, color=BLUE, scale=.78)
    c.text(1010, 655, "1. Encode new observations", size=17)
    c.module(1240, 570, 185, 62, ["Update Spatial Cache", "anchor + recent ≤16"], learned=False, border=ORANGE, layers=3, font=16)
    c.line(1045, 595, 1240, 600, color=ORANGE)
    c.tokens(1245, 690, 6, ORANGE_LIGHT, w=20, h=34, gap=2, label="2. Use all cached tokens")
    c.line(1332, 632, 1332, 690, color=ORANGE)
    c.module(1490, 660, 150, 70, ["Noise", "Predictor"], learned=True, font=17)
    c.line(1380, 707, 1490, 695, color=ORANGE)
    c.path("M 1180 610 C 1380 610, 1400 675, 1490 680", color=BLUE)
    c.action(1488, 770, label="Next Action Chunk")
    c.line(1565, 730, 1565, 770)
    c.text(1515, 637, "3. Predict", size=17)
    legend(c)
    return c.finish()


def draw_no_history() -> str:
    c = Canvas("No-History — Short-Context Diffusion Policy (No Long-Term Memory)")
    panel_headers(c)
    top_timeline(c)
    c.frame_stack(575, 195, count=2, color=BLUE)
    c.text(640, 175, "Context Window (2)", size=18, color=BLUE)
    c.module(540, 305, 205, 76, ["Multi-modal Encoder", "+ Task Embedding"], learned=True, font=18)
    c.line(640, 242, 640, 305, color=BLUE)
    c.tokens(585, 440, 2, BLUE_LIGHT, label="Working Memory Emb.")
    c.line(642, 381, 642, 440, color=BLUE)
    predictor(c, 540, 560, 210, 95)
    c.line(642, 478, 642, 560, color=BLUE)
    c.action(575, 700)
    c.line(645, 655, 645, 700)
    c.rect(105, 305, 275, 180, fill=LIGHT, stroke="#888", sw=1.5, rx=8, dash="7 6")
    c.multiline(243, 350, ["No Long-History Input", "No Observation Cache", "No Memory Compressor", "0 Long-Memory Tokens"], size=19, weight="700", gap=31)
    c.text(245, 530, "Only the latest two observations condition the policy.", size=16)

    # Training
    c.frame_stack(910, 170, count=2, color=BLUE, scale=.9)
    c.text(970, 150, "Short Context Only", size=18, color=BLUE)
    c.module(1110, 150, 175, 66, ["Multi-modal Encoder", "+ Task Embedding"], learned=True, font=16)
    c.line(1020, 185, 1110, 183, color=BLUE)
    c.tokens(1320, 166, 2, BLUE_LIGHT, w=28, h=36, label="context")
    c.line(1285, 183, 1320, 183, color=BLUE)
    noise_training(c, 850, 290, 1390, 184)
    c.rect(1445, 135, 255, 95, fill=LIGHT, stroke="#888", sw=1.3, rx=7, dash="6 5")
    c.multiline(1572, 166, ["No history sampling", "No long-memory branch"], size=16, weight="700")

    # Inference
    c.line(865, 535, 1710, 535, sw=1.7)
    for x, lab in [(930, "t−1"), (1050, "t"), (1170, "t+1"), (1290, "t+2"), (1410, "t+3")]:
        c.line(x, 527, x, 543, sw=1.2, arrow=False)
        c.text(x, 518, lab, size=17)
    c.frame_stack(925, 585, count=2, color=BLUE, scale=.9)
    c.text(990, 660, "1. Read latest 2 observations", size=17)
    c.module(1155, 570, 175, 68, ["Multi-modal Encoder", "+ Task Embedding"], learned=True, font=16)
    c.line(1040, 605, 1155, 605, color=BLUE)
    c.tokens(1370, 585, 2, BLUE_LIGHT, w=28, h=38, label="context")
    c.line(1330, 604, 1370, 604, color=BLUE)
    c.module(1490, 570, 150, 70, ["Noise", "Predictor"], learned=True, font=17)
    c.line(1430, 604, 1490, 604, color=BLUE)
    c.action(1498, 730, label="Next Action Chunk")
    c.line(1565, 640, 1565, 730)
    c.path("M 1640 603 C 1695 603, 1700 520, 1655 500", color="#777", sw=1.8, dash="7 6")
    c.multiline(1260, 700, ["2. Predict and execute 8 actions", "3. Slide the 2-frame context window"], size=18, weight="700")
    c.rect(900, 750, 400, 62, fill=LIGHT, stroke="#888", sw=1.3, rx=7, dash="6 5")
    c.text(1100, 787, "The policy cache is cleared before every action chunk.", size=16)
    legend(c)
    return c.finish()


def main() -> None:
    out = Path("reports/momart_1m_four_methods/assets")
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "direct_token2_architecture.svg": draw_direct_token2(),
        "direct_architecture.svg": draw_direct(),
        "no_history_architecture.svg": draw_no_history(),
    }
    for name, svg in figures.items():
        (out / name).write_text(svg, encoding="utf-8")
        print(out / name)


if __name__ == "__main__":
    main()
