#!/usr/bin/env python3
"""Draw code-audited architecture diagrams for DP, DP-PTP and History-4."""

from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/history6_architecture_20260920"
FONT = "Noto Sans CJK SC, Noto Sans SC, sans-serif"


def svg_start(width, height, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<title>{escape(title)}</title>
<defs>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.10"/></filter>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#475569"/></marker>
</defs>
<rect width="100%" height="100%" fill="#f8fafc"/>
'''


def text(x, y, value, size=24, weight=500, fill="#172033", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>')


def multiline(x, y, lines, size=19, weight=500, fill="#172033", anchor="middle", gap=27):
    parts = [f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">']
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else gap
        parts.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    parts.append('</text>')
    return ''.join(parts)


def rounded(x, y, w, h, fill, stroke, radius=18, sw=3, shadow=False):
    filt = ' filter="url(#shadow)"' if shadow else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{filt}/>'


def arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="9 7"' if dashed else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="3" marker-end="url(#arrow)"{dash}/>'


def pill(x, y, label, fill, stroke, width=None):
    width = width or max(100, len(label) * 19 + 30)
    return (rounded(x, y, width, 38, fill, stroke, 19, 2) +
            text(x + width / 2, y + 27, label, 17, 700, stroke, "middle"))


PALETTE = {
    "input": ("#e0f2fe", "#0284c7"),
    "encoder": ("#dbeafe", "#2563eb"),
    "memory": ("#ede9fe", "#7c3aed"),
    "decoder": ("#dcfce7", "#16a34a"),
    "output": ("#ffedd5", "#ea580c"),
    "none": ("#f1f5f9", "#64748b"),
}


def stage_box(x, y, w, h, lines, kind):
    fill, stroke = PALETTE[kind]
    body = [rounded(x, y, w, h, fill, stroke, 16, 3)]
    total = (len(lines) - 1) * 25
    body.append(multiline(x + w / 2, y + h / 2 - total / 2 + 7, lines, 18, 650, "#172033", "middle", 25))
    return ''.join(body)


def architecture_card(x, y, w, h, title_value, subtitle, tag, accent, stages):
    parts = [rounded(x, y, w, h, "#ffffff", "#d8e1ee", 24, 2, True)]
    parts.append(f'<rect x="{x}" y="{y}" width="10" height="{h}" rx="5" fill="{accent}"/>')
    parts.append(text(x + 34, y + 47, title_value, 30, 800))
    parts.append(text(x + 34, y + 78, subtitle, 17, 500, "#64748b"))
    parts.append(pill(x + w - 188, y + 25, tag, "#f8fafc", accent, 155))
    gap = 25
    inner_x = x + 34
    stage_y = y + 115
    stage_h = 145
    widths = [178, 205, 245, 255, 180]
    scale = (w - 68 - gap * 4) / sum(widths)
    widths = [v * scale for v in widths]
    cursor = inner_x
    for i, (lines, kind) in enumerate(stages):
        parts.append(stage_box(cursor, stage_y, widths[i], stage_h, lines, kind))
        if i < len(stages) - 1:
            parts.append(arrow(cursor + widths[i] + 4, stage_y + stage_h / 2,
                               cursor + widths[i] + gap - 5, stage_y + stage_h / 2))
        cursor += widths[i] + gap
    # Bottom tensor summary.
    summary_y = y + h - 44
    parts.append(text(x + 34, summary_y, stages[-1][0][-1], 16, 650, accent))
    return ''.join(parts)


def draw_overview():
    width, height = 2400, 1880
    p = [svg_start(width, height, "六种 Diffusion Policy 网络结构对照")]
    p += [text(90, 82, "六种 Diffusion Policy 网络结构对照", 48, 850),
          text(90, 125, "原版 DP / DP-PTP + 当前 MoMaRT History-4；依据仓库实现绘制", 24, 500, "#64748b"),
          text(2310, 83, "obs=2 · horizon=9 · action chunk=8 · d=256", 21, 650, "#475569", "end")]

    cards = [
        ("原版 DP", "ChiTransformer：没有长期历史，所有动作位置可读取全部近期观测", "基线", "#0ea5e9", [
            (["近期观测", "o₀, o₁"], "input"),
            (["多模态编码", "ResNet / MLP", "→ 2×256"], "encoder"),
            (["无长期历史", "Tm = 0"], "none"),
            (["[τ, o₀, o₁]", "DP cross-attn", "两帧全可见"], "decoder"),
            (["εθ", "DDPM×50", "执行8动作"], "output"),
        ]),
        ("原版 DP-PTP", "ChiTransformerPTP：结构与 DP 相同，改为时间对齐的观测可见性", "PTP 基线", "#2563eb", [
            (["近期观测", "o₀, o₁"], "input"),
            (["多模态编码", "ResNet / MLP", "→ 2×256"], "encoder"),
            (["无长期历史", "Tm = 0"], "none"),
            (["[τ, o₀, o₁]", "PTP mask", "动作 i 仅见 oⱼ≤i"], "decoder"),
            (["εθ", "DDPM×50", "执行8动作"], "output"),
        ]),
        ("No-History", "多任务 fair-history 下界：LongMem 骨干保留，但长期历史 token 数为 0", "0 history token", "#64748b", [
            (["raw 当前观测", "To = 2", "+ task id"], "input"),
            (["全局多模态编码", "+ task embedding", "→ 2×256"], "encoder"),
            (["memory=None", "Tm = 0", "不建 Q-Former"], "none"),
            (["[τ, o₀, o₁]", "LongMem PTP", "Decoder×8"], "decoder"),
            (["εθ", "DDPM×50", "执行8动作"], "output"),
        ]),
        ("Direct", "保留全部空间历史：四路视觉 × 4×4 网格 × 16 帧", "1024 history tokens", "#16a34a", [
            (["raw 当前+历史", "16 history frames", "+ task id"], "input"),
            (["当前：全局编码", "历史：ResNet trunk", "4视图×4×4"], "encoder"),
            (["64 token/帧", "×16 帧", "= 1024×256"], "memory"),
            (["[τ, m₁…m₁₀₂₄,", "o₀, o₁]", "LongMem PTP×8"], "decoder"),
            (["εθ", "DDPM×50", "执行8动作"], "output"),
        ]),
        ("Direct-token2", "与 Direct 共用空间历史编码，然后用无参数池化压成 2 个 token", "2 pooled tokens", "#f59e0b", [
            (["raw 当前+历史", "16 history frames", "+ task id"], "input"),
            (["四路视觉 trunk", "4×4 / 视图", "→ 1024 token"], "encoder"),
            (["AdaptiveAvgPool1d", "1024 → 2", "无可学习参数"], "memory"),
            (["[τ, m₁, m₂,", "o₀, o₁]", "LongMem PTP×8"], "decoder"),
            (["εθ", "DDPM×50", "执行8动作"], "output"),
        ]),
        ("VPWEM", "当前 1M 对照采用 dp_emb 路径：FIFO 有界缓存 + 两层 Q-Former", "2 learned tokens", "#7c3aed", [
            (["当前/历史 dp_emb", "≤16 history frames", "+ task id"], "input"),
            (["任务条件编码", "+ episode 正弦位置", "每帧 256维"], "encoder"),
            (["FIFO cache = 8", "Q-Former×2", "2 learned queries"], "memory"),
            (["[τ, m₁, m₂,", "o₀, o₁]", "LongMem PTP×8"], "decoder"),
            (["εθ", "DDPM×50", "执行8动作"], "output"),
        ]),
    ]
    card_w, card_h = 1080, 470
    positions = [(90, 175), (1230, 175), (90, 685), (1230, 685), (90, 1195), (1230, 1195)]
    for args, pos in zip(cards, positions):
        p.append(architecture_card(pos[0], pos[1], card_w, card_h, *args))

    p.append(rounded(90, 1710, 2220, 105, "#eef2ff", "#c7d2fe", 18, 2))
    p.append(text(120, 1751, "共同骨干", 19, 800, "#4338ca"))
    p.append(text(245, 1751, "带噪动作 xτ → Linear + 动作位置编码 → 8层因果 Transformer Decoder → LayerNorm + Linear → 预测噪声 εθ", 19, 550, "#334155"))
    p.append(text(120, 1788, "关键边界", 19, 800, "#be123c"))
    p.append(text(245, 1788, "Direct/Direct-token2/No-History 是 raw-image、effective batch 64；当前 VPWEM 1M 是 dp_emb、effective batch 256。", 19, 550, "#334155"))
    p.append('</svg>')
    (OUT / "history6_overview.svg").write_text(''.join(p), encoding="utf-8")


def draw_masks():
    width, height = 1800, 1050
    p = [svg_start(width, height, "DP 与 DP-PTP attention mask 对照")]
    p += [text(70, 78, "原版 DP 与 DP-PTP：网络几乎相同，差别在 cross-attention mask", 42, 850),
          text(70, 120, "绿色 = 可读取；灰色 = 被 mask。示意 To=2、Ta=9。τ 是 diffusion timestep token。", 22, 500, "#64748b")]

    def matrix(x, y, title_value, ptp):
        parts = [rounded(x, y, 790, 720, "#ffffff", "#d8e1ee", 24, 2, True),
                 text(x + 35, y + 52, title_value, 30, 800),
                 text(x + 35, y + 86,
                      "动作 self-attention 都是 causal；这里只画条件 cross-attention", 17, 500, "#64748b")]
        cols = ["τ", "o₀", "o₁"]
        cell = 56
        mx, my = x + 250, y + 145
        for c, label in enumerate(cols):
            parts.append(text(mx + c * cell + cell / 2, my - 18, label, 20, 750, "#334155", "middle"))
        for r in range(9):
            parts.append(text(mx - 25, my + r * cell + 36, f"a{r}", 18, 650, "#475569", "end"))
            for c in range(3):
                visible = True if not ptp else (c == 0 or r >= c - 1)
                # c=1(o0): r>=0; c=2(o1): r>=1
                fill = "#bbf7d0" if visible else "#e2e8f0"
                stroke = "#16a34a" if visible else "#94a3b8"
                parts.append(rounded(mx + c * cell, my + r * cell, cell - 5, cell - 5, fill, stroke, 7, 1))
                parts.append(text(mx + c * cell + (cell - 5)/2, my + r * cell + 35,
                                  "✓" if visible else "×", 20, 750, stroke, "middle"))
        if ptp:
            lines = ["PTP：a₀ 只能读取 o₀；", "从 a₁ 开始才能读取 o₁。", "避免未来观测泄漏到更早动作位置。"]
        else:
            lines = ["DP：所有动作位置都能读取", "o₀ 和 o₁。", "近期观测作为完整全局条件。"]
        parts.append(multiline(x + 530, y + 350, lines, 21, 650, "#334155", "middle", 32))
        return ''.join(parts)

    p.append(matrix(70, 180, "DP / ChiTransformer", False))
    p.append(matrix(940, 180, "DP-PTP / ChiTransformerPTP", True))
    p.append(rounded(70, 940, 1660, 70, "#fff7ed", "#fdba74", 16, 2))
    p.append(text(900, 984, "History-4 使用 LongMemChiTransformerPTP：memory token 对所有动作位置可见，近期 observation token 仍遵循 PTP 时间对齐。", 20, 650, "#9a3412", "middle"))
    p.append('</svg>')
    (OUT / "dp_vs_dpptp_masks.svg").write_text(''.join(p), encoding="utf-8")


def draw_method_detail(filename, title_value, accent, left_title, left_lines, memory_title, memory_lines, tokens, note):
    width, height = 1800, 980
    p = [svg_start(width, height, title_value),
         text(70, 80, title_value, 46, 850),
         text(70, 122, note, 21, 500, "#64748b")]
    # Main two branches.
    p.append(rounded(70, 175, 1660, 610, "#ffffff", "#d8e1ee", 26, 2, True))
    p.append(text(115, 230, "当前观测路径", 25, 800, "#1d4ed8"))
    p.append(stage_box(115, 265, 285, 145, [left_title] + left_lines, "input"))
    p.append(arrow(405, 337, 485, 337))
    p.append(stage_box(500, 265, 330, 145, ["当前观测编码"] + (["global 256 token ×2"] if "dp_emb" not in left_title else ["task-conditioned", "dp_emb ×2"]), "encoder"))

    p.append(text(115, 505, "长期历史路径", 25, 800, accent))
    p.append(stage_box(115, 540, 285, 155, [memory_title] + memory_lines[:2], "input" if tokens != "0" else "none"))
    p.append(arrow(405, 617, 485, 617, tokens == "0"))
    mem_kind = "none" if tokens == "0" else "memory"
    p.append(stage_box(500, 540, 330, 155, memory_lines[2:] or ["无历史模块"], mem_kind))

    # Merge and decoder.
    p.append(arrow(835, 337, 975, 435))
    p.append(arrow(835, 617, 975, 505, tokens == "0"))
    p.append(stage_box(990, 360, 310, 220, ["条件序列", f"[τ, memory({tokens}),", "obs(2)]", "memory 全局可见", "obs 用 PTP mask"], "decoder"))
    p.append(arrow(1305, 470, 1380, 470))
    p.append(stage_box(1395, 360, 280, 220, ["Decoder ×8", "causal self-attn", "+ cross-attn", "→ εθ", "DDPM×50"], "output"))

    p.append(rounded(70, 825, 1660, 95, "#f1f5f9", "#cbd5e1", 16, 2))
    p.append(text(100, 865, "张量口径", 19, 800, accent))
    p.append(text(225, 865, f"近期 obs = [B,2,256]；长期 memory = [B,{tokens},256]；带噪动作 = [B,9,10]；输出 εθ = [B,9,10]。", 19, 550, "#334155"))
    p.append(text(100, 900, "动作执行", 19, 800, accent))
    p.append(text(225, 900, "50 次 DDPM 反向去噪得到 9-step action horizon，环境执行其中 8 个动作。", 19, 550, "#334155"))
    p.append('</svg>')
    (OUT / filename).write_text(''.join(p), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw_overview()
    draw_masks()
    draw_method_detail("dp.svg", "原版 DP / ChiTransformer 网络结构", "#0ea5e9",
                       "raw 当前观测", ["To = 2", "RGB / depth / low-dim"],
                       "无长期历史", ["标准 DP 不读 episode 历史", "Tm = 0", "条件为 [τ,o₀,o₁]", "所有动作位都可见 o₀,o₁"],
                       "0", "原版 Transformer Diffusion Policy；近期观测作为对所有动作位置可见的全局条件。")
    draw_method_detail("dpptp.svg", "原版 DP-PTP / ChiTransformerPTP 网络结构", "#2563eb",
                       "raw 当前观测", ["To = 2", "RGB / depth / low-dim"],
                       "无长期历史", ["标准 DP-PTP 不读长历史", "Tm = 0", "条件为 [τ,o₀,o₁]", "aᵢ 仅可见 oⱼ（j≤i）"],
                       "0", "与 DP 使用相同主干；PTP cross-attention mask 阻止未来观测泄漏。")
    draw_method_detail("no_history.svg", "No-History 网络结构", "#64748b",
                       "raw 当前观测", ["RGB×2 + Depth×2", "proprio + scan + task id"],
                       "不读取历史", ["memory=None", "ep_step=None", "Tm = 0；不实例化 Q-Former"],
                       "0", "当前 MoMaRT fair-history 的无长期历史下界；不是移除整个 Transformer。")
    draw_method_detail("direct.svg", "Direct 网络结构", "#16a34a",
                       "raw 当前观测", ["RGB×2 + Depth×2", "proprio + scan + task id"],
                       "raw 历史视觉", ["16帧；仅四路视觉", "不含历史 proprio / scan", "ResNet trunk → 4×4/视图", "64 token/帧 ×16 = 1024"],
                       "1024", "保留完整空间历史 token；信息量最大，也带来最高 cross-attention 成本。")
    draw_method_detail("direct_token2.svg", "Direct-token2 网络结构", "#f59e0b",
                       "raw 当前观测", ["RGB×2 + Depth×2", "proprio + scan + task id"],
                       "raw 历史视觉", ["16帧；四路视觉", "先形成 1024 spatial tokens", "AdaptiveAvgPool1d", "1024 → 2（无参数）"],
                       "2", "与 Direct 共用编码路径，只在送入 decoder 前固定池化到 2 个历史 token。")
    draw_method_detail("vpwem.svg", "VPWEM 网络结构（当前 MoMaRT 1M）", "#7c3aed",
                       "当前 dp_emb", ["预计算 256维", "+ task id"],
                       "历史 dp_emb", ["≤16帧 + episode pos", "subsample ratio = 20", "FIFO cache = 8", "2层 Q-Former → 2 queries"],
                       "2", "当前对照使用 embedding policy；可学习 Q-Former 把可变长历史压成固定 2 token。")
    print(OUT)


if __name__ == "__main__":
    main()
