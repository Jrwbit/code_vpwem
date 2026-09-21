#!/usr/bin/env python3
"""Build a four-page, fully editable diagrams.net MoMaRT architecture file."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/assets/momart_four_methods_1m/architectures.drawio"

PALETTE = {
    "input": ("#E0F2FE", "#0284C7"),
    "encoder": ("#DBEAFE", "#2563EB"),
    "memory": ("#F3E8FF", "#7C3AED"),
    "condition": ("#DCFCE7", "#16A34A"),
    "action": ("#FFEDD5", "#EA580C"),
    "muted": ("#F1F5F9", "#64748B"),
}


class Page:
    def __init__(self, parent: ET.Element, name: str, description: str) -> None:
        self.diagram = ET.SubElement(parent, "diagram", id=name.lower().replace("-", "_"), name=name)
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            dx="1600", dy="900", grid="1", gridSize="10", guides="1",
            tooltips="1", connect="1", arrows="1", fold="1", page="1",
            pageScale="1", pageWidth="2100", pageHeight="1120",
            math="0", shadow="0",
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")
        self.next_id = 2
        self.label(name, 55, 30, 1800, 58, 36, bold=True)
        self.label(description, 55, 94, 1900, 40, 19, color="#64748B")
        self.label("当前观测路径", 65, 170, 450, 38, 24, bold=True, color="#1D4ED8")
        self.label("长期历史路径", 65, 500, 450, 38, 24, bold=True, color="#7C3AED")

    def _id(self) -> str:
        value = str(self.next_id)
        self.next_id += 1
        return value

    def node(self, title: str, detail: str, x: int, y: int, w: int = 225,
             h: int = 112, kind: str = "encoder") -> str:
        fill, stroke = PALETTE[kind]
        value = f"<b>{title}</b><br><font style='font-size:15px'>{detail}</font>"
        cell = ET.SubElement(
            self.root, "mxCell", id=self._id(), value=value, vertex="1", parent="1",
            style=(
                "rounded=1;whiteSpace=wrap;html=1;arcSize=14;"
                f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
                "fontColor=#172033;fontSize=19;align=center;verticalAlign=middle;"
                "spacing=10;"
            ),
        )
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w),
                      height=str(h), **{"as": "geometry"})
        return cell.attrib["id"]

    def label(self, value: str, x: int, y: int, w: int, h: int, size: int,
              *, bold: bool = False, color: str = "#172033") -> None:
        cell = ET.SubElement(
            self.root, "mxCell", id=self._id(), value=value, vertex="1", parent="1",
            style=(
                "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                "align=left;verticalAlign=middle;spacingLeft=0;"
                f"fontSize={size};fontColor={color};fontStyle={1 if bold else 0};"
            ),
        )
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w),
                      height=str(h), **{"as": "geometry"})

    def edge(self, source: str, target: str, label: str = "", *, dashed: bool = False) -> None:
        cell = ET.SubElement(
            self.root, "mxCell", id=self._id(), value=label, edge="1", parent="1",
            source=source, target=target,
            style=(
                "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
                "jettySize=auto;html=1;endArrow=block;endFill=1;"
                f"strokeColor={'#94A3B8' if dashed else '#475569'};strokeWidth=2;"
                f"dashed={1 if dashed else 0};fontSize=15;fontColor=#475569;"
                "labelBackgroundColor=#FFFFFF;"
            ),
        )
        ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})

    def common_action(self, recent: str, history: str | None, tm: int) -> None:
        condition = self.node(
            "条件序列", f"[扩散步, 历史({tm}), 近期观测(2)]<br>PTP 时间对齐 mask",
            1420, 350, 270, 138, "condition",
        )
        decoder = self.node(
            "动作去噪网络", "8 层 PTP Transformer<br>256维 · 4头 · FFN 1024",
            1760, 350, 285, 138, "action",
        )
        ddpm = self.node("DDPM", "50 次反向去噪<br>9步 × 10维动作", 1760, 570, 285, 112, "action")
        action = self.node("在线执行", "每次执行 8 步动作", 1760, 760, 285, 100, "action")
        self.edge(recent, condition)
        if history is not None:
            self.edge(history, condition)
        self.edge(condition, decoder)
        self.edge(decoder, ddpm)
        self.edge(ddpm, action)
        self.label(
            "共同骨干：带噪动作使用因果自注意力；近期观测按 PTP 掩码可见；长期历史对所有动作位置可见。",
            65, 965, 1900, 55, 19, color="#475569",
        )


def make_no_history(parent: ET.Element) -> None:
    p = Page(parent, "No-History", "近期观测基线｜不使用长期历史；历史 token 数 0")
    raw = p.node("最近 2 帧观测", "RGB×2 · 深度×2<br>proprio · scan", 70, 250, 260, 125, "input")
    visual = p.node("多模态编码", "四路 ResNet-18<br>低维特征融合 + MLP", 400, 250, 260, 125)
    task = p.node("任务条件", "加入可学习 task embedding<br>得到 [B,2,256]", 740, 250, 270, 125)
    missing = p.node("历史关闭", "memory=None<br>Tm=0；无 Q-Former", 70, 600, 260, 125, "muted")
    p.edge(raw, visual)
    p.edge(visual, task)
    p.common_action(task, None, 0)
    p.label("相对基准：完整保留动作 Transformer，仅关闭长期历史输入。", 70, 850, 1450, 55, 21, bold=True)


def make_direct(parent: ET.Element, *, pooled: bool) -> None:
    name = "Direct-token2" if pooled else "Direct"
    subtitle = (
        "无参数压缩｜与 Direct 共用空间视觉编码，将 1024 个历史 token 平均池化为 2 个"
        if pooled else "完整空间历史｜16 帧 × 4 路视觉 × 4×4 网格 = 最多 1024 个历史 token"
    )
    p = Page(parent, name, subtitle)
    current = p.node("最近 2 帧 raw 观测", "四路视觉 + proprio + scan", 70, 250, 255, 115, "input")
    current_encoder = p.node("当前观测全局编码", "ResNet-18 + MLP<br>加 task embedding", 405, 250, 270, 115)
    recent = p.node("近期条件", "[B,2,256]", 760, 250, 245, 115, "condition")
    p.edge(current, current_encoder)
    p.edge(current_encoder, recent)

    raw = p.node("最多 16 帧历史", "仅四路 RGB/深度<br>无历史 proprio/scan", 70, 590, 255, 128, "input")
    trunk = p.node("空间视觉主干", "每路 ResNet trunk<br>AdaptiveAvgPool2d(4×4)", 405, 590, 280, 128)
    spatial = p.node("空间 token", "每格投影至 256维<br>加空间位置与视角编码", 775, 590, 280, 128, "memory")
    p.edge(raw, trunk)
    p.edge(trunk, spatial)
    if pooled:
        packed = p.node("固定平均池化", "1024 → 2 token<br>AdaptiveAvgPool1d(2)", 1125, 590, 245, 128, "memory")
        p.edge(spatial, packed)
        history, tm = packed, 2
        change = "相对 Direct：仅在历史送入 Decoder 前做无参数平均池化；视觉主干仍可训练。"
    else:
        packed = p.node("完整历史条件", "16×4×16 = 1024 token<br>[B,1024,256]", 1125, 590, 245, 128, "memory")
        p.edge(spatial, packed)
        history, tm = packed, 1024
        change = "相对 No-History：新增空间历史分支，完整历史 token 直接进入交叉注意力。"
    p.common_action(recent, history, tm)
    p.label(change, 70, 850, 1640, 55, 21, bold=True)


def make_vpwem(parent: ET.Element) -> None:
    p = Page(parent, "VPWEM", "学习式情景记忆｜预计算 dp_emb + FIFO 缓存 + 2 层 Q-Former → 2 个记忆 token")
    current = p.node("最近 2 帧 dp_emb", "预计算 256维 embedding", 70, 250, 255, 115, "input")
    task = p.node("任务条件", "加 task embedding", 405, 250, 250, 115)
    recent = p.node("近期条件", "[B,2,256]", 750, 250, 245, 115, "condition")
    p.edge(current, task)
    p.edge(task, recent)

    history = p.node("历史 dp_emb", "最多 16 帧<br>每帧 256维", 70, 590, 225, 128, "input")
    position = p.node("时间编码", "加入 episode 内的<br>正弦位置编码", 350, 590, 225, 128)
    fifo = p.node("FIFO 观测缓存", "容量 8<br>保留最近历史证据", 630, 590, 225, 128, "memory")
    queries = p.node("可学习 query", "2 个 query<br>读取历史 query 摘要缓存", 630, 790, 225, 110, "memory")
    qformer = p.node("Q-Former ×2", "自注意力 → 交叉注意力<br>→ 前馈网络；8头", 925, 590, 240, 128, "memory")
    tokens = p.node("记忆摘要", "2 token<br>[B,2,256]", 1205, 590, 190, 128, "memory")
    for a, b in ((history, position), (position, fifo), (fifo, qformer),
                 (queries, qformer), (qformer, tokens)):
        p.edge(a, b)
    p.common_action(recent, tokens, 2)
    p.label("相对 Direct-token2：固定池化改为可学习记忆；当前实验还使用不同的预计算 embedding 输入链路。", 70, 915, 1640, 48, 20, bold=True)


def main() -> None:
    mxfile = ET.Element("mxfile", host="app.diagrams.net", type="device", version="24.7.17")
    make_no_history(mxfile)
    make_direct(mxfile, pooled=False)
    make_direct(mxfile, pooled=True)
    make_vpwem(mxfile)
    ET.indent(mxfile, space="  ")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mxfile).write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {OUTPUT} ({len(mxfile)} editable pages)")


if __name__ == "__main__":
    main()
