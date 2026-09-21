#!/usr/bin/env python3
"""Build module-level editable MoMaRT diagrams with tensor content and shape."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from build_momart_editable_diagrams import Page


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/assets/momart_four_methods_1m/architectures_detailed.drawio"


class DetailPage(Page):
    def __init__(self, parent: ET.Element, name: str, summary: str) -> None:
        self.diagram = ET.SubElement(parent, "diagram", id="detail_" + name.lower().replace("-", "_"), name=name)
        self.model = ET.SubElement(
            self.diagram, "mxGraphModel", dx="1900", dy="1200", grid="1",
            gridSize="10", guides="1", tooltips="1", connect="1", arrows="1",
            fold="1", page="1", pageScale="1", pageWidth="3400",
            pageHeight="2430", math="0", shadow="0",
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")
        self.next_id = 2
        self.label(name + "：模块级张量流", 55, 25, 3200, 62, 39, bold=True)
        self.label(summary, 55, 92, 3200, 42, 20, color="#475569")
        self.label("① 最近观测：输入 → 各模态编码 → 近期条件", 75, 155, 1450, 43, 25, bold=True, color="#1D4ED8")
        self.label("② 长期历史：采样 → 编码/压缩 → 历史条件", 1650, 155, 1660, 43, 25, bold=True, color="#7C3AED")
        self.label("③ 扩散动作骨干：条件 + 带噪动作 → 去噪 → 执行", 75, 1790, 3120, 43, 25, bold=True, color="#EA580C")


def raw_current(p: DetailPage, *, include_task: bool = True) -> str:
    names = ("正面 RGB", "腕部 RGB", "正面深度", "腕部深度")
    channels = (3, 3, 1, 1)
    outputs = []
    for index, (name, channels_in) in enumerate(zip(names, channels)):
        y = 225 + index * 170
        raw = p.node(name, f"[B,2,{channels_in},90,120]<br>原始视觉观测", 75, y, 250, 112, "input")
        crop = p.node("随机裁剪", f"[B,2,{channels_in},75,100]<br>训练期裁剪", 390, y, 250, 112)
        resnet = p.node("独立 ResNet-18", "GroupNorm；trunk [B,2,512,3,4]<br>全局池化 → [B,2,512]", 705, y, 280, 112)
        p.edge(raw, crop)
        p.edge(crop, resnet)
        outputs.append(resnet)
    proprio = p.node("proprio", "机器人本体状态<br>[B,2,10]", 75, 925, 250, 112, "input")
    scan = p.node("scan", "一维扫描数据<br>[B,2,1,100]", 390, 925, 250, 112, "input")
    scan_encoder = p.node("ScanCore", "Conv1d 32→64→64<br>SpatialSoftmax → [B,2,64]", 705, 925, 280, 112)
    p.edge(scan, scan_encoder)
    merge = p.node("特征拼接", "4×512 + 10 + 64<br>输出 [B,2,2122]", 1080, 975, 310, 120, "encoder")
    for output in (*outputs, proprio, scan_encoder):
        p.edge(output, merge)
    mlp = p.node("多模态 MLP", "Linear 2122→256<br>LeakyReLU→Linear 256→256<br>[B,2,256]", 1080, 1195, 310, 135)
    if include_task:
        task_id = p.node("任务 ID", "每个样本 1 个任务编号<br>[B]", 390, 1415, 210, 125, "input")
        task_emb = p.node("任务 embedding", "Embedding(5,256)<br>[B,256]", 670, 1415, 300, 125, "condition")
        task = p.node("逐帧相加", "观测 [B,2,256] + 任务 [B,1,256]<br>输出 [B,2,256]", 1080, 1415, 310, 125, "condition")
        p.edge(task_id, task_emb)
        p.edge(task_emb, task)
        recent_y = 1610
    else:
        task = None
        recent_y = 1450
    recent = p.node("近期观测 token", "2 个全局 token<br>[B,2,256]", 1080, recent_y, 310, 115, "condition")
    path = ((merge, mlp), (mlp, task), (task, recent)) if task else ((merge, mlp), (mlp, recent))
    for a, b in path:
        p.edge(a, b)
    return recent


def raw_history(p: DetailPage, *, pool_two: bool) -> str:
    names = ("正面 RGB", "腕部 RGB", "正面深度", "腕部深度")
    channels = (3, 3, 1, 1)
    tokens = []
    for index, (name, channels_in) in enumerate(zip(names, channels)):
        y = 225 + index * 170
        raw = p.node(name + " 历史", f"[B,H,{channels_in},90,120]<br>H≤16，时间顺序保留", 1650, y, 265, 112, "input")
        trunk = p.node("裁剪 + ResNet trunk", f"[B,H,{channels_in},75,100]<br>特征图 [B,H,512,3,4]", 1975, y, 285, 112)
        spatial = p.node("4×4 池化 + 投影", "16 个空间位置/视角<br>Linear 512→256<br>[B,H,16,256]", 2320, y, 290, 112, "memory")
        p.edge(raw, trunk)
        p.edge(trunk, spatial)
        tokens.append(spatial)
    concat = p.node("四视角拼接", "4×16 = 64 token/帧<br>[B,H,64,256]", 2730, 705, 300, 120, "memory")
    for item in tokens:
        p.edge(item, concat)
    pos = p.node("空间/视角/任务 embedding", "空间位置 + 相机视角 + 任务<br>[B,H,64,256]", 2730, 945, 300, 120, "memory")
    pad = p.node("历史长度规整", "首帧锚点 + 最近帧；不足补齐<br>[B,16,64,256]", 2730, 1165, 300, 120, "memory")
    flat = p.node("展平帧与空间轴", "16×64 = 1024 token<br>[B,1024,256]", 2730, 1385, 300, 120, "memory")
    for a, b in ((concat, pos), (pos, pad), (pad, flat)):
        p.edge(a, b)
    if pool_two:
        pooled = p.node("无参数平均池化", "AdaptiveAvgPool1d(2)<br>[B,2,256]", 2730, 1580, 300, 120, "memory")
        p.edge(flat, pooled)
        return pooled
    return flat


def vpwem_current(p: DetailPage) -> str:
    emb = p.node("当前 dp_emb", "外部 encoder 预计算<br>[B,2,256]", 75, 290, 305, 125, "input")
    task_id = p.node("任务 ID", "每个样本 1 个任务编号<br>[B]", 75, 480, 305, 110, "input")
    task_emb = p.node("任务 embedding", "Embedding(5,256)<br>[B,256]", 470, 480, 305, 110, "condition")
    add = p.node("逐帧相加", "dp_emb [B,2,256] + 任务 [B,1,256]<br>输出 [B,2,256]", 880, 290, 310, 125, "condition")
    recent = p.node("近期观测 token", "2 个全局 token<br>[B,2,256]", 1280, 290, 310, 125, "condition")
    p.edge(emb, add)
    p.edge(task_id, task_emb)
    p.edge(task_emb, add)
    p.edge(add, recent)
    return recent


def vpwem_history(p: DetailPage) -> str:
    emb = p.node("历史 dp_emb", "最多 16 帧 × 256维<br>[B,H,256]，H≤16", 1650, 270, 265, 115, "input")
    task = p.node("任务 embedding", "与当前观测共用 5×256 表<br>[B,H,256]", 1970, 270, 275, 115)
    pos = p.node("episode 时间编码", "sinusoidal(ep_step) 相加<br>[B,H,256]", 2300, 270, 280, 115)
    fifo = p.node("FIFO 观测缓存", "按时序逐帧写入；容量 8<br>[B,≤8,1,256]", 2700, 270, 300, 115, "memory")
    projection = p.node("Q-Former 输入投影", "Linear 256→256<br>缓存展平 [B,≤8,256]", 2700, 465, 300, 115, "memory")
    for a, b in ((emb, task), (task, pos), (pos, fifo), (fifo, projection)):
        p.edge(a, b)
    query = p.node("2 个 learned queries", "每层输入 [B,2,256]", 1660, 660, 280, 112, "memory")
    cache = p.node("历史 query 摘要缓存", "每层各维护 FIFO<br>[B,≤8,2,256]", 1995, 660, 285, 112, "memory")
    self_attn = p.node("自注意力 ×2 层", "query 读取历史摘要<br>输出 [B,2,256]", 2390, 660, 290, 112, "memory")
    cross = p.node("交叉注意力 ×2 层", "query 读取 FIFO 观测缓存<br>输出 [B,2,256]", 2730, 660, 295, 112, "memory")
    ffn = p.node("FFN + 残差 + LayerNorm", "256→1024→256；dropout 0.1<br>输出 [B,2,256]", 2730, 910, 295, 118, "memory")
    output = p.node("Q-Former 输出", "LayerNorm → Linear 256→256<br>2 个记忆 token [B,2,256]", 2730, 1235, 295, 120, "memory")
    p.edge(query, self_attn)
    p.edge(cache, self_attn)
    p.edge(self_attn, cross)
    p.edge(projection, cross)
    p.edge(cross, ffn)
    p.edge(ffn, output)
    p.label("Q-Former：2 层；每层 8 个注意力头；历史 query 在写入缓存前 detach。", 1680, 1470, 1550, 45, 19, color="#475569")
    return output


def action_backbone(p: DetailPage, recent: str, memory: str | None, tm: int,
                    *, ptp: bool = True) -> None:
    noise_step = p.node("扩散时间步 τ", "PositionalEmbedding<br>[B,1,256]", 1120, 1860, 250, 112, "input")
    cond = p.node("条件 token 拼接", f"[τ, memory({tm}), obs(2)]<br>[B,{tm + 3},256]", 1460, 1860, 290, 112, "condition")
    cond_mlp = p.node("条件逐 token MLP", f"位置编码；256→1024→256<br>Mish；[B,{tm + 3},256]", 1830, 1860, 290, 112, "condition")
    p.edge(noise_step, cond)
    p.edge(recent, cond)
    if memory is not None:
        p.edge(memory, cond)
    p.edge(cond, cond_mlp)

    noisy = p.node("带噪动作 xₜ", "归一化 9 步 × 10维<br>[B,9,10]", 75, 2070, 265, 112, "input")
    linear = p.node("动作线性映射", "Linear 10→256<br>[B,9,256]", 425, 2070, 270, 112)
    position = p.node("动作位置编码", "加可学习 position embedding<br>[B,9,256]", 780, 2070, 280, 112)
    cross = "PTP 时间对齐 cross-attn" if ptp else "全可见条件 cross-attn"
    decoder = p.node("Transformer Decoder ×8", f"因果 self-attn + {cross}<br>4头；FFN 256→1024→256<br>[B,9,256]", 2180, 2040, 335, 140, "action")
    head = p.node("LayerNorm + 输出头", "Linear 256→10<br>预测噪声 εθ [B,9,10]", 2570, 2040, 270, 140, "action")
    ddpm = p.node("DDPM 反向去噪", "循环 50 次<br>动作序列 [B,9,10]", 2890, 2040, 250, 140, "action")
    execute = p.node("在线执行", "取动作块 8 步<br>[B,8,10]", 3150, 2040, 215, 140, "action")
    for a, b in ((noisy, linear), (linear, position), (position, decoder),
                 (cond_mlp, decoder), (decoder, head), (head, ddpm), (ddpm, execute)):
        p.edge(a, b)
    note = (
        "PTP 掩码：动作 self-attention 为因果掩码；长期历史对所有动作位置可见；第 j 帧近期观测仅对第 j 个及之后动作位置可见。"
        if ptp else
        "DP 掩码：动作 self-attention 为因果掩码；扩散时间步和两个近期观测 token 对全部 9 个动作位置可见。"
    )
    p.label(note, 75, 2250, 3250, 54, 20, color="#475569")


def main() -> None:
    parent = ET.Element("mxfile", host="app.diagrams.net", type="device", version="24.7.17")
    nohistory = DetailPage(parent, "No-History", "原始图像链路；长期历史关闭。B=批量，H=历史帧数。")
    recent = raw_current(nohistory)
    nohistory.node("不使用长期历史", "memory=None；Tm=0<br>无需 Q-Former", 2100, 850, 330, 132, "muted")
    action_backbone(nohistory, recent, None, 0)

    for name, pool_two in (("Direct", False), ("Direct-token2", True)):
        page = DetailPage(parent, name, "原始图像链路；当前观测与空间历史分别编码。B=批量，H≤16。")
        recent = raw_current(page)
        memory = raw_history(page, pool_two=pool_two)
        action_backbone(page, recent, memory, 2 if pool_two else 1024)

    vpwem = DetailPage(parent, "VPWEM", "预计算 dp_emb 输入链路；FIFO + 两层 Q-Former 学习 2-token 历史摘要。")
    recent = vpwem_current(vpwem)
    memory = vpwem_history(vpwem)
    action_backbone(vpwem, recent, memory, 2)

    ET.indent(parent, space="  ")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(parent).write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {OUTPUT} ({len(parent)} module-level pages)")


if __name__ == "__main__":
    main()
