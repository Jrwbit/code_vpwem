#!/usr/bin/env python3
"""Export an interactive, code-derived architecture viewer for MoMaRT models."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import torch

from print_momart_architectures import SHAPE_META, build


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/momart_architecture_visualization_20260920"
METHODS = ["dp", "dpptp", "no_history", "direct", "direct_token2", "vpwem"]
LABELS = {
    "dp": "原版 DP",
    "dpptp": "原版 DP-PTP",
    "no_history": "No-History",
    "direct": "Direct",
    "direct_token2": "Direct-token2",
    "vpwem": "VPWEM",
}
COLORS = {
    "dp": "#0ea5e9", "dpptp": "#2563eb", "no_history": "#64748b",
    "direct": "#16a34a", "direct_token2": "#f59e0b", "vpwem": "#7c3aed",
}
DESCRIPTIONS = {
    "dp": "无长期历史；ChiTransformer 将两帧近期观测作为所有动作位置均可读取的条件。",
    "dpptp": "与 DP 主干相同；PTP mask 阻止未来近期观测泄漏到更早动作位置。",
    "no_history": "多任务 History-4 下界；保留 LongMem PTP 骨干，但 Tm=0。",
    "direct": "四路视觉历史保留 4×4 空间网格；16 帧形成 1024 个历史 token。",
    "direct_token2": "与 Direct 共用空间编码，再用无参数 AdaptiveAvgPool1d 压为 2 token。",
    "vpwem": "当前 1M 使用 dp_emb；FIFO=8 的两层 Q-Former 以两个 learned query 输出 2 token。",
}


def shape_of(value: Any):
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, dict):
        return {str(k): shape_of(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [shape_of(v) for v in value]
    if value is None:
        return None
    return type(value).__name__


def compact_shape(value: Any) -> str:
    shaped = shape_of(value)
    return json.dumps(shaped, ensure_ascii=False, separators=(",", ":"))


def register_shape_hooks(module, prefix, records, handles):
    for name, child in module.named_modules():
        path = prefix if not name else f"{prefix}.{name}"
        def hook(_module, inputs, output, key=path):
            entry = {"in": compact_shape(inputs), "out": compact_shape(output)}
            records.setdefault(key, [])
            if entry not in records[key]:
                records[key].append(entry)
        handles.append(child.register_forward_hook(hook))


def raw_obs(batch, steps):
    obs = {}
    for key, spec in SHAPE_META["obs"].items():
        obs[key] = torch.zeros(batch, steps, *spec["shape"])
    return obs


def run_shapes(name, built):
    records, handles = {}, []
    for component in ("condition", "memory", "diffusion"):
        module = built[component]
        if module is not None:
            module.eval()
            register_shape_hooks(module, component, records, handles)
    try:
        with torch.no_grad():
            condition_model = built["condition"]
            if name in {"dp", "dpptp"}:
                condition = condition_model(raw_obs(1, 2))
                memory = None
            elif name == "vpwem":
                task = torch.zeros(1, dtype=torch.long)
                condition = condition_model({
                    "dp_emb": torch.zeros(1, 2, 256), "_task_id": task})
                history = condition_model({
                    "dp_emb": torch.zeros(1, 16, 256), "_task_id": task})
                ep_step = torch.arange(16).unsqueeze(0)
                memory = built["memory"](history, ep_step)
            else:
                task = torch.zeros(1, dtype=torch.long)
                current = raw_obs(1, 2)
                current["_task_id"] = task
                condition = condition_model(current)
                if name == "no_history":
                    memory = None
                else:
                    history = raw_obs(1, 16)
                    history["_task_id"] = task
                    memory = condition_model.encode_spatial_history(history)
            x = torch.zeros(1, 9, 10)
            noise = torch.zeros(1)
            if name in {"dp", "dpptp"}:
                built["diffusion"](x, noise, condition)
            else:
                built["diffusion"](x, noise, condition, memory)
    finally:
        for handle in handles:
            handle.remove()
    return records


def parameter_count(module, recurse=True, trainable_only=False):
    if module is None:
        return 0
    return sum(
        p.numel() for p in module.parameters(recurse=recurse)
        if not trainable_only or p.requires_grad)


def module_node(module, path, shapes):
    children = []
    for name, child in module.named_children():
        child_path = f"{path}.{name}"
        children.append(module_node(child, child_path, shapes))
    return {
        "name": path.rsplit(".", 1)[-1],
        "path": path,
        "type": type(module).__name__,
        "params": parameter_count(module),
        "trainable": parameter_count(module, trainable_only=True),
        "direct_params": parameter_count(module, recurse=False),
        "shapes": shapes.get(path, []),
        "children": children,
    }


def mask_data(diffusion):
    mask = diffusion.memory_mask.detach().cpu()
    values = [[bool(torch.isfinite(v)) for v in row] for row in mask]
    tm = int(getattr(diffusion, "Tm", 0))
    labels = ["τ"] + [f"m{i}" for i in range(tm)] + [f"o{i}" for i in range(2)]
    return {"rows": int(mask.shape[0]), "cols": int(mask.shape[1]),
            "tm": tm, "labels": labels, "values": values}


def collect(name):
    built = build(name)
    shapes = run_shapes(name, built)
    components = []
    total = trainable = 0
    for key in ("condition", "memory", "diffusion"):
        module = built[key]
        if module is None:
            components.append({"key": key, "none": True})
            continue
        node = module_node(module, key, shapes)
        components.append({"key": key, "none": False, "tree": node})
        total += node["params"]
        trainable += node["trainable"]
    return {
        "id": name, "label": LABELS[name], "color": COLORS[name],
        "description": DESCRIPTIONS[name], "entry": built["entry"],
        "protocol": built["protocol"], "parameters": total,
        "trainable": trainable, "components": components,
        "mask": mask_data(built["diffusion"]),
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MoMaRT 网络结构浏览器</title>
<style>
:root{--bg:#f4f7fb;--panel:#fff;--ink:#142033;--muted:#64748b;--line:#d9e2ef;--accent:#2563eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif}
header{padding:32px 42px 24px;background:linear-gradient(135deg,#101d3b,#18356b);color:#fff}h1{margin:0;font-size:34px}header p{margin:9px 0 0;color:#cbd8f5}
.wrap{max-width:1600px;margin:auto;padding:25px}.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}.method{border:2px solid transparent;background:#fff;border-radius:16px;padding:16px;cursor:pointer;box-shadow:0 5px 18px #18315310;text-align:left}.method:hover,.method.active{border-color:var(--c);transform:translateY(-1px)}.method b{display:block;font-size:18px;color:var(--c)}.method small{color:var(--muted)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;margin-top:20px;padding:24px;box-shadow:0 8px 28px #19345a0d}.top{display:grid;grid-template-columns:2fr 1fr;gap:20px}.titleline{display:flex;align-items:center;gap:12px}.dot{width:12px;height:36px;border-radius:8px;background:var(--accent)}h2{margin:0;font-size:30px}.desc{color:var(--muted);font-size:17px}.meta{display:grid;grid-template-columns:1fr 1fr;gap:10px}.metric{background:#f7f9fc;border:1px solid var(--line);border-radius:12px;padding:13px}.metric span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase}.metric b{font-size:18px}.route{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:18px}.route .box{background:#eef4ff;border:1px solid #bdd0f7;border-radius:10px;padding:9px 12px;font-weight:650}.route i{color:#64748b;font-style:normal;font-size:20px}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.toolbar input{min-width:310px;padding:10px 13px;border:1px solid var(--line);border-radius:10px}.toolbar button{padding:10px 13px;border:1px solid var(--line);border-radius:10px;background:#fff;cursor:pointer}.components{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:16px}.component{border:1px solid var(--line);border-radius:14px;overflow:hidden}.component>h3{margin:0;padding:13px 16px;background:#edf3fb;text-transform:capitalize}.tree{padding:9px 12px;max-height:720px;overflow:auto}.tree details{margin-left:12px;border-left:1px solid #dfe6f0;padding-left:8px}.tree summary{cursor:pointer;padding:5px;border-radius:7px;list-style:none}.tree summary:hover{background:#f0f5fb}.tree summary::-webkit-details-marker{display:none}.caret{display:inline-block;width:16px;color:#64748b}.node-type{font-weight:700}.path{font-family:ui-monospace,monospace;color:#64748b;font-size:11px}.badge{display:inline-block;margin-left:7px;padding:2px 6px;border-radius:8px;background:#eef2ff;color:#4338ca;font-size:11px}.shape{margin:3px 0 5px 22px;color:#0f766e;font-family:ui-monospace,monospace;font-size:11px;white-space:pre-wrap;word-break:break-all}.none{padding:25px;color:#94a3b8}
.mask-wrap{display:grid;grid-template-columns:1fr 1fr;gap:20px}.canvas-box{overflow:auto;border:1px solid var(--line);border-radius:12px;padding:12px;background:#fafcff}canvas{image-rendering:pixelated;max-width:100%;height:260px}.legend{display:flex;gap:16px;color:var(--muted);font-size:13px}.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px}.code{background:#0f172a;color:#dbeafe;border-radius:12px;padding:16px;font-family:ui-monospace,monospace;white-space:pre-wrap;word-break:break-word}
.diff-title{display:flex;align-items:center;gap:10px}.hint{background:#fff7ed;color:#9a3412;border:1px solid #fdba74;padding:7px 10px;border-radius:9px;font-size:13px}.evolution{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:17px}.step{position:relative;border:1px solid var(--line);border-radius:14px;padding:14px;background:#f8fafc;min-height:145px}.step:not(:last-child):after{content:'→';position:absolute;right:-14px;top:58px;z-index:2;color:#64748b;font-size:24px;font-weight:800}.step b{display:block;font-size:17px;margin-bottom:8px}.step .change{display:block;background:#fef3c7;color:#92400e;border-radius:8px;padding:7px;margin-top:7px;font-size:13px}.step .same{display:block;color:#64748b;font-size:12px}.matrix{width:100%;border-collapse:separate;border-spacing:0;margin-top:18px;font-size:13px}.matrix th,.matrix td{border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:10px;vertical-align:top}.matrix th{background:#eaf0f8}.matrix tr:first-child th:first-child{border-radius:10px 0 0 0}.matrix .rowhead{font-weight:750;background:#f8fafc;white-space:nowrap}.matrix .changed{background:#fff7d6;color:#713f12;font-weight:650}.matrix .samecell{color:#64748b}.boundary{margin-top:14px;padding:12px 14px;border-left:5px solid #e11d48;background:#fff1f2;color:#881337;border-radius:8px}
@media(max-width:1050px){.cards{grid-template-columns:repeat(2,1fr)}.top,.mask-wrap{grid-template-columns:1fr}.components{grid-template-columns:1fr}}
</style></head><body>
<header><h1>MoMaRT 网络结构浏览器</h1><p>由实际实验类实例化 + dummy forward 自动采集；点击模块展开真实 PyTorch 层、参数量与运行时 shape。</p></header>
<main class="wrap"><div id="cards" class="cards"></div>
<section class="panel"><div class="diff-title"><h2 style="font-size:25px">先看这里：六个方法到底改了什么</h2><span class="hint">黄色 = 相对左边方法发生变化</span></div>
<p class="desc">前两个是原始短历史基线；论文四方法对照从 No-History 开始。下面按“上一列 → 下一列”只列改动。</p>
<div class="evolution">
 <div class="step"><b style="color:#0ea5e9">① 原版 DP</b><span class="same">起点：当前两帧观测 + ChiTransformer。</span><span class="change">所有动作位置都能读取两帧观测</span></div>
 <div class="step"><b style="color:#2563eb">② DP-PTP</b><span class="same">编码器、Decoder、参数量不变。</span><span class="change">只改 cross-attention mask：未来观测不可见</span></div>
 <div class="step"><b style="color:#64748b">③ No-History</b><span class="same">仍然没有长期历史。</span><span class="change">加 task embedding；换 LongMemPTP 接口，但 Tm=0</span></div>
 <div class="step"><b style="color:#16a34a">④ Direct</b><span class="same">当前观测路径和 PTP Decoder 保持。</span><span class="change">新增四路视觉空间历史：16帧 → 1024 token</span></div>
 <div class="step"><b style="color:#f59e0b">⑤ Direct-token2</b><span class="same">与 Direct 共用同一历史视觉编码器。</span><span class="change">新增无参数 AvgPool：1024 → 2 token</span></div>
 <div class="step"><b style="color:#7c3aed">⑥ VPWEM</b><span class="same">Decoder 仍接收 2 个 memory token。</span><span class="change">把固定池化换成 FIFO + 两层可学习 Q-Former</span></div>
</div>
<table class="matrix"><tr><th>对比项</th><th>DP</th><th>DP-PTP</th><th>No-History</th><th>Direct</th><th>Direct-token2</th><th>VPWEM</th></tr>
<tr><td class="rowhead">当前观测输入</td><td>raw image</td><td class="samecell">同 DP</td><td class="changed">raw image + task id</td><td class="samecell">同 No-History</td><td class="samecell">同 Direct</td><td class="changed">预计算 dp_emb + task id</td></tr>
<tr><td class="rowhead">长期历史</td><td>无</td><td class="samecell">无</td><td class="samecell">无</td><td class="changed">四路视觉 × 16帧 × 4×4</td><td class="samecell">同 Direct</td><td class="changed">历史 dp_emb + episode 位置</td></tr>
<tr><td class="rowhead">历史处理模块</td><td>无</td><td>无</td><td>无</td><td class="changed">保留全部空间 token</td><td class="changed">AdaptiveAvgPool1d（无参数）</td><td class="changed">FIFO=8 + Q-Former×2</td></tr>
<tr><td class="rowhead">送入 Decoder 的历史 token</td><td>0</td><td>0</td><td>0</td><td class="changed">1024</td><td class="changed">2</td><td>2</td></tr>
<tr><td class="rowhead">Diffusion Decoder</td><td>ChiTransformer</td><td class="changed">ChiTransformerPTP</td><td class="changed">LongMemChiTransformerPTP</td><td class="samecell">同 No-History，Tm=1024</td><td class="samecell">同 Direct，Tm=2</td><td class="samecell">同 Direct-token2，Tm=2</td></tr>
<tr><td class="rowhead">核心科学问题</td><td>普通短历史 DP</td><td class="changed">时间对齐是否重要</td><td class="changed">不用长期历史能多好</td><td class="changed">全部保留是否更好</td><td class="changed">简单压缩是否足够</td><td class="changed">学习式压缩是否更好</td></tr></table>
<div class="boundary"><b>重要边界：</b>VPWEM 这一列不只是把 AvgPool 换成 Q-Former；当前 1M 实验还同时从 raw-image、effective batch 64 变成了 dp_emb、effective batch 256，所以它不是严格单变量替换。</div>
</section>
<section class="panel"><div class="top"><div><div class="titleline"><span class="dot"></span><h2 id="title"></h2></div><p id="desc" class="desc"></p><div id="route" class="route"></div></div><div class="meta"><div class="metric"><span>总参数量</span><b id="params"></b></div><div class="metric"><span>可训练参数</span><b id="trainable"></b></div><div class="metric"><span>实验入口</span><b id="entry" style="font-size:13px"></b></div><div class="metric"><span>Attention mask</span><b id="maskdim"></b></div></div></div></section>
<section class="panel"><div class="toolbar"><h2 style="font-size:23px;margin-right:auto">真实 nn.Module 树</h2><input id="search" placeholder="搜索模块类型或路径，例如 ResNet / decoder.layers.0"><button onclick="setAll(true)">全部展开</button><button onclick="setAll(false)">全部收起</button></div><div id="components" class="components"></div></section>
<section class="panel"><h2 style="font-size:23px">实际 Cross-Attention Mask</h2><p class="desc">直接读取实例化模型的 <code>memory_mask</code>。绿色可读，灰色被屏蔽；Direct 的 1027 列按真实像素密度显示。</p><div class="mask-wrap"><div class="canvas-box"><canvas id="mask"></canvas></div><div><div id="masktext" class="code"></div><p class="legend"><span><i class="sw" style="background:#22c55e"></i>可读取</span><span><i class="sw" style="background:#cbd5e1"></i>屏蔽</span></p></div></div></section>
<section class="panel"><h2 style="font-size:23px">实验协议</h2><div id="protocol" class="code"></div></section></main>
<script>const DATA=__DATA__;
let current=DATA[0];const fmt=n=>n.toLocaleString('en-US');
function routeFor(m){let mem=m.mask.tm;return ['输入/Condition',m.id==='vpwem'?'QFormerMemory':(mem?'History tokens':'无长期记忆'),`条件 [τ, memory(${mem}), obs(2)]`,'8层 TransformerDecoder','预测噪声 εθ'];}
function makeCards(){cards.innerHTML=DATA.map(m=>`<button class="method ${m.id===current.id?'active':''}" style="--c:${m.color}" onclick="selectMethod('${m.id}')"><b>${m.label}</b><small>${fmt(m.parameters)} 参数</small></button>`).join('')}
function nodeHtml(n,depth=0){const has=n.children.length;const shape=n.shapes.map(s=>`<div class="shape">in ${esc(s.in)}<br>out ${esc(s.out)}</div>`).join('');return `<details data-search="${esc((n.path+' '+n.type).toLowerCase())}" ${depth<1?'open':''}><summary><span class="caret">${has?'▸':'·'}</span><span class="node-type">${esc(n.type)}</span> <span class="path">${esc(n.path)}</span><span class="badge">${fmt(n.params)}</span></summary>${shape}${n.children.map(c=>nodeHtml(c,depth+1)).join('')}</details>`}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function drawMask(m){const cv=document.getElementById('mask'),ctx=cv.getContext('2d'),rows=m.rows,cols=m.cols,scale=Math.max(1,Math.floor(1500/cols));cv.width=cols*scale;cv.height=rows*34;ctx.clearRect(0,0,cv.width,cv.height);for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){ctx.fillStyle=m.values[r][c]?'#22c55e':'#cbd5e1';ctx.fillRect(c*scale,r*34,Math.max(1,scale),30)}masktext.textContent=`shape = [${rows}, ${cols}]\ncolumns = [τ] + memory(${m.tm}) + obs(2)\nrows = action positions a0…a${rows-1}\n\nmemory token 对全部动作位置可见；\n近期 observation token 遵循对应 DP/PTP mask。`}
function render(){document.documentElement.style.setProperty('--accent',current.color);makeCards();title.textContent=current.label;desc.textContent=current.description;params.textContent=fmt(current.parameters);trainable.textContent=fmt(current.trainable);entry.textContent=current.entry;maskdim.textContent=`${current.mask.rows} × ${current.mask.cols}`;route.innerHTML=routeFor(current).map((x,i,a)=>`<span class="box">${esc(x)}</span>${i<a.length-1?'<i>→</i>':''}`).join('');components.innerHTML=current.components.map(c=>`<div class="component"><h3>${c.key}</h3>${c.none?'<div class="none">None</div>':`<div class="tree">${nodeHtml(c.tree)}</div>`}</div>`).join('');protocol.textContent=current.protocol+'\n'+current.description;drawMask(current.mask);search.value=''}
function selectMethod(id){current=DATA.find(x=>x.id===id);render()}
function setAll(open){document.querySelectorAll('.tree details').forEach(x=>x.open=open)}
search.addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('.tree details').forEach(d=>{const hit=!q||d.dataset.search.includes(q)||[...d.querySelectorAll('details')].some(x=>x.dataset.search.includes(q));d.style.display=hit?'':'none';if(q&&hit)d.open=true})});render();
</script></body></html>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = [collect(name) for name in METHODS]
    (OUT / "architectures.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (OUT / "index.html").write_text(
        HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
