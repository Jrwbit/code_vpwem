#!/usr/bin/env python3
"""Package the six reviewed figures as an offline visual comparison workbench."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/momart_main_figures_20261028"
SCHEMES = [
    ("native", "A · 可编辑学术矢量版", "SVG / PDF · 精确改字、改箭头、改布局"),
    ("drawio", "B · Next AI Draw.io", "真实 draw.io 画布导出 · 可拖动的模块与连线"),
    ("imagegen", "C · AI 科研插画版", "内置 imagegen · 光栅图，按 prompt 迭代"),
]


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build(out: Path) -> Path:
    figures = []
    for method, title in [("no_history", "No-History"), ("boxed", "Boxed No-History / 画框")]:
        for scheme, name, detail in SCHEMES:
            path = out / f"{scheme}_{method}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Required reviewed figure is missing: {path}")
            figures.append(dict(method=method, title=title, scheme=scheme, name=name,
                                detail=detail, src=data_url(path), filename=path.name))
    payload = json.dumps(figures, ensure_ascii=False).replace("</", "<\\/")
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MoMaRT 论文主图 · 三种方案对照</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#192834;font:16px system-ui,"Noto Sans CJK SC",sans-serif}
main{max-width:1900px;margin:auto;padding:35px 40px}h1{font-size:32px;margin:0 0 12px;letter-spacing:-.5px}
p{line-height:1.65;color:#55616d}header{margin-bottom:28px}.tag{font-size:13px;color:#357b8b;font-weight:700;letter-spacing:2px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px;margin:18px 0 32px}
.card{border:1px solid #d7dee3;border-radius:12px;background:white;overflow:hidden;box-shadow:0 5px 20px #243c4f05}
.card img{width:100%;height:300px;object-fit:contain;background:white;padding:8px;cursor:zoom-in}
.card figcaption{padding:16px;border-top:1px solid #edf0f2}.card h3{margin:0 0 8px;font-size:18px}.card p{font-size:13px;margin:0}
figure{margin:0}h2{font-size:22px;margin:22px 0 10px}.small{font-size:13px}.panel{background:white;border:1px solid #d7dee3;border-radius:12px;padding:24px;margin-top:25px}
.actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,a.button{font:inherit;padding:9px 16px;background:white;border:1px solid #b6c4ce;border-radius:7px;cursor:pointer;color:#234653;text-decoration:none}
button:hover,a.button:hover{background:#eaf4f5}input[type=range]{width:280px}.compare{position:relative;max-width:1600px;margin:20px auto;border:1px solid #e0e6ea;background:white;overflow:hidden}
.compare img{display:block;width:100%}.compare .over{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:.5}
.legend{display:flex;justify-content:space-between;font-size:14px;color:#426271}
dialog{width:96vw;max-width:none;height:94vh;max-height:none;border:0;border-radius:10px;padding:20px;background:#fff}dialog::backdrop{background:#142530cc}
dialog img{width:100%;height:calc(100% - 50px);object-fit:contain}dialog .actions{justify-content:space-between;height:40px}
.notice{padding:15px 18px;background:#fff8eb;border-left:3px solid #d3a44b;color:#67552d;font-size:14px;line-height:1.7}
@media(max-width:900px){main{padding:24px 16px}.grid{grid-template-columns:1fr}.card img{height:auto}}
@media print{.interactive{display:none}body{background:white}main{padding:0}.card{break-inside:avoid}}
</style>
<main><section id="contact"><header><div class="tag">MOMART · METHOD FIGURES · 2026-10-28</div>
<h1>同一方法，三种主图表达</h1><p>基于实际代码与已完成实验核对。点击图片放大；主图不混入不同训练预算的成功率。</p></header>
<h2>01 / No-History：保留最近两帧，移除长期记忆</h2><div id="plain" class="grid"></div>
<h2>02 / 画框：在 No-History 的主 RGB 上增加目标提示</h2><div id="boxed" class="grid"></div></section>
<section class="panel interactive"><h2>查看两种方法的结构变化</h2>
<p>原生 SVG 两张图共用布局。切换查看，或拖动透明度滑块，观察画框模块的变化。</p>
<div class="actions"><button id="showPlain">仅 No-History</button><button id="showBoxed">仅画框版</button>
<button id="showBoth">叠加对照</button><label>画框版透明度 <input id="mix" type="range" min="0" max="100" value="50"></label><span id="amount">50%</span></div>
<div class="compare"><img id="under" alt="No-History 原生矢量版"><img id="over" class="over" alt="画框版原生矢量图"></div>
<div class="legend"><span>No-History：原始主 RGB</span><span>画框：红色矩形提示；其他策略骨干相同</span></div></section>
<section class="panel"><h2>源文件与实际运行范围</h2><p>附件包中的 native/*.svg 可逐元素编辑；drawio/*.drawio 可在 draw.io 或 Next AI Draw.io 打开。PNG 用于预览；prompts/ 保存 AI 出图要求。</p>
<p class="small">A：代码生成 SVG，导出 PNG/PDF。B：实际调用 Next AI Draw.io MCP 并由真实编辑器导出。C：调用内置 imagegen。AutoFigure-Edit 与 PaperBanana 的独立自动生成管线未完成出图，具体预检阻塞见随附报告；本页不把其他方案的图片冒充其结果。</p></section></main>
<dialog id="zoom"><div class="actions"><strong id="zoomTitle"></strong><button id="close">关闭 / Esc</button></div><img id="zoomImage" alt="放大后的方法图"></dialog>
<script>const figures=__FIGURES__;
for(const f of figures){const card=document.createElement('figure');card.className='card';
const img=document.createElement('img');img.src=f.src;img.alt=f.title+' '+f.name;img.addEventListener('click',()=>{document.querySelector('#zoomTitle').textContent=f.title+' · '+f.name;document.querySelector('#zoomImage').src=f.src;document.querySelector('#zoom').showModal()});
const cap=document.createElement('figcaption');const h=document.createElement('h3');h.textContent=f.name;const p=document.createElement('p');p.textContent=f.detail;cap.append(h,p);card.append(img,cap);document.querySelector(f.method==='no_history'?'#plain':'#boxed').append(card)}
document.querySelector('#under').src=figures.find(f=>f.method==='no_history'&&f.scheme==='native').src;
document.querySelector('#over').src=figures.find(f=>f.method==='boxed'&&f.scheme==='native').src;
function mix(value){document.querySelector('#over').style.opacity=value/100;document.querySelector('#mix').value=value;document.querySelector('#amount').textContent=value+'%'}
document.querySelector('#mix').addEventListener('input',e=>mix(e.target.value));document.querySelector('#showPlain').onclick=()=>mix(0);document.querySelector('#showBoxed').onclick=()=>mix(100);document.querySelector('#showBoth').onclick=()=>mix(50);document.querySelector('#close').onclick=()=>document.querySelector('#zoom').close();
</script></html>'''.replace("__FIGURES__", payload)
    target = out / "preview.html"
    target.write_text(html, encoding="utf-8")
    return target


def render_and_check(path: Path, out: Path) -> None:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                                    args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1900, "height": 1500}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(path.as_uri(), wait_until="networkidle")
        page.wait_for_function("Array.from(document.images).filter(i => i.getAttribute('src')).every(i => i.complete && i.naturalWidth > 0)")
        assert page.locator(".card").count() == 6
        page.locator("#contact").screenshot(path=str(out / "comparison.png"))
        page.locator(".card img").first.click()
        assert page.locator("#zoom").evaluate("e => e.open")
        page.locator("#close").click()
        for selector, expected in [("#showPlain", "0"), ("#showBoxed", "1"), ("#showBoth", "0.5")]:
            page.locator(selector).click()
            assert page.locator("#over").evaluate("e => e.style.opacity") == expected
        assert not errors, errors
        browser.close()
    print(json.dumps({"preview": str(path), "figures": 6, "image_loads": "passed",
                      "zoom": "passed", "comparison_controls": "passed",
                      "javascript_errors": errors}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    path = build(args.out.resolve())
    if args.render:
        render_and_check(path, args.out.resolve())
    else:
        print(path)
