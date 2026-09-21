#!/usr/bin/env python3
"""Build editable module-level diagrams for the MoMaRT DP and DP-PTP baselines."""

from pathlib import Path
from xml.etree import ElementTree as ET

from build_momart_detailed_diagrams import DetailPage, action_backbone, raw_current


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/assets/momart_four_methods_1m/dp_dpptp_detailed.drawio"


def make_page(parent: ET.Element, *, ptp: bool) -> None:
    name = "DP-PTP" if ptp else "DP"
    summary = (
        "MoMaRT 单任务 raw-image 基线；PTP 时间对齐条件掩码；无任务 embedding、无长期历史。"
        if ptp else
        "MoMaRT 单任务 raw-image 基线；条件观测全可见；无任务 embedding、无长期历史。"
    )
    page = DetailPage(parent, name, summary)
    recent = raw_current(page, include_task=False)
    page.node("长期历史", "不读取；Tm=0<br>无记忆压缩模块", 1710, 330, 400, 120, "muted")
    page.node("条件 token", "τ [B,1,256] + o₀,o₁ [B,2,256]<br>拼接 [B,3,256]", 2270, 330, 450, 120, "condition")
    if ptp:
        first = "动作位置 0：τ、o₀ 可见；o₁ 不可见"
        later = "动作位置 1–8：τ、o₀、o₁ 均可见"
        mask = "第 0 行 [✓,✓,×]<br>第 1–8 行 [✓,✓,✓]"
    else:
        first = "动作位置 0：τ、o₀、o₁ 均可见"
        later = "动作位置 1–8：τ、o₀、o₁ 均可见"
        mask = "全部 9 行 [✓,✓,✓]"
    page.node("Decoder 条件交叉注意力掩码", f"[9,3]；列顺序 [τ,o₀,o₁]<br>{mask}",
              1910, 650, 700, 155, "condition")
    page.label(first, 1760, 930, 1150, 46, 23, color="#475569")
    page.label(later, 1760, 995, 1150, 46, 23, color="#475569")
    page.label("动作 self-attention：9×9 下三角因果掩码。", 1760, 1100, 1150, 46, 22, color="#475569")
    action_backbone(page, recent, None, 0, ptp=ptp)


def main() -> None:
    mxfile = ET.Element("mxfile", host="app.diagrams.net", type="device", version="24.7.17")
    make_page(mxfile, ptp=False)
    make_page(mxfile, ptp=True)
    ET.indent(mxfile, space="  ")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mxfile).write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {OUTPUT} ({len(mxfile)} module-level pages)")


if __name__ == "__main__":
    main()
