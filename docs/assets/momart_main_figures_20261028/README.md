# MoMaRT 论文主图工作台

2026-10-28：普通 No-History 和 Boxed No-History 各制作三种表达。完整报告为 `reports/momart_main_figures_20261028.md`，证据为本目录 `evidence.md`。

飞书图集与附件：https://feishu.cn/docx/Smc1d6JbTozl1NxAYT4cda6Enqg

## 交付文件

- `native/no_history.svg`、`native/boxed.svg`：2000×1200 可编辑矢量主图，相同主要布局和模块 ID。
- `drawio/no_history.drawio`、`drawio/boxed.drawio`：真实 Next AI Draw.io MCP 编辑器可加载的对象源文件。
- `prompts/`：内置 imagegen 的初稿要求和两轮结构纠错要求。
- `tool_runs.json`：开源工具的真实运行/预检记录，区分官方、作者维护版和第三方实现。
- 仓库根目录 `artifacts/momart_main_figures_20261028/`：最终 PNG/PDF、离线 `preview.html`、六图总览、真实输入参考、初稿留档及附件 ZIP。该目录已被现有 Git 规则忽略。

源文件中的英文用于论文排版，报告提供中文解释。图中的机器人/碗均为示意；`boxed_input_reference.png` 才是真实 MoMaRT 输入预览，来自 2026-10-09，不是 250k rollout 截图。

## 复现

在仓库根目录运行。A 方案需要 Python 和 CairoSVG（本次隔离环境 `/tmp/nativefigvenv`）；B 方案需要 Node.js、Python Playwright 和 Chrome。

```bash
python tools/figure_workbench_20261028/render_native.py

python tools/figure_workbench_20261028/next_drawio_bridge.py \
  docs/assets/momart_main_figures_20261028/drawio/no_history.drawio \
  --output-prefix artifacts/momart_main_figures_20261028/drawio_no_history

python tools/figure_workbench_20261028/next_drawio_bridge.py \
  docs/assets/momart_main_figures_20261028/drawio/boxed.drawio \
  --output-prefix artifacts/momart_main_figures_20261028/drawio_boxed

python tools/figure_workbench_20261028/build_gallery.py --render
python tools/figure_workbench_20261028/package.py
```

`next_drawio_bridge.py` 固定使用官方 MCP 包 0.4.6；必要时首次运行由 npx 下载。它只加载/导出已审阅的源图，不调用绘图模型 API。C 方案使用内置 imagegen，按保存的 prompt 与上版图继续修改；随机生成不保证逐像素重现。本次未使用 imagegen API/CLI 回退。

## 查看改动

打开 `preview.html`，点击任一图放大；下方提供两方法的原位切换与透明度叠加。A 版的共同模块保持布局和稳定 ID；修改时可按 `recent-observations`、`multimodal-encoder`、`working-memory`、`condition-sequence`、`action-denoiser`、`method-change-panel` 定位。B 版可以直接在 draw.io 拖动和编辑。

方法边界：两帧短观测保留，长期 token 为0；画框只改变主 RGB；条件为 `[τ, o(t−1), o(t)]`；任务嵌入加到观测特征；预测9个动作位置，默认执行8步。原生 SVG 与 draw.io 源文件用于核对这些结构；AI 插画的符号画法只作高层示意。

C2 插画的框坐标标签是通用示意，真实实现使用归一化 `(x1,y1,x2,y2)`；其条件箭头进入整个采样组，实际应直接进入 Transformer cross-attention，与带噪动作独立。A/B 为精确连接基准。`iterations/` 初稿仅供查看迭代过程，不能当最终图使用。

## 发布与验证

飞书正文包含六张完整主图、总览、真实输入参考和最新结果表；实际附件包含图源、最终图片、离线预览、prompt、证据与导出脚本。发布后回读正文、所有图片块和附件并校验内容；验证记录保存在 `artifacts/`，不记录凭据。
