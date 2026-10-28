# MoMaRT 画框与 No-History：论文主图三方案对照

日期：2026-10-28（Asia/Shanghai）。两种方法各制作三版，共六张主图：A 为可编辑学术矢量图，B 为实际 Next AI Draw.io 编辑器导出图，C 为内置 imagegen 科研插画图。各版均依据本机代码、VPWEM 原论文结构和 BHR 上的最新结果核对。没有运行新的训练或策略评测。

建议优先用 **A 版作为论文方法图的结构底稿**，用 **B 版做日常拖动和模块修改**，用 **C 版比较插画风格与视觉表现**。A/B 保留对象级源文件；C 为光栅图，不具备逐元素编辑能力。论文正式图仍应以可审查的结构和准确连线为准。

## 先看六张图的整体效果

![No-History 与画框的三种主图方案总览](../artifacts/momart_main_figures_20261028/comparison.png)

图内使用英文以便后续论文排版；本页用中文解释。机器人和碗的绘制场景均为示意图，不是评测截图。真实 MoMaRT 输入示例单独放在后文。

## A：可编辑学术矢量版

两图共用画布和主要模块布局，便于直观看出新增的红框提示。SVG 含稳定模块 ID，可修改文字、箭头、颜色、位置；同时提供高清 PNG 和 PDF。

### A1 · No-History

![A1 No-History 可编辑矢量主图](../artifacts/momart_main_figures_20261028/native_no_history.png)

保留 VPWEM 的最近两步观测与 PTP 动作扩散骨干，删除长期 observation cache、summary cache、Q-Former 与 episodic memory tokens。No-History 表示没有长期记忆，仍然读取最近两帧。

### A2 · Boxed No-History / 画框

![A2 画框 No-History 可编辑矢量主图](../artifacts/momart_main_figures_20261028/native_boxed.png)

在主 RGB 最近两帧上叠加 1 像素红色目标矩形；没有有效框则保留原图。腕部 RGB、两路深度、scan 和 proprio 保持原样。框坐标用于画图，不作为额外策略 token；本图不包含另一个“坐标/ROI token”实验版本。

## B：Next AI Draw.io 实际编辑器版

使用官方 `@next-ai-drawio/mcp-server@0.4.6`、自带离线 draw.io 画布与 Chrome 实际加载和导出。主代理根据证据构造模块与连线，经 MCP 送入编辑器；本次没有调用该 Web 应用另外配置的模型 API。源文件为真正的 `.drawio`，各模块、标签和连线均可在编辑器里继续修改。

### B1 · No-History

![B1 Next AI Draw.io No-History 主图](../artifacts/momart_main_figures_20261028/drawio_no_history.png)

### B2 · Boxed No-History / 画框

![B2 Next AI Draw.io 画框 No-History 主图](../artifacts/momart_main_figures_20261028/drawio_boxed.png)

使用方式：在 [Next AI Draw.io](https://next-ai-drawio.jiang.jp/) 或 draw.io 打开附件包中的 `drawio/no_history.drawio`、`drawio/boxed.drawio`。可直接拖动模块、双击改字，或接入模型后选中局部要求修改。项目链接：[DayuanJiang/next-ai-draw-io](https://github.com/DayuanJiang/next-ai-draw-io)。

## C：AI 科研插画版

使用内置 imagegen，根据已核实的统一方法说明生成，再进行结构纠错迭代。完整初稿与修改 prompt 保存在附件 `prompts/`。这些图片适合比较视觉风格；图中机器人、碗和传感器画面是明确标注的 schematic。

### C1 · No-History

![C1 AI 科研插画 No-History 主图](../artifacts/momart_main_figures_20261028/imagegen_no_history.png)

### C2 · Boxed No-History / 画框

![C2 AI 科研插画 画框 No-History 主图](../artifacts/momart_main_figures_20261028/imagegen_boxed.png)

C 为高层视觉风格示意，精确连接与张量结构以 A/B 为准。C2 的框坐标标签是通用示意，实际实现使用归一化 (x1, y1, x2, y2)；图中条件箭头经过采样组的噪声区域，实际 3 个条件 token 直接供 Transformer cross-attention，9×10 噪声动作走独立动作输入，两者不拼接。图中场景均不是实际 rollout 截图。

初稿审查重点包括：两帧共享同一组模态编码器；任务嵌入加到两个观测特征；条件顺序为时间 token 加两个观测 token；DDPM 50 次迭代包含重复调用 Transformer；预测 9 个动作位置并执行其中 8 步；无框分支保留原 RGB。修改过程与仍需注意的示意边界记录在附件说明中。

## 怎样直观看到后续修改

附件中的 `preview.html` 是可直接双击打开的离线对比页，不依赖外网。它包含六张图的并排展示、点击放大，以及 A 版 No-History / 画框版的原位切换和透明度叠加。拖动滑块可以观察同一布局中输入提示和说明面板的变化。

建议保留原 `.svg` / `.drawio` 再另存新版本，按模块名指定修改。A 版 SVG 的稳定 ID 与 B 版的对象 ID 都保留在源文件中；PNG 用于预览。图像版本比较负责检查视觉变化，方法证据记录用于检查结构是否改错。

## 真实画框输入参考

![真实 MoMaRT 原始主 RGB 与带框输入对照](../artifacts/momart_main_figures_20261028/boxed_input_reference.png)

左侧为真实原始主 RGB，右侧为相同画面加红框的策略输入。文件来自 BHR 的 2026-10-09 输入预览，保持原样复制，**不是最新 250k 在线 rollout 截图**。该图只用于说明实际画框规则。

## 最新结果与口径

本次读取了 BHR 上完成状态及逐任务 `result.npy`；250k 与 action4 的结果均按 `episode_success` 复算。各项为 10 个任务配置 × 30 episodes，共 300 episodes，评测 seed 为 20260917。

| 方法 / 协议 | 成功数 | 成功率 | 状态 |
|---|---:|---:|---|
| **画框 No-History，250k，执行8步** | **194/300** | **64.67%** | **2026-10-28 05:21 CST 全部完成，最新带框主结果** |
| 画框 No-History，150k，执行4步 | 183/300 | 61.00% | 2026-10-27 21:37 CST 全部完成 |
| 画框 No-History，150k，执行8步 | 178/300 | 59.33% | 原参考结果 |
| 普通 No-History，1M，执行8步 | 217/300 | 72.33% | 本机找到的最新完整普通无框报告，2026-09-20 |
| 画框＋坐标/ROI token，150k | 160/300 | 53.33% | 独立方法，本次两张主图不包含它 |

250k 与 150k 的差值是续训后的描述性结果，不能归因于结构改动；150k/action4 的 +1.67 个百分点还伴随闭环观测、在线跟踪和成功标志检查频率变化。普通 No-History 的 1M、effective batch64 与画框的 250k、batch256 不同，不能用 72.33% 对 64.67% 直接判断画框是否有效。主图因此不放跨预算成绩柱状图。

旧的 2026-10-27 action4 启动记录写的是“仍在运行”；本页补充其后已完成的结果，并保留旧记录的历史状态。

## 所有尝试的实际运行范围

| 路线 | 本次实际完成情况 | 可继续编辑的形式 |
|---|---|---|
| 原生矢量绘图 | 两张图，SVG→PNG/PDF，逐张目检 | SVG 元素与生成脚本 |
| Next AI Draw.io | 官方 MCP、实际编辑器加载、PNG/SVG 导出，两张图 | `.drawio` 模块与连线 |
| 内置 imagegen | 两张图，初稿及纠错迭代 | 保存的图片与 prompt；无对象级结构 |
| AutoFigure-Edit | 尝试启动预检；缺少 Python 依赖和可用模型凭据，未完成独立出图 | 未把其他路线产物标成此工具结果 |
| PaperBanana | 官方/作者仓库及第三方复现分别记录预检；依赖或模型凭据未满足，未完成独立出图 | 未把其他路线产物标成此工具结果 |

相关项目：[AutoFigure-Edit](https://github.com/ResearAI/AutoFigure-Edit)、[PaperBanana 作者仓库](https://github.com/dwzhu-pku/PaperBanana)。具体仓库、版本、命令和阻塞记录在附件 `tool_runs.json`。没有为预检调用额外收费的模型 API。

## 附件与复现

文末的实际飞书附件包包含六张最终 PNG、A 版 PDF/SVG、B 版 `.drawio` 和 SVG、离线预览页、出图与修改 prompt、生成/导出脚本、证据记录与运行清单。无需依赖本地文件链接下载。

源码入口：`docs/assets/momart_main_figures_20261028/` 与 `tools/figure_workbench_20261028/`；渲染图片与打包产物位于被 Git 忽略的 `artifacts/momart_main_figures_20261028/`。没有把模型、数据集、API 密钥或运行缓存放入 Git。

验证覆盖：方法与张量规格审阅、逐张渲染检查、SVG/XML 对象 ID 检查、真实 draw.io 导出、六图加载、点击放大和叠加切换。飞书发布后另外回读正文、图片块及附件，并核对上传图片内容。
