# MoMaRT 画框与 No-History 主图：方法及结果证据

核查日期：2026-10-28；本记录整理时间：11:36 CST（Asia/Shanghai）。证据以本次只读访问 BHR（SSH alias `bitbot421`）获得的代码、配置和原始结果为准；本次没有启动训练或评测。

## 画框确切含义

`boxed_no_history` 在主相机最近两帧 RGB 上叠加 1 像素红色矩形轮廓（RGB `[255,0,0]`），不裁出目标、不覆盖背景、不将坐标作为独立策略输入。没有有效框时保持原始图像。腕部 RGB、主视角深度、腕部深度、proprio 和 scan 保持原输入链路。

这仍是 No-History：模型为 `nn_memory=None`、长期记忆 token 数 `Tm=0`；使用最近两帧而非没有任何时序信息。不要在主图中添加长期缓存、Q-Former 或情景记忆模块。

训练框来自离线 `momart_bbox_gdino_red_v1` sidecar，检测器为 Grounding DINO tiny，RGB-only，文本提示 `a red bowl.`。在线评测使用冻结的 RGB-only `combined_blend` 检测/跟踪，`max_age=3`、`refine_alpha=0.5`。两者共享画框规则，但框生成链路不同；不能画成同一检测器端到端训练。检测 bundle 的 `use_roi:true` 属于检测器内部设置，不代表策略有 ROI token 分支。

## 主图真实模块与数据流

1. 最近两帧主 RGB（90×120）经过目标检测/跟踪后叠加红框；训练时则读取离线框。
2. 主 RGB、腕部 RGB、主深度、腕部深度分别经过 ResNet18；scan（1×100）经过 ScanCore，proprio 为10维。训练裁剪配置为75×100。
3. 各模态特征拼接，经 MLP 映射为每帧256维，再加同一任务嵌入（5种语义任务），产生2个短观测 token。
4. 条件序列为 `[diffusion timestep, observation(t−1), observation(t)]`，长期 token 数为0。
5. `LongMemChiTransformerPTP` 使用8层、4个注意力头、256维模型；带噪动作经过因果 self-attention，并通过有时间对齐约束的 cross-attention 读取条件。
6. DDPM 50步采样生成9×10动作序列；标准执行从 `obs_steps−1=1` 开始的8步，即索引 `1:9`；随后获取新观测并重新规划。
7. action4 消融保持 horizon=9、obs_steps=2、150k权重，只执行索引 `1:5` 的4步，训练配置的 action_steps 仍为8。

## 本次核实的结果

所有表列结果为10个任务配置、每配置30 episodes，共300 episodes。250k和action4均已逐项读取原始 `result.npy`，复算 `episode_success`，核对 checkpoint、eval_seed=20260917及每项30 episodes。action4额外逐项核对 training_action_steps=8、execution_action_steps=4、horizon=9、obs_steps=2。

| 版本 | 成功/总数 | 成功率 | 完成时间或证据日期 |
|---|---:|---:|---|
| **Boxed No-History 250k，执行8步** | **194/300** | **64.67%** | **2026-10-28 05:21:51 CST，10/10完成，failures=0，missing=0** |
| Boxed No-History 150k，执行4步 | 183/300 | 61.00% | 2026-10-27 21:37:10 CST，complete=true，三条队列退出码均0 |
| Boxed No-History 150k，执行8步 | 178/300 | 59.33% | 2026-10-26在线评测；2026-10-27审计复算 |
| 画框＋坐标/ROI token，150k | 160/300 | 53.33% | 同期在线评测；2026-10-27审计复算 |

### 每任务计数

| 任务配置 | 250k / action8 | 150k / action4 | 150k / action8 |
|---|---:|---:|---:|
| cleanup dishwasher / expert | 14/30 | 13/30 | 15/30 |
| cleanup dishwasher / suboptimal | 11/30 | 6/30 | 9/30 |
| cleanup sink / expert | 18/30 | 14/30 | 14/30 |
| cleanup sink / suboptimal | 19/30 | 9/30 | 15/30 |
| setup dishwasher / expert | 25/30 | 24/30 | 25/30 |
| setup dishwasher / suboptimal | 23/30 | 27/30 | 30/30 |
| setup dresser / expert | 24/30 | 28/30 | 23/30 |
| setup dresser / suboptimal | 19/30 | 19/30 | 12/30 |
| unload dishwasher to dresser / expert | 19/30 | 19/30 | 16/30 |
| unload dishwasher to dresser / suboptimal | 22/30 | 24/30 | 19/30 |
| 合计 | **194/300** | **183/300** | **178/300** |

250k续训保留原离线训练框；manifest记录保留 Adam/EMA，把 cosine T_max 延长到250000。metrics最后记录step250000、learning_rate=0，并存在model_250000.pt。相对150k/action8的+5.33个百分点是本次续训的描述性差值，不能据此归因于架构改变。

150k/action4相对150k/action8为+1.67个百分点，单seed，不能视为稳定或显著提升；执行块改变同时影响闭环观测与在线tracker更新频率。同seed也不代表不同采样调用次数下的扩散噪声逐episode严格配对。原wrapper只向evaluator回传每动作块最后两个info，本次保持旧计分实现，更密集的检查也可能影响捕获短暂成功。action8结果里的episode_steps写为1504，是外层计数；wrapper实际在1500截断，不代表额外执行4步。

## 与普通 No-History / VPWEM 的关系和比较限制

普通 No-History 使用 VPWEM 的 PTP 扩散动作骨干，但移除长期记忆输入；图可参考 VPWEM 总体布局，不能因此保留其长期记忆模块。画框版在 No-History 输入侧增加红框视觉提示。

已有普通 MoMaRT No-History 1M结果为217/300、72.33%，记录于 `reports/momart_1m_four_methods/report_feishu.md`。它是不同训练预算的旧实验，**不能把72.33%当画框150k或250k结果，也不能与64.67%按同预算公平对比以归因于画框效果**。历史1M VPWEM与raw-image No-History还存在观测编码器和有效batch差异，不能直接把分数差解释为长期记忆机制的因果效应。

本图中的框是视觉提示，不是检测准确率证明；`bbox_valid_rate`仅为valid_frames/processed_frames，不是IoU或检测准确率。主方法不存在策略ROI token分支；“画框＋坐标/ROI token”是另一个已测版本。

## 真实图片素材

- 原始服务器文件：`bitbot421:/home/jrw/vpwem_outputs/momart_boxed_no_history_preview_20261009/boxed_input_preview.png`。
- 本机未修改副本：`artifacts/momart_main_figures_20261028/boxed_input_reference.png`。
- 图片已视觉检查：左侧实际MoMaRT主RGB，右侧同画面加红色目标矩形，目标红碗被机械臂部分遮挡。该图展示真实画框输入规则，来自2026-10-09预览，不是250k在线评测截图。

## 原始证据路径

服务器为BHR，SSH alias `bitbot421`。

- 250k原始结果：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100/eval_online_combined_blend_250k/step_250000/<task>/result.npy`。
- 250k汇总、完成状态：上述评测目录中的 `gpu7_success_rates.tsv`、`gpu7_status.tsv`，最后状态为2026-10-28T05:21:51+08:00 FINISH all all failures=0 missing=0。
- 250k续训协议、训练完成证据：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100/continuation_manifest.json`、`metrics.jsonl`、`models/model_250000.pt`。
- action4原始结果：`/home/jrw/vpwem_outputs/momart_boxed_no_history_150k_action4_20261027_gpu356/step_150000/<task>/result.npy`；同根目录的 `summary.json`、`completion.json`。
- 150k标准评测：`/home/jrw/vpwem_outputs/momart_boxed_online_150k_combined_blend_20261026_gpu1/`，两方法子目录的 `gpu1_success_rates.tsv` 及 `step_150000/<task>/result.npy`。
- 150k真实配置及输入规格：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_150k/seed100/train_config.yaml`、`bbox_input.json`。
- 模型：`/home/jrw/code_vpwem_boxed_continue_20261027/momart_no_history_baseline/model.py`。
- 画框与数据/评测：`/home/jrw/code_vpwem_boxed_eval_20261026/momart_boxed_no_history/painting.py`、`dataset.py`、`evaluate.py`。
- 检测bundle：`/home/jrw/vpwem_outputs/bbox_strategies_20261020/comparison_v2/detector_bundle.json`；SHA256 `af62860467a1c95d2e60a9df5d89ee1b9001116cb45c1aa7a285c1810f6f7c07`。
- 150k checkpoint SHA256：`b33b859502decba0c734734bedf91df2b3ba28a35c17b7eeff7eae25989a59ab`。
- 本地前置审计：`docs/momart_boxed_nohistory_150k_improvement_20261027.md`、`reports/momart_boxed_nohistory_action4_20261027.md`。后者记录启动时尚未完成，本记录补充后续读取到的完整结果，保留原历史状态。
