# MoMaRT 150k 带框 No-History：提分诊断与优先实验

日期：2026-10-27。对象由用户确认：输入图像带目标框、训练到 150k 的 No-History。本次只读核查 BHR 的实际配置、实现和已有评测；没有启动训练或修改运行任务。

## 结论

最值得验证的训练改进是统一训练和部署的框生成链路；最低成本的先行实验是用现有 150k 权重把动作执行块从 8 改成 4。随后独立消融短条件 dropout。以上都是待验证假设，尚无新增成功率结果。

不能把旧 MoMaRT 1M 普通 No-History 的 72.33% 当成本次成绩。本次在线 combined_blend 框评测为 **178/300，59.33%**；同期“画框＋坐标/ROI token”版本为 **160/300，53.33%**，描述性差值为 -6.00 个百分点，因此增加 ROI token 不是尚未尝试的新方案，也没有在这轮显示收益。

已逐一读取两种方法的 20 个原始 result.npy，复算 episode_success，核对每项 30 episodes、checkpoint=150000、eval_seed=20260917，结果与汇总 TSV 一致。均为训练 seed100，不能据单 seed 断言结构优劣。

## 真实运行设置与结果

| 项目 | 实际设置 |
|---|---|
| 模型 | No-History，长期 token=0；最近 2 帧 |
| 训练 | 150k；physical/effective batch 均为 256；seed100 |
| 动作 | horizon=9，每次执行 8 步；DDPM 50 步 |
| 视觉 | ResNet18；90×120 输入，75×100 随机裁剪 |
| 画框 | 只在主 RGB 最近两帧画 1 像素红色轮廓；缺框保持原图 |
| 训练框 | momart_bbox_gdino_red_v1；Grounding DINO tiny；RGB-only；prompt 为 a red bowl. |
| 评测框 | combined_blend 在线检测/跟踪；track combined、max_age=3、refine_alpha=0.5 |
| 短条件丢弃 | short_cond_dropout=0.3 |
| 评测 | 10 配置×30 episodes；10 并行环境；eval seed20260917 |

| 任务配置 | 带框 No-History | 带框＋坐标/ROI token | No-History 有效框比例 |
|---|---:|---:|---:|
| cleanup dishwasher / expert | 15/30 | 13/30 | 73.32% |
| cleanup dishwasher / suboptimal | 9/30 | 0/30 | 84.06% |
| cleanup sink / expert | 14/30 | 17/30 | 72.62% |
| cleanup sink / suboptimal | 15/30 | 13/30 | 80.81% |
| setup dishwasher / expert | 25/30 | 26/30 | 51.07% |
| setup dishwasher / suboptimal | 30/30 | 22/30 | 57.17% |
| setup dresser / expert | 23/30 | 22/30 | 44.80% |
| setup dresser / suboptimal | 12/30 | 13/30 | 35.50% |
| unload / expert | 16/30 | 19/30 | 59.64% |
| unload / suboptimal | 19/30 | 15/30 | 63.81% |
| 合计 | **178/300，59.33%** | **160/300，53.33%** | — |

有效框比例仅为检测器统计的 valid_frames/processed_frames，不是准确率或 IoU。cleanup dishwasher / suboptimal 有框率 84.06%，成功率只有 30%；setup dishwasher / suboptimal 有框率 57.17%，却成功 30/30。不能据覆盖率直接认定漏检是失败主因，更不能简单降低阈值来追求更多框。应检查任务关键阶段的框是否框对目标、是否滞后，以及抓取/放置动作是否失败。

## 优先实验

### 1. 先做低成本评测：执行块 8→4

保持 150k checkpoint、horizon=9、obs_steps=2、检测器、最大环境步数和初始条件不变，只在每次预测后执行前 4 个有效动作，再观测和重新规划。若有收益，再试 2 步。推理调用次数对相同环境步数约为原来的 2 倍/4 倍，必须同时记录延迟。

实际 evaluate.py 的环境步长、动作切片、elapsed_steps 和 inference_action_steps 都读取 action_steps，模型 horizon 不必改变。应在评测配置副本中修改，不改原训练配置，也不通过训练入口重新断言 horizon=obs_steps+action_steps-1。

这是闭环纠错假设，不能保证更短一定更好。在线 tracker 的更新频率也随重观测频率改变，要记录该协议变化；同 reset seed 不代表不同采样次数下的扩散随机轨迹完全相同。

### 2. 最值得做的训练实验：让训练框接近最终在线框

已确认训练固定读取 Grounding DINO 离线 sidecar，当前评测使用 combined_blend 跟踪器。两者的框位置、缺框频率、错误框和时序滞后可能不同；当前证据能证明输入链路不同，不能证明它已造成多少百分点损失。

先在同一组训练轨迹图像上比较两套框的坐标、缺框与错框差异，再用冻结的最终在线检测/跟踪器重生成训练框。回放应遵守部署的取帧顺序、间隔和 episode reset，不能把随机训练样本接进持续 tracker。以相同 150k 起点、相同后续更新预算和学习率计划，对照“旧框续训”与“在线同源框续训”。

随后再独立增加框鲁棒训练：偶尔不画框但保留原 RGB、按实测误差添加小幅坐标抖动。框 dropout 与短条件 dropout 不同：前者只去掉提示，后者会清掉模型的整条短观测条件。不要一次同时更换框、加抖动、改条件 dropout，否则无法判断收益来源。

对覆盖低的 setup dresser / suboptimal 和有框仍低分的 cleanup dishwasher / suboptimal 分别查看失败阶段；最终仍报告全部 10 配置宏平均，避免只优化选中的弱项。

### 3. 独立降低短条件 dropout

这次 150k 配置确实为 short_cond_dropout=0.3，并沿用 LongMemDDPM。当前观测编码后按样本丢弃整条短条件；No-History 又没有长期记忆分支。这提供了尝试 0.1 和 0 的具体依据。

建议从同一 checkpoint 做同预算分支，保持数据、检测框、执行块、学习率计划一致。降低该值可能改善观测利用，也可能削弱正则化；应以在线成功率判定，而非仅看训练 diffusion loss。不要与普通 attention dropout=0.3 混为一谈。

## 已有续训及实验顺序

2026-10-27 核查时，已有 150k→250k 续训目录，最新读到的 metrics 为 **197500 step**；目录已有 175k checkpoint。其 manifest 记录保留 Adam/EMA，把 cosine T_max 延长到 250000；训练仍使用原离线框，评测计划使用冻结 combined_blend。不是本次新启动的任务。

不能仅凭 150k 就认定训练不足，也不能从几个不同任务 batch 的 loss 值推断成功率还在上涨。先取得已有续训的固定节点在线评测，作为“只增加训练”的控制；接着做 150k 执行块 4 的评测，再安排同源框和条件 dropout 单变量训练。比较不同框版本时必须使用相同训练步数和更新预算，不能用新框 250k 对旧框 150k 直接归因。

初筛可沿用原 30 episodes/配置；确认方案后用独立固定测试初始条件、更多 episodes 和多个训练 seed 复核。每项保留失败阶段、逐 episode 结果、执行块与检测器配置。不要把在同一批测试上反复选出的最佳值当独立最终成绩。

## 证据入口

服务器：BHR，SSH alias bitbot421。

- 150k 训练：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_150k/seed100/`，读取 train_config.yaml、bbox_input.json 和 metrics.jsonl。
- 正式在线评测：`/home/jrw/vpwem_outputs/momart_boxed_online_150k_combined_blend_20261026_gpu1/`，读取 manifest.json、两方法 gpu1_success_rates.tsv、各 step_150000/<task>/result.npy。
- 续训：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100/`，读取 continuation_manifest.json、metrics.jsonl 和 checkpoint 文件名。
- 实现：`/home/jrw/code_vpwem_boxed_eval_20261026/momart_boxed_no_history/` 的 dataset.py、painting.py、evaluate.py；`/home/jrw/code_vpwem_boxed_continue_20261027/momart_no_history_baseline/model.py` 与 momart_boxed_tokens 模块。
- No-History 150k checkpoint SHA256：`b33b859502decba0c734734bedf91df2b3ba28a35c17b7eeff7eae25989a59ab`（评测与续训 manifest 一致）。
- 在线 detector bundle SHA256：`af62860467a1c95d2e60a9df5d89ee1b9001116cb45c1aa7a285c1810f6f7c07`。

方法背景：[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 使用动作序列的滚动执行与重规划；这支持执行块消融的设计，不证明本任务 4 步优于 8 步。[VIOLA](https://proceedings.mlr.press/v205/zhu23a.html) 提供对象表征用于策略的先例，但本项目已经有坐标/ROI token 实验，不能用文献替代本次负面结果。

本次验证为已有结果复算和代码/配置审计，未运行新训练或在线 rollout。没有需要上传的新增图像或模型附件。
