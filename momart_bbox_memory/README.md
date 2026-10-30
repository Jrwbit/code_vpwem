# MoMaRT bbox interaction memory: BS256 / 150k

本目录为 BHR 独立实验，基于 `6e80f34` 建立；不覆盖已归档的 No-History。
用户要求本轮必须包含 bbox。`experiment.yaml` 是设计合同，不是已完成的训练入口。

固定训练设置：5 个任务 ID、10 个 expert/suboptimal 数据集混合训练；effective batch 256；150000 次完成的 optimizer update；每 25000 次保存，保留 25k/50k/75k/100k/125k/150k 六组模型和恢复状态。
原训练入口从 step=0 开始更新且使用 <= 上界，新入口必须修正为恰好 150000 次更新，不能照搬 off-by-one。

## Bbox 前置检查

所有十个 MoMaRT Zarr 均无 bbox 字段。HDF5 元数据中的任务物体是 YCB `024_bowl`。
首选固定 Grounding DINO tiny 检测器，仅使用主视角 RGB 和固定提示 `a red bowl.`。
模型版本、阈值和坐标约定在 detector.py 固定；同一实现用于离线标注和在线评测。
不把 `obs/object` 三维真值、未来帧或任务成功标志提供给检测器。
缺失检测输出 valid=false 和零占位框，不用全图伪装目标框；多个去重候选标记 ambiguous。
已检查 10 数据集、20 个训练 episode 共 120 张固定抽样图。通用 bowl 提示会误框桌面和垃圾桶；最终使用 red bowl 提示，加红色像素支持过滤（R>=35、R>1.5G/B、|G-B|<=0.2*(R-min(G,B))、框内至少 3 个且占比至少 10%）。
这是显式的红碗颜色先验，不是通用物体识别能力；训练和在线推理必须一致。最终 77/120 帧保留框、1 帧多候选；这不是精度或召回率，未做人工真值 IoU 评测，边缘少量可见目标仍有漏检。
抽样检查页在 /home/jrw/vpwem_outputs/momart_bbox_preflight_20261008_hue；原图有框时绘制，无框则 valid=false。阈值选择使用的这些图不是独立测试集。

```bash
cd /home/jrw/code_vpwem_momart_bbox_memory
/home/jrw/anaconda3/envs/vpwem/bin/python -m unittest momart_bbox_memory.test_detector -v
CUDA_VISIBLE_DEVICES=0 /home/jrw/anaconda3/envs/vpwem/bin/python -m momart_bbox_memory.audit_bbox \
  --output /home/jrw/vpwem_outputs/momart_bbox_preflight_20261008
```

## Memory 设计

主视角场景特征、碗的局部特征和框位置、机器人状态、时间及检测质量形成交互状态。
首轮不把过去动作直接作为条件，避免额外动作复制/对齐变量。
时间库最多 32 条，首次有效锚点 1 + 较早 reservoir 15 + 近期 16，互不重复。
只用最早近期观测之前的记录，读取时生成 age；当前两帧从原 baseline 编码。
2 层 query compressor，8 个 learnable queries，256 维，4 heads，FFN 1024。
DP 条件为 `[diffusion-step, memory x8, observation x2]`；动作序列在 cross-attention 中读取条件。
缺失对象时保留场景和机器人状态，并单独屏蔽对象内容，不凭空生成可见目标。
记忆不在 diffusion 内循环更新。历史 mask、episode reset 和首帧空历史需要测试。

权重、检测缓存、图片、训练输出均保留在仓库外。bbox 数据质量确认后才能完成训练入口与速度测量。

## 交互目标标注合同

`interaction.py` 提供不改变策略输入维度的轻量标注工具：关键帧记录当前交互目标和放置区域，目标 mask 转为可见区域的紧致 normalized `xyxy` 框，跨帧传播只在 `target_id` 一致且没有遮挡时插值。遮挡保留目标身份，但框置空，不用上一帧框伪造可见目标。`transform_box_to_crop` 对 bbox 应用与随机 crop 相同的半开像素坐标变换，保证位置 token 与 crop 后 RGB 对齐。

`annotation_hit_rate` 只报告可见参考帧上的 IoU 命中率，同时分开漏框、误报和双方均缺失帧；它不是策略成功率。正式缓存应保留关键帧来源、目标身份、遮挡状态和标注标准版本，不能把传播结果当成人工真值。

## 全量生成

使用 `python -m momart_bbox_memory.launch_bbox --gpus 4 5 6` 按帧数分配 10 数据集。缓存默认在 `/home/jrw/vpwem_data/momart_bbox_gdino_red_v1`，可断点续跑；manifest 和日志在该目录内，未加入 Git。全部 `complete=true` 后才能用于全量训练。权重固定版本，SHA256 为 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`。

## 在线检测器独立评测

`evaluate_online.py` 不加载策略或 VPWEM checkpoint。它从每个 HDF5 的 `mask/valid` 留出集合固定抽取 3 个 episode，与 `mask/train` 检查互斥；每个 episode 在 5 个时间位置初始化仿真状态，再执行录制动作。每段采样 8 帧、间隔 3 个动作步，每个数据集 120 帧，十个数据集共 1,200 帧。片段表示时间位置，不能直接当成已验证的抓取或放置阶段。

主相机 RGB 和语义分割在同一次渲染中产生，统一到策略主视角的 90×120。每段首帧还与环境 `get_observation()['rgb']` 的缩放结果检查对齐，平均像素误差超过 1 时终止该任务。检测器只接收 RGB；class 50 可见 mask 只作为评测参考，至少 1 像素即计为可见，另提供至少 3 像素的敏感性统计。该实验评估实时仿真片段，不代表原始录制 RGB 的标注精度或闭环策略任务成功率，也没有评测目标放置区域。

`QualityGatedDetector` 与离线缓存共用 `quality.py`：拒绝多候选、score < 0.35、非法几何框。训练数据加载器将质量门版本和阈值写入 detector metadata；在线策略入口按该 metadata 构造相同的过滤器，版本不匹配会报错。旧缓存和旧 checkpoint 仍按原 detector metadata 解释。无有效框时 `OnlineBoxedRGB` 保留原图，不传播陈旧坐标，不改变输入维度。

在 BHR 的代码目录运行：

```bash
bash tools/run_momart_bbox_online_bhr.sh
# 也可传入互不重叠的数据集名称分组运行；BBOX_GPU 指定检测 GPU。
BBOX_GPU=3 bash tools/run_momart_bbox_online_bhr.sh table_cleanup_to_sink_expert
/home/jrw/anaconda3/envs/vpwem/bin/python -m momart_bbox_memory.summarize_online /home/jrw/vpwem_outputs/bbox_online_20261020
```

专用 BHR 快照中的启动脚本名为 `run_online.sh`。可用 `BBOX_CODE_ROOT`、`BBOX_EVAL_ROOT`、`BBOX_PYTHON` 覆盖路径。运行会覆盖指定数据集的当前结果，应为不同参数使用不同输出目录。

每任务保存 `frames.jsonl`、`result.json` 和用于复核的 RGB / mask PNG；汇总脚本仅在十任务结果齐全时报告 `complete=true`。这些运行产物留在仓库外。`hit_rate` 的分母是全部可见目标帧，包括漏检；`localization_precision` 的分母是所有输出框帧，IoU≥0.5 才算正确；`absent_false_positive_rate` 单独统计目标不可见时仍出框的比例。连续漏检按采样帧计数，并在片段边界重置，不能当成连续物理动作步数。

### 2026 年 10 月 20 日 BHR 测试结果

十个数据集均完成 120 帧，总计 30 个留出 episode、150 个初始化片段、1,200 帧；824 帧目标可见、376 帧目标不可见。所有片段首帧与在线传感器的 RGB 对齐 MAE 为 0。分割参考来自当前实时图像的同次渲染，因此不依赖原始录制 RGB 的 state replay 对齐。

| 数据集 | 可见帧 | 筛选前命中率 | 质量门命中率 | 质量门输出框准确比例 |
|---|---:|---:|---:|---:|
| `table_cleanup_to_dishwasher_expert` | 90 | 91.11% | 73.33% | 98.51% |
| `table_cleanup_to_dishwasher_suboptimal` | 96 | 90.62% | 83.33% | 100.00% |
| `table_cleanup_to_sink_expert` | 96 | 96.88% | 79.17% | 100.00% |
| `table_cleanup_to_sink_suboptimal` | 96 | 95.83% | 79.17% | 100.00% |
| `table_setup_from_dishwasher_expert` | 80 | 90.00% | 88.75% | 100.00% |
| `table_setup_from_dishwasher_suboptimal` | 72 | 84.72% | 77.78% | 100.00% |
| `table_setup_from_dresser_expert` | 57 | 98.25% | 96.49% | 100.00% |
| `table_setup_from_dresser_suboptimal` | 69 | 89.86% | 75.36% | 100.00% |
| `unload_dishwasher_to_dresser_expert` | 80 | 90.00% | 88.75% | 100.00% |
| `unload_dishwasher_to_dresser_suboptimal` | 88 | 100.00% | 92.05% | 100.00% |

按帧汇总，筛选前命中 765/824（92.84%）、输出框准确比例 765/768（99.61%）、漏框 56/824（6.80%）；质量门命中 684/824（83.01%）、输出框准确比例 684/685（99.85%）、漏框 139/824（16.87%）。两种版本在本次 376 帧不可见样本上均无出框。两者均有框的平均 IoU 分别为 0.798 和 0.801。至少 3 个可见像素的参考下，质量门命中率为 83.21%（684/822）。

严格质量门少了 2 个定位错误框，也拒绝了 81 个原本正确的框。唯一保留下来的 IoU<0.5 样本为 `table_cleanup_to_dishwasher_expert/demo_47` 的初始化片段 0、动作位置 21：可见目标仅 22 像素，IoU 为 0.463。不能将“出框后很准”解读为“很少漏检”，也不能将样本中的 100% 解读为总体保证。本轮未根据留出结果调整检测阈值。

漏框时保留原始 RGB 的输入行为已验证；该实验不测试策略对连续漏框的动作鲁棒性，不覆盖随机裁剪后的独立检测精度，也没有训练新检测器或 VPWEM。检测结果仍是 Grounding DINO tiny 加既有颜色先验，质量门没有提高基础检测器的识别能力。

完整运行产物：`/home/jrw/vpwem_outputs/bbox_online_20261020/`，其中 `summary.json` 为十任务汇总，`code_manifest.json` 记录代码校验值，每个任务子目录保留逐帧坐标、得分、IoU、RGB 和参考 mask。BHR 代码快照位于 `/home/jrw/code_vpwem_bbox_online_20261020/`。17 项相关单元测试通过；另已核对十个离线质量门缓存与在线端的 detector metadata 完全一致。本地系统 Python 缺少 NumPy，单元测试在 BHR 的 vpwem 环境运行。


## 低漏检与精确边界方案比较

2026 年 10 月 20 日在 BHR 比较了 87 组开发集配置，涵盖低分框关联、模板匹配、前后向检查的稀疏光流、局部 DINO 重检、颜色边界细化、组合方案，以及小型分割检测器、分割框细化和仅漏检时补检。小型分割网络为 13,381 个参数，训练 20 epoch；没有训练 VPWEM。

每个数据集从 `mask/train` 分别选 3 个拟合 episode 和 2 个开发 episode；探索测试从 `mask/valid` 选 3 个 episode，排除前次审计的 3 个 episode。十数据集共 1,800 帧拟合、1,200 帧开发、2,400 帧探索测试。小模型只用拟合帧及其分割标签训练。时序策略只使用当前及过去 RGB，候选框关联和局部重检不接收参考 mask、state 或后续帧。

初轮的 57 组配置在开发集按输出框准确比例至少 99.5%、再最大化可见帧命中率进行选择；冻结各类参数后，在 2,400 帧探索测试上比较：

| 方案类别 | IoU≥0.5 命中率 | 输出框准确比例 | IoU≥0.75 命中率 |
|---|---:|---:|---:|
| 原检测器 | 91.90% | 99.35% | 75.39% |
| 严格质量门 | 82.35% | 99.71% | 69.69% |
| 单纯降低阈值 | 94.18% | 99.05% | 76.53% |
| 低分框时序关联 | 93.94% | 99.43% | 76.35% |
| 模板与颜色细化 | 95.08% | 99.62% | 86.31% |
| 光流 | 94.90% | 99.06% | 76.65% |
| 局部重检 | 94.60% | 99.24% | 77.13% |
| 关联与模板及局部重检 | 95.32% | 99.31% | 77.37% |
| 单帧颜色细化 | 92.20% | 99.68% | 84.03% |
| 小型分割检测器 | 96.94% | 98.48% | 79.47% |
| 小模型细化原框 | 91.36% | 98.77% | 74.55% |
| 小模型补检及细化 | 91.66% | 99.03% | 75.69% |

探索测试用于分析方法差异，不作为最终独立确认。小模型单独检测覆盖率较高，但开发和探索测试中的框准确比例均逊于原检测器，因此未选入最终组合。第二轮追加原框与颜色框的混合强度、小模型仅补检和边界扩缩等开发集配置，共 87 组。最终仍按预设的开发集排序冻结组合：初始分数 ≥0.25，低分候选 ≥0.15，关联 IoU ≥0.15；局部状态最长 3 帧；模板相关阈值 0.85；启用局部重检；颜色细化框与原框各取 50%。没有增加神经网络参数。

### 新 episode 实时确认

确认从每个 HDF5 的验证集另选 2 个 episode，排除前次审计与探索测试的全部 episode。每个 episode 在 5 个时间位置初始化，每段 16 帧，每个数据集 160 帧，共 20 个 episode、1,600 帧。该轮每帧实时调用 `PrecisionDetector.detect`，再执行下一动作；直接比较同帧原检测器与组合输出，未重新选择参数。

参考为同次渲染的 class 50 可见 mask，经既有 nearest 缩放到 90×120 后转 tight bbox；至少 1 像素即计可见。十任务每段首帧与在线 RGB 传感器的对齐 MAE 均为 0。指标是该可见 mask 标准下的结果；极小边缘目标的 IoU 对像素采样和取整较敏感。

| 数据集 | 可见帧 | 原命中率 | 组合命中率 | 组合输出框准确比例 |
|---|---:|---:|---:|---:|
| `table_cleanup_to_dishwasher_expert` | 112 | 94.64% | 94.64% | 98.15% |
| `table_cleanup_to_dishwasher_suboptimal` | 112 | 84.82% | 100.00% | 100.00% |
| `table_cleanup_to_sink_expert` | 134 | 93.28% | 93.28% | 100.00% |
| `table_cleanup_to_sink_suboptimal` | 128 | 93.75% | 100.00% | 100.00% |
| `table_setup_from_dishwasher_expert` | 90 | 95.56% | 98.89% | 100.00% |
| `table_setup_from_dishwasher_suboptimal` | 93 | 93.55% | 93.55% | 100.00% |
| `table_setup_from_dresser_expert` | 73 | 87.67% | 87.67% | 100.00% |
| `table_setup_from_dresser_suboptimal` | 115 | 79.13% | 83.48% | 100.00% |
| `unload_dishwasher_to_dresser_expert` | 118 | 83.05% | 83.90% | 100.00% |
| `unload_dishwasher_to_dresser_suboptimal` | 93 | 88.17% | 90.32% | 96.55% |

| 汇总指标 | 原检测器 | 组合方案 |
|---|---:|---:|
| 可见目标 IoU≥0.5 命中率 | 89.33% | 92.70% |
| 输出框准确比例 | 99.79% | 99.50% |
| 漏检率 | 10.49% | 6.84% |
| 可见目标 IoU≥0.75 命中率 | 68.91% | 79.21% |
| 双方有框时平均 IoU | 0.802 | 0.839 |
| 双方有框时平均中心误差 px | 1.291 | 1.128 |

1,068 帧目标可见、532 帧不可见。组合共命中 990 帧，原检测器命中 954 帧，补回 36 帧，未丢失原本命中的帧；漏框从 112 帧降为 73 帧。不可见样本中两版均未出框。组合有 5 个 IoU<0.5 框，原检测器有 2 个；组合提高了覆盖率和边界贴合度，输出框准确比例略降。确认集的准确比例为 990/995=99.497487%，略低于开发阶段 99.5% 的筛选门槛，不能把四舍五入的 99.50% 当成严格达标。

五个错误框均位于画面边缘，参考目标可见区域为 8–32 像素，集中在 `table_cleanup_to_dishwasher_expert/demo_104` 和 `unload_dishwasher_to_dresser_suboptimal/demo_24`。全部计入错误，没有据此修改标注或过滤确认样本。`unload_dishwasher_to_dresser_suboptimal` 的输出框准确比例仅 96.55%，这是当前方案的明确弱点。最长连续可见漏检仍为 16 帧，不能声称时序方案解决了全部遮挡或离开画面问题。

组合实际在线调用平均延迟约 114 ms/帧（共享 BHR GPU，含 DINO、必要的局部重检和后处理，不含环境渲染）。原组合在探索流上的逐帧部署复测与缓存评测的命中计数完全一致，2,400 帧中 3 帧坐标有微小数值差异；最终表格取新 episode 上的同步实时推理结果。

### 代码和复现入口

`PrecisionDetector.detect(rgb, stream_id)` 接收原始 90×120 uint8 RGB，返回 normalized `xyxy` 和 valid 标志。每个环境使用独立 stream_id，episode reset 时必须调用 `reset(stream_id)`。未确认的框不作为可见目标无限传播；检测缺失时，现有 RGB 画框入口可以继续保留原图。该接口没有改变策略输入维度，也未接入或运行 VPWEM。

```python
from momart_bbox_memory.precision_detector import PrecisionDetector

detector = PrecisionDetector(
    "/home/jrw/vpwem_outputs/momart_precision_detector_v1/detector_bundle.json",
    device="cuda:0",
)
detector.reset(stream_id=0)
detection = detector.detect(rgb_uint8_90x120, stream_id=0)
```

BHR 代码快照：`/home/jrw/code_vpwem_bbox_online_20261020/`。
全部实验产物：`/home/jrw/vpwem_outputs/bbox_strategies_20261020/`。
最终独立确认：`confirmation/summary.json`、每任务 `frames.jsonl`、连续 RGB/mask NPZ 和日志。
开发集冻结参数：`comparison_v2/frozen_dev_selection.json`。
小模型权重和拟合记录：`tiny/tiny_mask.pt`、`tiny/training.json`，留在仓库外。

复现顺序为 `tools/run_momart_bbox_corpus_bhr.sh`、`tools/run_momart_bbox_strategy_comparison_bhr.sh`；追加配置的选择使用 `python -m momart_bbox_memory.compare_strategies select --root CORPUS --output comparison_v2`，随后用 `tools/run_momart_bbox_confirmation_bhr.sh` 传入数据集名称执行实时确认，最后调用 `python -m momart_bbox_memory.summarize_confirmation CONFIRMATION_ROOT` 汇总。BHR 快照中的三个启动器分别名为 `run_corpus.sh`、`run_comparison.sh`、`run_confirmation.sh`。不同参数应使用新的输出根目录。

23 项相关单元测试在 BHR vpwem 环境通过，覆盖低分框关联、无当前证据不传播旧框、episode/并行环境状态隔离、裁剪坐标还原、可靠原框不被补检模型覆盖以及已有检测与输入规则。代码提交包括 `dc29a43`、`a67136a`、`2f05ccd`、`8cc7271`、`7aff097`、`d8cb154`。该实验只评估当前交互目标红碗，不包含放置区域检测或策略任务成功率。
