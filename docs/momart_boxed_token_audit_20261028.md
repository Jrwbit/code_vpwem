# 为什么画框可能有效，而加入坐标和 ROI token 后成绩下降？

日期：2026-10-28。范围：MoMaRT 150k 画框版与画框加当前对象 token 版。此次先读取 BHR 实际评测工作树、配置和 manifest，随后 SSH 连接被拒绝，未能运行服务器测试或新的 rollout。下面区分代码事实、解释假设与未完成验证。

## 结论

目前没有发现足以确定这次下降由实现 bug 引起的证据，也不能宣布实现已经通过完整验证。明确发现的风险是：坐标采用高频 Fourier 编码；ROI 与整图共用主 RGB ResNet；原图坐标与经过裁剪的场景特征使用不同参考系；训练离线框与评测在线跟踪框来源不同。这些可能使对象分支比像素画框更敏感，但尚未证明哪一项造成成功率下降。

“画框比不画框好”的前提需要同预算结果支持。现有本地报告中的普通 No-History 是 1M、217/300（72.33%），画框版是 150k、178/300（59.33%）；这两个结果不能用于证明画框收益。若所指的是另一组未画框 150k 对照，需要定位该组的原始结果与配置后再判定。

能够确认的同期比较是：画框版 178/300（59.33%），画框加坐标/ROI token 版 160/300（53.33%），差 18 次成功、6.00 个百分点。两者均为 seed100、150k、batch256、执行块8；单个训练 seed 不足以证明稳定优劣。数字沿用此前逐一核对过的原始结果，本次读取 manifest 确认 checkpoint 身份，未再次加载全部 result.npy。

## 首先纠正“只多加坐标”的理解

画框版仅修改主 RGB 像素，视觉编码器仍自行融合目标和背景。新版本同时增加两个 256 维 token：

1. 最新帧的归一化 xyxy 坐标，经 16 频率 sin/cos 和 MLP 编码。
2. 从未画框的最新 RGB 中裁出的 ROI，缩放到 64×64，经共享主 RGB ResNet 和线性层编码。

它们与任务 embedding 相加后，加入动作 Transformer 的条件序列：`[diffusion_step, position, appearance, obs(t-1), obs(t)]`。原观测仍存在。这改变了条件序列、网络参数、梯度来源及训练随机数消耗，不是单纯增加四个观测数值。

画框可能帮助网络在已有图像空间内定位任务目标，保留目标与手、容器、背景的关系。显式对象 token 则提供额外的注意力入口，错误框、紧裁剪的外观及坐标扰动都可能更直接地影响策略。上述为机制假设，不能代替消融结果。

## 已核查的实现路径

读取位置：BHR `/home/jrw/code_vpwem_boxed_eval_20261026/`，manifest 的 source_commit 为 `dbf6779ac90c7a19b7543a1eb3d5509ef2253f13`。

| 检查项 | 代码事实 | 当前判断 |
|---|---|---|
| 坐标顺序和量纲 | `object_inputs` 接收 normalized xyxy；x 乘图宽、y 乘图高，左上 floor、右下 ceil | 未见 xy 颠倒或像素/归一化混用 |
| 训练最新帧 | `frame=min(start+obs_steps-1,end-1)`；父数据集 `pad_before=0`，末尾重复最后一帧 | 在当前配置下与场景最新帧一致；不能脱离 pad_before 条件泛化 |
| 在线最新帧 | batch-major 展平后，读取 `detections[b*frames+frames-1]` 和 `source[b,-1]` | 静态路由一致 |
| ROI 原图 | 从未画框 RGB 裁 ROI，然后另一路画框 | 未把红色边框当 ROI 外观输入 |
| 训练/评测归一化 | ROI 都是 `2*x-1`；bbox/valid 不作图像归一化 | 未见两端范围不一致 |
| 缺框 | 空框、空 ROI、valid=0；Transformer 将两个对象列设为负无穷 mask | 不只是填零；默认条件 encoder 为逐 token MLP，未见缺框 token 经跨 token encoder 泄漏 |
| PTP 可见性 | 最早动作位置屏蔽最新对象 token；后续位置可读 | 未见最新帧错误泄漏到最早动作位置 |
| DDPM 缓存 | 对象信息打包在 `[B,2,769]` 通道，Transformer 从最后一行解包 | 当前设计避免将对象 token 当时间行裁掉 |
| dropout | 编码后整条条件 dropout，包含 valid 字段，丢弃时对象被屏蔽 | 与整条条件共同丢弃的意图一致 |
| 权重加载 | 评测对 model、model_ema 均 strict=True | 入口要求严格匹配；本次未重新加载权重验证 |

以上是静态核查，不等于动态测试通过。服务器源码中已有帧路由、梯度、mask、实际数据集边界和归一化测试；此次执行 unittest 的 SSH 在启动前失败，不能引用为本次测试成绩。也尚未核对原始训练工作树与评测副本的源码哈希一致性。

## 最值得怀疑的三处设计

### 1. 坐标编码对小幅扰动敏感

实际频率为 `2**arange(16) * pi`，最高为 `32768*pi`。归一化坐标上最高频率的周期是 `2/32768`，映射到 120 像素宽图像仅约 **0.0073 像素**。

本次用 Python 标准库独立复算：对单个 x 坐标的 32 维原始 sin/cos 编码，位移 0.25 像素后的余弦相似度约 0.534595，位移 1 像素后约 0.341935。计算式为 `mean(cos((pixels/120) * 2**k * pi), k=0..15)`，是完整 sin/cos 向量的精确相似度，不依赖起始坐标。

这说明原始编码容易因亚像素变化而明显改变；不说明经过 MLP 的 256 维结果或最终动作一定同样敏感，网络可以学会忽略高频。应在真实 checkpoint 上测量 bbox ±0.25/±1/±2 像素时的位置 token 和固定噪声动作变化，再与低频或直接坐标 MLP 比较。

### 2. ROI 与整图共用可训练 ResNet

`ObjectTokenCondition` 直接调用 `self.encoder.encoder.key_model_map['rgb'](roi)`。ROI 损失梯度会更新同一个主 RGB backbone；因此即便保留场景输入，也不能假设原来的场景表示不变。

ROI 是紧裁剪后拉伸到 64×64，场景分支是 90×120 输入再裁到 75×100，尺度、上下文和形状比例均不同。共享梯度可能有益，也可能冲突。当前配置使用 GroupNorm，不能把此问题直接归因于 BatchNorm 运行均值污染。

先在旧权重上推理时屏蔽对象 token，只能检查该权重对对象分支的依赖；不能将它当成原画框基线，因为共享 backbone 已受对象分支训练影响。判断梯度干扰需单独训练共享/独立 backbone 的对照。

### 3. 原图坐标、随机裁剪与框来源差异

坐标 token 表示原图坐标；场景 RGB 会独立随机裁剪，评测使用对应中心裁剪。框也先画入原图再被裁剪，所以画框位置与画框版 RGB 天然同步；原图坐标 token 则未转换到裁剪坐标系。这是需要验证的表征参考系差异，不是已经确认的错位 bug：若有意表达全图位置，可以保留原坐标，但应让模型明确区分参考系或提供裁剪变换。

训练使用离线 Grounding DINO sidecar，当前评测使用 `combined_blend` 在线检测/跟踪。框误差不仅影响画框，还会同时改变坐标和 ROI 内容；对象分支可能放大该差异。没有框误差和动作敏感性实测之前，不能量化其贡献。

## 能区分原因的验证顺序

1. 恢复 BHR 后先运行已有 8 项测试（`momart_boxed_tokens.test_tokens` 与 `test_implementation_audit`），核对训练/评测源文件哈希和两个实际 train_config 的差异；保留逐项输出。
2. 固定观测和初始扩散噪声，对现有对象版权重分别测试正常输入、屏蔽对象、只保留 position、只保留 appearance、小幅框扰动。报告动作变化及逐任务 rollout，推理屏蔽只作依赖诊断。
3. 新训练先只改单一因素：例如 position 编码改为直接 xyxy/低频 MLP，保持 ROI 和其他配置；另开共享/独立 ROI backbone 对照。坐标与 ROI 分别从头训练消融，才能分辨两个分支的贡献。
4. 将同源框训练作为另一条独立实验，不与编码、ROI、dropout 一起修改。使用相同预算和多个训练 seed；保留 10 配置逐项结果，尤其检查已有 9/30→0/30 和 30/30→22/30 的退化项。

本次没有修改运行中的模型、训练配置或评测任务，没有新增成功率结果。不建议在根因未确认时直接改生产副本并将原 53.33% 解释为“已修复”。

## 证据入口

- 本地既有结果核查：`docs/momart_boxed_nohistory_150k_improvement_20261027.md`。
- 服务器评测 manifest：`/home/jrw/vpwem_outputs/momart_boxed_online_150k_combined_blend_20261026_gpu1/manifest.json`。
- 对象版 checkpoint SHA256：`263f25f6452db9253d1619006a9fa0113c005bbb4881d8ace2da6491253a175b`。
- 已读取源码：评测工作树的 `momart_boxed_tokens/{model,inputs,dataset,train,evaluate,test_tokens,test_implementation_audit}.py`、`momart_boxed_no_history/dataset.py`、`momart_no_history_baseline/model.py`、`cleandiffuser/nn_diffusion/longmem_chitransformerptp.py`，以及 LongMemDDPM 条件处理相关行。
- 验证限制：后续 SSH 到 `bitbot421` 的 `10.1.77.25:22022` 多次 Connection refused；unittest 未启动，训练代码同一性与权重敏感性实测待完成。
