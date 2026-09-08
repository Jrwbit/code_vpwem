**核心结论**

VPWEM 是一项典型的“问题驱动型”研究。它不是先发明复杂模块、再寻找有利任务，而是从“短上下文策略解决不了非马尔可夫任务”出发，逐层建立假设、设计模型、选择正负对照数据集，再通过主实验、效率实验和消融实验闭合证据链。

## 一、从 idea 到方法，逻辑是怎么走出来的

### 1. 先找到一个真实且可验证的问题

普通视觉运动策略近似学习：

```text
aₜ ~ π(oₜ₋L₊₁ : oₜ)
```

也就是只看最近 L 帧。

这对 Square 之类任务通常够用，因为当前画面已经包含完成动作所需的信息。但 ShellGameTouch 不一样：

1. 球一开始出现在某个杯子下面；
2. 随后球被遮住；
3. 机器人过一段时间才需要选择杯子；
4. 做决策时，当前画面里已经没有答案。

这时，同一个当前观测可能对应不同正确动作。任务本质上是 POMDP，策略需要的是：

```text
π(aₜ | o₀, …, oₜ)
```

而不是只看当前或最近几帧。

### 2. 最直接方案：把上下文拉长

但这会带来两个问题：

- Transformer 注意力计算随长度近似 O(L²) 增长，推理越来越慢；
- 更长历史不一定更好，策略可能利用过去动作、机器人姿态等伪相关信息，出现 copycat 或 causal confusion。

因此，VPWEM 真正的研究问题不是“怎样使用更长上下文”，而是：

**核心研究问题：能否保留对决策有用的长期信息，同时让每一步的计算和存储保持有界？**

这是比“加长 context”更准确、更有论文价值的问题。

### 3. 核心假设

VPWEM 把历史分成两部分。

最近 L 个高分辨率观测是工作记忆 Working Memory：

```text
Wₜ = oₜ₋L₊₁ : oₜ
```

更早的历史由学习型压缩器 Cφ 压成固定数量 M 个情景记忆 token：

```text
Eₜ = Cφ(o₀ : oₜ₋L)
```

最终策略是：

```text
πθ(aₜ : aₜ₊H | Wₜ, Eₜ)
```

这个假设非常干净：

- 最近信息需要细节，所以原样保留；
- 很早的信息只需要保留“发生过什么”，所以压缩；
- 压缩器不做人工监督，而是和动作策略一起使用行为克隆损失训练；
- 哪些历史重要，由下游动作预测目标决定。

需要注意：这里的 episodic memory 是单个 episode 内的轨迹记忆，不是跨 episode 的终身经验库。评估时每个新 episode 都会清空 cache。

## 二、VPWEM 模型内部到底发生了什么

可以把一次决策理解成以下几步。

### 1. 把一帧观测编码成 token

多视角 RGB、机器人 proprioception 经过视觉编码器和 MLP，得到 256 维表示：

```text
oₜ ∈ R²⁵⁶
```

最近的 L 个 token 直接组成工作记忆。

### 2. 旧帧进入长期记忆系统

当观测滑出工作窗口后：

- 加上 episode timestep 的位置编码；
- 放进 observation cache；
- 用固定数量的可学习 query token 查询历史；
- 输出 M 个固定大小的记忆 token。

核心实现在：

```text
cleandiffuser/nn_memory/qformer_memory.py
```

QFormer 每一层大致包含：

- self-attention：当前 query 查询过去的 summary query；
- cross-attention：query 查询历史 observation cache；
- FFN：更新压缩表示。

它不是把历史简单平均，而是在学习：“为了预测正确动作，过去哪些信息值得留下？”

### 3. 长短记忆共同条件化扩散策略

扩散模型看到的条件序列是：

```text
diffusion timestep token
+ M 个 episodic memory tokens
+ L 个 working memory tokens
```

然后预测整个 action chunk 的噪声。核心实现位于：

```text
cleandiffuser/nn_diffusion/longmem_chitransformerptp.py
```

训练目标仍然是普通扩散噪声预测：

```text
Loss = || ε - εθ(aₖ, Wₜ, Eₜ, k) ||²
```

VPWEM 没有额外的“记忆重建损失”。因此，memory compressor 完全依靠动作预测 loss 间接学习。

这种设计的优点是记忆会趋向任务相关信息；风险是没有直接监督时，记忆可能退化、塌缩或学习到新的伪相关特征。

### 4. PTP 在这里是什么

代码中的 VPWEM 实际建立在 ChiTransformerPTP 上。PTP 是 Past Token Prediction。

以常见配置为例：

```text
obs_steps = 2
action_steps = 8
horizon = 9
```

9 个动作训练 token 中：

- 第 1 个是过去动作，作为辅助预测；
- 后 8 个是需要执行的未来动作。

评估时从 `obs_steps - 1` 开始截取未来 8 步。PTP 的作用不是直接提供长期记忆，而是帮助扩散 Transformer 更好地学习时序结构。

## 三、数据集为什么这样选

VPWEM 很值得学习的一点，是它没有只挑对自己有利的数据集。

| 研究问题 | 数据集 | 实验角色 |
| --- | --- | --- |
| 真正需要记忆时是否有效 | MIKASA ShellGameTouch、RememberColor3 | 强正对照 |
| 自然长时程任务是否有效 | MoMaRT 五类移动操作任务，expert/suboptimal | 现实复杂场景 |
| 不需要记忆时会不会伤害性能 | Robomimic Square、Transport | 负对照 |
| 能否控制变量研究记忆 | Long-Horizon Square | 消融平台 |

这套选择对应四个问题：

1. **Can it work?** MIKASA 中答案会从当前观测消失。
2. **Does it matter in realistic long tasks?** MoMaRT 包含多子目标移动操作。
3. **Does it hurt?** 标准 Robomimic 近似 Markov，加入记忆不应明显退化。
4. **Why does it work?** LongSquare 可以控制记忆长度、cache 和 token 数。

选择数据集时要学会区分：

- 长 horizon 不等于需要长期记忆；
- 非马尔可夫任务的关键是：当前相同观测下，正确动作依赖过去不可见变量；
- 最好同时有正对照和负对照，否则性能提升可能只是来自模型变大。

## 四、VPWEM 里的 trick，各自在解决什么

这些 trick 不是一个平铺的技巧列表。每一个都应该对应具体瓶颈。

### 1. 核心方法 trick

- **固定数量记忆 token**：解决完整历史越来越长的问题。
- **Working + Episodic 双通路**：避免把近期精细控制信息也过度压缩。
- **位置编码**：否则压缩器只知道发生过什么，不知道何时发生。
- **summary cache + observation cache**：一个记录已有抽象，一个保留有限原始证据。
- **短、长条件分别 dropout 0.3**：减少策略只依赖某一路条件的风险。
- **FIFO cache**：消融中 FIFO 65%、K-Means 65%、AdjSim 67%，差异很小，因此主实验选择更简单的 FIFO。

### 2. 训练与效率 trick

- **历史随机分段采样**：训练时每隔 r 帧划一个区间，从每段随机取一帧，而不是读取全部历史。
- **训练时随机、推理时周期采样**：相当于一种时间增强，但必须确保训练与推理的采样率一致。
- **detach cache**：历史 token 放入 cache 前切断梯度，避免完整 BPTT。
- **按历史长度分组 batch**：减少 variable-length padding 的计算浪费。
- **预计算 `dp_emb`**：先训练短上下文视觉编码器，再把 256 维 embedding 写入数据集，长期策略训练时不再反复运行 ResNet。
- **EMA、action normalization、action chunking**：提高扩散策略稳定性。
- **三个随机种子 100/200/300**：避免用单次最好结果代表方法。

预计算 embedding 是工程加速，而不是 VPWEM 的核心创新。`*_emb.py` 版本也不是视觉编码器端到端训练；它只联合训练 memory compressor 和 diffusion backbone。

## 五、实验是怎样一层层设计出来的

一项完整工作至少要回答四层问题。

### 第一层：主结果——方法有没有用

比较对象包括：

- RNN；
- DP；
- DP-PTP；
- MaIL；
- DP-VPWEM；
- MaIL-VPWEM；
- MIKASA 上额外比较 VLA 模型。

论文结果包括：

- MIKASA：DP-VPWEM 在 ShellGameTouch 为 91%，RememberColor3 为 82%；
- MoMaRT 平均：DP-VPWEM 69%，MaIL-VPWEM 71%，相比对应基线整体提高；
- 标准 Robomimic：与基线大体持平。

这正好符合假设：

```text
强记忆任务：大幅提升
自然长时程任务：中等提升
近似 Markov 任务：基本不退化
```

如果所有数据集都同幅度提升，反而可能说明收益来自参数量、训练时间或其他混杂因素。

### 第二层：效率——为什么不直接加长 context

论文把 DP-PTP 的 context 从 4 拉到 128，再和 VPWEM L2S2 比较。

在 MoMaRT unload-suboptimal 上：

| 方法 | GPU 显存 | 推理时间 | 成功率 |
| --- | ---: | ---: | ---: |
| DP-PTP，context 128 | 1792 MB | 0.72 s | 40.4% |
| VPWEM，L2S2 | 734 MB | 0.22 s | 58.3% |

这个实验非常重要，因为它证明收益不是“偷看了更多帧”这么简单，而是“压缩历史比直接拉长窗口更有效”。

### 第三层：消融——究竟是什么起作用

LongSquare 消融回答：

- L1S1 已经有 65%，说明一个长记忆 token 就可能保存初始目标；
- L2S2 为 70%，L4S2 达到 85%，但 token 过多也可能降到约 50%；
- compressor 从 1 层到 3 层：51% → 67% → 73%；
- cache 从 2 增加到 8：50% → 65%，再到 16 只有 66%，收益饱和；
- Random cache 只有 54%，FIFO/K-Means/AdjSim 分别约 65/65/67%。

这里还可以批判性地看：L2S2 并不是消融中的最高数值。选择它更像是在效果、参数量和统一配置之间做折中，而不是“搜索出的最优超参数”。

### 第四层：统计与复现

正式结论使用：

- 3 个随机种子；
- 每个种子约 30 个评估 episode；
- 报告 mean ± std；
- 训练 60 万步（MIKASA）或 100 万步（其他）；
- batch 64、学习率 1e-4、EMA 0.999；
- 50 个扩散采样步。

这才叫完整实验。单个 seed 跑通只叫 feasibility check。

## 六、在这个仓库里跑完整实验的正确顺序

不要直接从“三个 seed × 一百万步”开始。

### 阶段 0：冻结实验定义

先固定：

- commit hash；
- 数据集文件及哈希；
- 短上下文 checkpoint；
- `obs_steps/action_steps/horizon`；
- L/M/N、cache size、subsample ratio；
- 环境和成功判定；
- 评估 episode 数。

论文主配置可概括为：

```text
working tokens L = 2
episodic tokens M = 2
compressor layers N = 2
cache size = 8
cache method = FIFO
subsample = 5（MIKASA）/ 20（其他）
action chunk = 8
```

### 阶段 1：训练短上下文基线

以 Robomimic LongSquare 为例：

```bash
python pipelines/dpptp_robomimic_image.py \
  --config-name=longsquare_abs \
  dataset_path=PATH_TO_DATASET \
  save_path=PATH_TO_OUTPUT \
  seed=100 \
  +inference_action_steps=8
```

先确认：

- loss 正常下降；
- checkpoint 能加载；
- baseline rollout 有合理动作；
- 数据集 action 和环境 action 坐标系一致。

### 阶段 2：预计算 embedding

必须复制数据集，不要直接破坏原始 HDF5：

```bash
python pipelines/rewrite_with_embeddings_robomimic.py \
  --config-name=longsquare_abs \
  dataset_path=PATH_TO_COPIED_DATASET/demos_emb.hdf5 \
  model_path=PATH_TO_SHORT_CONTEXT/model_1000000.pt \
  device=cuda:0
```

然后检查每个 demo 是否都有形状正确的 `dp_emb`。

### 阶段 3：VPWEM 烟雾测试

```bash
python pipelines/longmemdpptp_robomimic_image_emb.py \
  --config-name=longsquare_abs_emb \
  dataset_path=PATH_TO_COPIED_DATASET/demos_emb.hdf5 \
  save_path=PATH_TO_OUTPUT \
  gradient_steps=10 \
  save_freq=5 \
  log_freq=1 \
  +inference_action_steps=8
```

这里检查的是：

- sampler 返回 `dp_emb` 和 `dp_emb_mem`；
- `ep_step` 与 memory 长度一致；
- QFormer 输出 `[B, M, 256]`；
- loss、梯度和 EMA 都是有限值；
- checkpoint 可以恢复。

### 阶段 4：单 seed 哨兵实验

先跑 seed 100 的 5 万或 10 万步，并进行小规模 rollout。

如果 success 始终为 0，应该立即停下来检查数据、动作空间、encoder checkpoint 和训练/推理 memory 对齐，而不是继续跑满 100 万步。

### 阶段 5：完整主实验

只有哨兵实验合理后，才运行：

```text
方法：DP、DP-PTP、MaIL、VPWEM
种子：100、200、300
训练：600k 或 1M
评估：每种子 30 episodes
```

每个运行目录应保存：

```text
resolved Hydra config
代码 commit
数据集标识
短策略 checkpoint 路径/哈希
训练日志
模型和 scheduler
评估 result.npy
视频样例
```

### 阶段 6：消融与效率实验

消融通常先用单 seed 筛选，再对最终配置运行三个 seed：

- L/M；
- compressor layers；
- cache size；
- FIFO/Random/K-Means/AdjSim；
- subsample ratio；
- 有无 memory；
- 有无位置编码；
- 长、短条件 dropout。

最后再测显存、训练耗时和推理延迟，不能只报告成功率。

## 七、当前仓库复现状态里需要特别警惕的地方

当前本地 LongSquare VPWEM 的 100、200、300 三个 seed 都是：

```text
mean_success = 0
```

因此，目前这份 checkout 不能称为已经复现论文的 LongSquare 结果。

优先排查点包括：

1. LongSquare embedding 评估创建 `QFormerMemory` 时没有传入 `dataset_memory_subsample_ratio`，会使用默认值 10，而训练配置是 20，造成训练/推理 memory 更新频率不一致。
2. 论文表格、部分 YAML 和 shell 脚本之间存在配置漂移。例如有的 YAML 是 `M=4、N=1、AdjSim`，论文主设置是 `M=2、N=2、FIFO`。
3. 训练和评估复用同一个 `work_dir` 时，评估 Hydra 配置可能覆盖训练时保存的 `.hydra/config.yaml`，以后很难知道 checkpoint 当时究竟使用了什么配置。
4. 部分评估脚本仍有硬编码 checkpoint 路径。
5. README 中暴露了 W&B API key。应立即撤销该 key，并从仓库和 Git 历史中移除，不能继续使用。

这些工程问题本身也是 VPWEM 能教我们的重要一课：论文方法正确，不代表公开仓库里的任意配置组合都能自动复现论文数字。

## 八、从 VPWEM 最值得学走的科研方法

### 1. 先定义任务为什么非马尔可夫

不是因为 episode 长，而是因为正确动作依赖已经从当前观测中消失的信息。

### 2. 把方法写成可证伪的假设

例如：

**可证伪假设：固定大小的学习型长期记忆，应在记忆任务上提升、在 Markov 任务上不明显伤害，并且比直接扩大 context 更高效。**

### 3. 数据集是证据链，不是任务列表

- MIKASA 是正对照；
- Robomimic 是负对照；
- MoMaRT 是现实验证；
- LongSquare 是机制分析。

### 4. 每个 trick 必须对应一个 failure mode

- subsampling 对应计算量；
- detach 对应 BPTT；
- 位置编码对应时序丢失；
- dropout 对应过度依赖；
- FIFO 对应简单有效。

### 5. 先证明必要性，再证明优越性

先证明短上下文确实失败，再比较 memory 方法。否则改进可能只是因为模型更大。

### 6. 完整实验不等于“训练跑完”

真正的完成标准是：

- 配置可追溯；
- 三个随机种子；
- 训练与评估一致；
- 有效性实验；
- 负对照；
- 消融；
- 效率分析。

## 九、最终可复用的研究模板

把 VPWEM 抽象成一套可以复用到自己课题中的流程：

```text
观察失败案例
→ 判断缺失的是哪类信息
→ 提出最小机制假设
→ 设计正对照、负对照和现实任务
→ 单变量验证核心机制
→ 做小规模烟雾测试和哨兵实验
→ 冻结配置后跑三个随机种子
→ 用主结果、消融和效率共同支持结论
```

这条链条，比具体的 QFormer 或 Diffusion Policy 实现更值得从 VPWEM 中学习。

## 十、关键代码索引

```text
论文：
vpwem.pdf

短上下文 DP-PTP：
cleandiffuser/nn_diffusion/chitransformerptp.py
pipelines/dpptp_robomimic_image.py

长期记忆压缩器：
cleandiffuser/nn_memory/qformer_memory.py

带长期记忆的扩散模型：
cleandiffuser/diffusion/longmem_ddpm.py
cleandiffuser/nn_diffusion/longmem_chitransformerptp.py

长期历史采样：
cleandiffuser/dataset/dataset_utils.py
cleandiffuser/dataset/robomimic_dataset_longmem.py

预计算视觉 embedding：
pipelines/rewrite_with_embeddings_robomimic.py

VPWEM 训练与评估：
pipelines/longmemdpptp_robomimic_image_emb.py
pipelines/longmemdpptp_robomimic_lhsq_image_emb_eval.py

配置：
configs/dp/robomimic_multi_modal/chi_transformer_ptp_longmem/
configs/dp/mikasa/chi_transformer_ptp_longmem/
configs/dp/momart/chi_transformer_ptp_longmem/
```
