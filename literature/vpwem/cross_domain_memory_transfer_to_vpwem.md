# 可迁移到 VPWEM 的跨领域记忆机制调研

更新时间：2026-07-22

## 0. 结论

VPWEM 当前最值得发展的方向，不是继续强调“工作记忆 + 情景记忆”这一概念组合。MemoryVLA、MovieChat、MemER、HAMLET 等已经占据了相邻叙事。更有论文价值的路线是把 VPWEM 的记忆变成一个**主动、结构化、可修正、受预测目标约束的隐状态**。

建议优先级如下：

| 优先级 | 可迁移创新 | 主要来源 | 对 VPWEM 的实际改动 | 预期价值 |
| --- | --- | --- | --- | --- |
| S | 预测误差/惊奇度驱动的写入、保留和遗忘 | Titans、TRIM-KV、LookaheadKV、DriveWorld | 以未来状态预测误差、动作敏感度、冗余和年龄共同决定写入/合并/淘汰 | 最适合形成主方法，直接回答“何时记、为何记” |
| S | 对象级持久 slot + 全局事件 token | StreamPETR、SparseDrive、MoM、SlotSSM | 将同质 Q-Former queries 拆成对象、阶段/计数和全局 residual slots | 降低记忆干扰，适合遮挡、交换、重现和空间任务 |
| S | 可回看的双层有界记忆 | ReMemR1、SAC、Compressive Transformer | 一层递归 latent summary，一层少量原始 event anchors；读失败时回查 anchors | 缓解不可逆压缩和 summary drift，同时仍保持固定预算 |
| A | 未来预测 + 状态可读性辅助目标 | DriveWorld、Dreamer/RSSM、SlotSSM | 在 action loss 外预测未来 observation latent、对象状态和进度 | 迫使 memory 表示任务状态，而非轨迹 ID |
| A | 冲突检测与重巩固 | HiMe、A-MEM、AFI | 新观测与旧 slot 冲突时 update/delete，而非永远 append/merge | 提高扰动恢复，避免 memory trap |
| A | 多时间尺度 memory experts 与路由 | MoM、Memory Experts、MemGPT | fast event、slow object/state、persistent task prior 分开更新和读取 | 适合多种记忆需求，但实现与消融成本较高 |
| B | 干扰感知 replay/保留 | MIR、持续学习 replay | 优先保护会被新写入破坏的 slot | 适合作为 consolidation 正则，不宜单独做主贡献 |
| B | 反思式 episodic-to-semantic consolidation | Generative Agents、HippoRAG | 周期性把多个事件压成关系/阶段语义 | 对高层 VLA 更自然；低层 DP 中需用 latent 版本 |

最推荐的整合方案是 **Predictive Event-Object Memory（PEOM，暂名）**：

1. 用短期窗口预测下一时刻视觉 latent、对象变化或 subgoal progress。
2. 以 prediction surprise、action influence、novelty 和 uncertainty 计算事件效用。
3. 将高效用观测写入对应对象 slot 或全局 event slot；低效用内容跳过。
4. 新证据和旧 slot 一致则合并，冲突则替换/重巩固，过期且低效用的 slot 衰减。
5. 保留极少量原始 event anchors，供当前查询在 latent summary 不确定时回看。

这比“给 Q-Former 再加一层”更容易形成清晰的新颖性：**VPWEM 不只压缩历史，而是通过预测误差发现事件，通过对象身份维持状态，通过冲突更新修正记忆。**

## 1. VPWEM 当前架构的真实边界

以下判断基于当前仓库代码，而非方法名称。

### 1.1 当前数据流

```text
历史 observation embeddings
        |
        v
固定间隔采样 -> condition/query cache -> Q-Former queries (2/4 tokens)
                                            |
当前短窗口 embeddings ----------------------+
                                            v
                              Diffusion Policy cross-attention
                                            |
                                            v
                                       action chunk
```

当前实现有六个关键性质：

1. `QFormerMemory.inference()` 按固定 `subsample_ratio` 写入，不判断事件是否重要。
2. 缓存溢出只支持 FIFO、random、KMeans、相邻相似合并或不压缩；没有 task-conditioned utility。
3. 所有 query token 同构，没有对象身份、阶段、计数、空间和不确定性的显式职责。
4. memory 仅通过 diffusion action loss 训练，没有未来预测、状态重建、容量约束或冲突更新目标。
5. 历史 cache 使用 `detach()`，因此旧状态是不可回溯优化的缓存；在线更新也没有显式的 overwrite/delete 语义。
6. 输出给策略的是固定 `mem_compress_length` 个 token，但内部 `condition_cache` 和各 Q-Former 层的 `query_cache` 仍可增长到 `cache_max_length`。因此准确说法应是“达到上限后每步成本有界”，不能直接说“全部历史始终只递归存在 2/4 个 summary tokens 中”。

对应代码入口：

- `cleandiffuser/nn_memory/qformer_memory.py`：写入、缓存压缩、Q-Former 和在线 inference。
- `cleandiffuser/diffusion/longmem_ddpm.py`：memory 训练目标和 diffusion 注入。
- `cleandiffuser/nn_diffusion/longmem_chitransformerptp.py`：memory token 与当前观测的拼接和 cross-attention。

这些边界恰好定义了可发表的改造空间：**learned write、typed slots、predictive objective、reconsolidation、revisitable evidence**。

## 2. 大模型与长上下文：可迁移机制

### 2.1 Titans：用“惊奇”决定学什么，并显式遗忘

论文：[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

机制不是普通 KV 检索。Titans 把长期记忆建成可在测试时更新的神经网络，以 associative-memory loss 的梯度衡量 momentary surprise，并用 momentum 累积“事件流”的 past surprise；weight decay 对应自适应遗忘。模型还区分短期 attention、长期 neural memory 和 data-independent persistent memory。

对 VPWEM 最可迁移的不是 test-time gradient descent 本身，而是它的控制信号：

```text
surprise_t = || predicted_latent_t - observed_latent_t ||
eventness_t = eta_t * eventness_(t-1) + theta_t * surprise_t
```

可将 `eventness_t` 用作写入 gate。若直接在机器人 rollout 中在线更新网络权重，稳定性和实时性风险较高；第一版应只更新 memory state，不更新主模型参数。

建议实验：固定写入频率 vs. feature change vs. 单步 surprise vs. momentum surprise；同时报告稀有关键事件 recall 和扰动后的错误记忆持续时间。

### 2.2 TRIM-KV：在 token 生成时预测其长期保留价值

论文：[Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs](https://openreview.net/forum?id=qCaq3jGb0S)

TRIM-KV 为每个 token、layer 和 head 学一个 retention score，并让它随时间衰减；容量满时淘汰低分 token。它通过冻结教师模型蒸馏 importance，加 capacity loss，只训练轻量 gate。一个重要结果是选择性保留有时超过 full cache，说明删除噪声本身可能是正则化。

迁移方式：在每个 VPWEM candidate memory 上预测

```text
u_i(t) = sigmoid(g(z_i, current_obs, task)) * exp(-lambda_i * age_i)
```

容量满时不是 FIFO，而是淘汰最低 `u_i`；`lambda_i` 可以按 slot 类型不同。事件 slot 快衰减，对象状态慢衰减，任务规则可近似不衰减。

训练 importance 的低成本教师信号：移除候选 token 后 action diffusion loss 的增量，或完整历史 teacher 与压缩策略动作分布的差异。

### 2.3 LookaheadKV：用“未来用途”而不是过去 attention 淘汰

论文：[LookaheadKV](https://openreview.net/forum?id=RVLMGPXt2i)

其核心判断是：一个历史 token 是否重要，取决于未来响应会不会使用它，当前 attention 不是可靠代理。论文用轻量模块预测未来的真实 KV importance，避免显式生成 draft。

迁移到 VPWEM：保留分数应预测“未来 H 步 action/状态对该记忆的依赖”，而非仅使用当前 query 和记忆余弦相似度。可用训练数据离线计算 token masking 后未来 `H` 步累计 action loss 差作为 teacher。

这一点与机器人任务尤其契合：目标颜色在当前很多步都不相关，但抓取末端才决定动作。纯当前相关性会过早删除它。

### 2.4 COMI：相关性减去冗余，即边际信息增益

论文：[COMI: Coarse-to-fine Context Compression via Marginal Information Gain](https://openreview.net/forum?id=OGDIXDfaN4)

COMI 定义 Marginal Information Gain，将 query relevance 与相对已保留内容的 redundancy 联合起来，再按组分配压缩预算并在组内合并。

可迁移为：

```text
MIG(z_i) = relevance(z_i, current/task query)
           - beta * max_j similarity(z_i, memory_j)
```

但 VPWEM 还应加第三项 `future_utility`。单独使用 MIG 仍会删除“当前不相关、未来关键”的事件。

### 2.5 SAC 与 Learned Meta-Tokens：让少量原始锚点承载上下文

论文：

- [Autoencoding-Free Context Compression via Contextual Semantic Anchors](https://openreview.net/forum?id=8Pi6Du0n7F)
- [Learned Meta-Tokens for Language Modeling](https://openreview.net/forum?id=eZ5jtFuk3e)

SAC 不把所有上下文强行 autoencode 到新 token，而是选择原上下文中的 anchor tokens，通过标记 embedding 和双向注意力让 anchor 聚合上下文。Meta-Tokens 则表明特殊锚点可压缩前文，并改善长度外推。

对 VPWEM 的含义：完全自由的 Q-Former query 容易产生不可解释的平均表示；可以保留 1–2 个真实观测 anchor，再配 2–4 个 learned summary tokens。前者保存关键视觉证据，后者保存连续状态。

### 2.6 ReMemR1 与 StateLM：记忆压缩不应是不可逆单向过程

论文：

- [Look Back to Reason Forward: Revisitable Memory](https://openreview.net/forum?id=1cymflI2Lh)
- [The Pensieve Paradigm: Stateful Language Models](https://openreview.net/forum?id=GymjF88oGQ)

ReMemR1 指出“边读边记”的单向压缩存在覆盖和信息丢失，因而允许 callback 到完整 memory history。StateLM 则让模型主动调用 pruning、indexing 和 note-taking 工具管理自身状态。

机器人控制不能保存无限完整历史，也不适合调用昂贵 agent tools，但可以做有界版本：

- `latent state slots`：每步递归更新，负责快速控制。
- `event anchor buffer`：只存 top-K 高 surprise 原始 embeddings/帧。
- `callback gate`：当策略 entropy、memory disagreement 或预测误差高时，当前 query cross-attend event anchors。

这形成“快路径 latent state + 慢路径证据回查”，比扩大 cache 更有明确作用。

### 2.7 MemGPT、Generative Agents、A-MEM、HippoRAG：管理与组织比容量更重要

论文：

- [MemGPT](https://arxiv.org/abs/2310.08560)：快慢 memory tiers 与显式换入换出。
- [Generative Agents](https://arxiv.org/abs/2304.03442)：recency、importance、relevance 检索，并把多个 episode 合成为高层 reflection。
- [A-MEM](https://arxiv.org/abs/2502.12110)：动态索引、链接新旧记忆，并由新证据更新旧记忆属性。
- [HippoRAG](https://arxiv.org/abs/2405.14831)：以知识图和 Personalized PageRank 做联想式多跳检索。

直接把文本知识图搬进低层 DP 不合适。可迁移的是两个原则：

1. 写入新证据时，也允许修改与其相关的旧 slot，即 reconsolidation。
2. memory 的读取不只按相似度，还按实体关系和时间连续性传播。

第一版可使用对象 slot 间的 learned relation matrix，不必生成文本或显式 scene graph。

### 2.8 MemoryBank：衰减必须允许“复习后强化”

论文：[MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250)

MemoryBank 使用受 Ebbinghaus forgetting curve 启发的时间衰减，并在记忆再次被使用时强化。迁移时不应机械套指数衰减；更合理的是 slot 被动作 decoder 高依赖、被新观测确认或成功完成 subgoal 后提高 strength。

## 3. 自动驾驶：真正可迁移的场景记忆

### 3.1 StreamPETR：对象 query 是可运动对齐的递归隐状态

论文：[Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection](https://arxiv.org/abs/2303.11926)

StreamPETR 不累计稠密 BEV 历史，而让 sparse object queries 逐帧传播；memory queue 保存历史对象 query，运动属性通过 motion-aware layer normalization 对齐到当前坐标。即便 memory queue 很短，递归 query 仍携带更长期的信息。

迁移到桌面操作：

- 用对象 slot 代替全部同质 Q-Former queries。
- slot 保存 appearance、position、motion、last-seen time、confidence。
- 根据机器人/相机位姿和时间间隔对 slot 做 motion-aware modulation。
- 遮挡时保留 slot；对象重现时按 identity 匹配并更新，而不是生成全新摘要。

这对 shell game、swap blocks、目标短暂出现后消失等任务很自然。

### 3.2 SparseDrive：统一的实例身份和几何 anchor 贯穿感知到规划

论文：[SparseDrive](https://arxiv.org/abs/2405.19620)

SparseDrive 用 instance feature + geometric anchor 表示动态 agent、地图元素和 ego，并用 instance memory queue 做时间建模。高置信目标被赋予稳定 ID，在后续传播中保持身份；规划读取 agent-temporal、agent-agent 和 agent-map 交互。

对 VPWEM 的关键启发不是 3D 检测，而是**memory address 应与下游决策中的实体一致**。若任务动作取决于“哪个块先前被交换”，全局平均 token 是不合适的地址空间。

可以实现为 `K_object + K_event + K_global` 固定预算，而非增加 token 总数。

### 3.3 DriveWorld：记忆要通过预测未来证明自己保存了状态

论文：[DriveWorld](https://arxiv.org/abs/2405.04390)

DriveWorld 的 Memory State-Space Model 分为 Dynamic Memory Bank 和 Static Scene Propagation，并通过当前/未来 3D occupancy 与 action 预测训练。动态和静态信息使用不同传播路径。

对 VPWEM 的直接迁移：

- dynamic slots：手、被操纵物、运动阶段，更新快。
- static slots：目标区、桌面布局、任务约束，更新慢。
- memory 同时预测未来 visual latent/low-dimensional state 与 action。

如果只优化动作 imitation，memory 可能学到 trajectory lookup shortcut；未来状态预测是更直接的 state-belief 约束。

### 3.4 为什么 BEVFormer、普通多帧融合不是优先创新

BEVFormer、SOLOFusion、普通 temporal BEV 或固定多帧 concatenation 证明历史有用，但大多是坐标对齐后的有限窗口融合。除非引入持久 identity、递归状态、显式读写或预测约束，否则迁移到 VPWEM 只相当于换一种 temporal encoder，难形成记忆创新。

## 4. 视频、世界模型与持续学习

### 4.1 Compressive Transformer：压缩记忆需要独立的保真目标

论文：[Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507)

它在普通 memory 之外维护 compressed memory，并比较 autoencoding、attention reconstruction 等压缩损失。核心启示是：只靠最终任务 loss，压缩器未必知道哪些远期细节应该保存。

对 VPWEM 可采用两种教师：

- `attention reconstruction`：压缩 memory 的 decoder attention 逼近完整历史 teacher。
- `action distribution distillation`：压缩策略的 denoising prediction 逼近 full-history 策略。

第二种更符合 VPWEM，但必须配合状态 probe，避免只蒸馏动作查表。

### 4.2 MovieChat 与 VideoChat-Flash：分层 consolidation 和 short-to-long curriculum

论文：

- [MovieChat](https://arxiv.org/abs/2307.16449)
- [VideoChat-Flash](https://openreview.net/forum?id=MUjdNcfNPv)

MovieChat 在短期 buffer 满时把早期 token consolidation 到长期 memory。VideoChat-Flash 用 clip-to-video 的分层压缩，在约 1/50 压缩下保持长视频能力，并采用 short-to-long 训练。

VPWEM 可迁移：把逐帧直接进入同一个 Q-Former 改成 `frame -> event segment -> episode state` 两级压缩，并在训练中逐渐增加 memory delay 和 episode horizon。风险是层级压缩本身已不新；必须与 surprise gating、typed slots 或预测目标结合。

### 4.3 Mixture-of-Memories：多个独立 state 减少相互覆盖

论文：[MoM: Linear Sequence Modeling with Mixture-of-Memories](https://openreview.net/forum?id=3PdOq8Rgue)

MoM 用路由器把输入分配给多个独立 memory states，目标是缓解所有信息挤入单一固定状态造成的干扰。

VPWEM 的轻量版本：不同 query groups 有独立 update gate 和 decay，而 decoder 可同时读取全部 groups。建议不要一开始用稀疏 MoE；先固定三类 slots 并验证 specialization。

### 4.4 Short Window Attention：训练时不要让模型逃避长期记忆

论文：[Short Window Attention Enables Long-Term Memorization](https://openreview.net/forum?id=btgVfhudI1)

该工作发现局部窗口过大时模型会走捷径而不使用长期状态；随机化短窗口能迫使模型学习长期记忆。

迁移非常便宜：训练 VPWEM 时随机化 `obs_steps`、随机遮蔽最近窗口，并对早期关键事件设置更大 delay。它更像训练策略而非主创新，但应该成为所有新 memory 方法的配套设置。

### 4.5 MIR：保护最容易被新信息破坏的记忆

论文：[Online Continual Learning with Maximally Interfered Retrieval](https://arxiv.org/abs/1908.04742)

MIR 估计一次新数据梯度更新后，哪些旧样本 loss 上升最多，并优先 replay 这些样本。对于 VPWEM，可将“参数更新”改成“slot 写入”：模拟写入候选事件后，测哪些已有 slots 的 state/action probe 退化，优先保护或分配到不同 slot。

这个计算可能较贵，适合离线训练正则或 teacher 标注，不适合每个控制步在线运行。

## 5. 推荐主方法：Predictive Event-Object Memory

### 5.1 固定容量状态

```text
M_t = {
  object_slots: K_o x D,
  event_slots:  K_e x D,
  global_slots: K_g x D,
  anchors:      K_a x D
}
```

每个 slot 另带少量 metadata：`age, strength, uncertainty, last_seen, type`。总容量固定，episode 变长不增加 decoder token 数和存储。

### 5.2 写入效用

```text
surprise_t = || stopgrad(z_t) - pred(M_(t-1), a_(t-1)) ||
influence_t = D_KL(pi(. | M, z_t) || pi(. | mask(z_t), M))
redundancy_t = max_i cosine(z_t, M_i)
utility_t = w_s * surprise_t
          + w_a * influence_t
          + w_u * uncertainty_t
          - w_r * redundancy_t
```

线上不必真的跑两次完整 diffusion 来算 `influence_t`；用训练出的轻量 retention head 预测 teacher influence。

### 5.3 三种写操作

- `merge`：与已有 slot 身份一致且状态不冲突，门控融合并增强 strength。
- `overwrite/reconsolidate`：身份一致但位置、状态或阶段冲突，以不确定性加权替换旧内容。
- `allocate`：新对象或高 surprise 新事件，写入空 slot；无空位时淘汰最低长期效用 slot。

不要只做 soft attention 加权平均。平均会同时保留互相矛盾的旧状态和新状态。

### 5.4 读取

当前 observation、任务条件和 diffusion timestep 共同生成 read query。先读 typed latent slots；若预测不确定性或 slot disagreement 高，再读 event anchors。memory 注入建议使用 gated cross-attention 或 FiLM modulation，并保留现有 concatenation 作为对照。

### 5.5 损失

```text
L = L_diffusion_action
  + lambda_f * L_future_latent
  + lambda_s * L_task_state_probe
  + lambda_d * L_full_history_distill
  + lambda_c * L_capacity
  + lambda_cf * L_counterfactual_memory
```

- `L_future_latent`：预测未来 observation embedding、对象位置/可见性或 low-dimensional state。
- `L_task_state_probe`：颜色、计数、目标身份、subgoal progress 等任务可用标签；只用于训练或分析。
- `L_full_history_distill`：完整历史 teacher 的 action/attention 蒸馏。
- `L_capacity`：约束平均写入率、slot 使用均衡和重复度。
- `L_counterfactual_memory`：正确 memory 的动作应优于错误轨迹 memory，防止忽略历史。

## 6. 最小可行版本与开发顺序

### Phase 0：先做诊断，不改主结构

1. 记录每步 memory attention、cache 内容和最终 tokens。
2. 做正确历史、清空、打乱、跨轨迹替换、只保留关键事件五种干预。
3. 训练 task-state probe 与 trajectory-ID probe。
4. 画成功率、延迟、显存随 episode length 和 `cache_max_length` 的曲线。

这一步能验证当前 VPWEM 是否真的使用了长期 memory，并校正论文叙事。

### Phase 1：Surprise-Retention VPWEM

只增加 future latent predictor、retention head、age decay 和 top-K eviction，保持 Q-Former 与 diffusion decoder 不变。这是最小改动，也是最容易证明“何时写”有效的一版。

必要对照：定频采样、FIFO、相邻相似合并、feature delta、random、oracle event、learned retention。

### Phase 2：Typed Slot VPWEM

将 query tokens 分成 object/event/global 三组；每组独立 update gate 和 decay。先不依赖 oracle segmentation，可用视觉 patch-to-slot attention 或预训练视觉特征聚类形成对象候选。

必要对照：相同总 token 数的 homogeneous queries、随机分组、共享 gate、独立 gate。

### Phase 3：Revisitable Anchors + Reconsolidation

加入 top-K 原始 event anchors、冲突检测和 overwrite。重点测外力扰动、物体换位、误动作、环境状态突然变化后的恢复时间。

## 7. 论文实验必须回答的问题

| 问题 | 最关键实验 | 否则会留下的替代解释 |
| --- | --- | --- |
| 记住的是任务状态吗？ | state probe、trajectory-ID probe、同当前帧异历史 | 只是定位训练轨迹 |
| learned write 真有用吗？ | 同写入预算比较定频、变化点、随机和 oracle | 只是多保存了帧 |
| typed slots 真有专门化吗？ | slot swap、slot-type mask、对象身份追踪 | 只是更多参数 |
| predictive loss 的作用是什么？ | action-only vs. future-only vs. joint | 辅助 loss 只做 regularization |
| 能否修正旧记忆？ | 状态突变与恢复时间 | memory 加强动作惯性 |
| 是否真正长度外推？ | train horizon `H`，test `2H/4H/8H` | 只拟合训练窗口 |
| 是否计算有界？ | episode length-显存-延迟曲线，含内部 cache | 只隐藏了增长成本 |
| 原始 anchors 是否必要？ | latent-only、anchor-only、二者结合 | 双层结构没有实际作用 |

优先任务建议：MIKASA RememberColor3/5/9、ShellGame、swap/交换、多对象遮挡、带外力扰动的 Robomimic LongSquare，以及 RoboMME 中按 temporal/object/spatial/procedural 分类的子集。普通 LIBERO 平均成功率不足以证明记忆机制。

## 8. 新颖性判断

### 单独做，创新偏弱

- Q-Former 改成 Mamba/xLSTM。
- 再增加一组“长期记忆 token”。
- FIFO 改成 KMeans 或相似度合并。
- 加 scene graph，但依赖 oracle object/subgoal。
- 只增加 future prediction head，没有 learned memory management。
- 只做短期/长期两层 buffer。

### 组合后有主方法潜力

1. **Predictive surprise-gated write + fixed typed slots**：机制清楚，代码改动适中，最好。
2. **Object identity memory + conflict-aware reconsolidation**：对 manipulation 很有针对性，恢复实验容易形成强证据。
3. **Latent belief + revisitable event anchors**：解决不可逆压缩，适合超长延迟任务。
4. **完整三者结合**：论文故事最强，但应按 Phase 1-3 逐步验证，避免一次实现后无法归因。

## 9. 与现有 VPWEM 相关工作清单的关系

仓库已有 `vpwem_p0_2024_2026/deep_reading_and_ideas.md` 对 MemoryVLA、HAMLET、MemER、HiMe、RoboMME、MIKASA、SlotSSM、TRM-VLA、AFI 等机器人/VLA 论文做了较充分精读。本报告刻意不重复其结论，而补足以下跨域机制：

- LLM：惊奇写入、learned retention、未来用途、主动管理、可回看和动态链接。
- 自动驾驶：对象身份、几何/运动对齐、动态与静态状态分离、预测式 world memory。
- 视频：分层 consolidation、原始 anchor 与长短课程。
- 持续学习：干扰感知保护和 replay。

最终的取舍原则不是“某论文也叫 memory”，而是它能否对应 VPWEM 的一个明确算子：`write / merge / overwrite / evict / read / train / verify`。
