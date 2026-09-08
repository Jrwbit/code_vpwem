# VPWEM 相关 P0 论文精读与 Idea 地图（2024–2026）

整理时间：2026-07-22

## 0. 先给结论

原始索引共有 26 篇 P0（2024: 4，2025: 4，2026: 18）。逐篇复核后，不能把这 26 篇都当作直接竞争者：

- **13 篇保留为 P0**：直接改变 VPWEM 的新颖性表述、强基线或核心实验设计。
- **7 篇降为 P1**：机制可迁移或邻近，但任务、记忆对象或控制层级不同。
- **6 篇降为 P2/排除**：主要是关键词误检，和 VPWEM 的单回合视觉运动记忆问题没有直接关系。

最重要的研究判断：

1. **MemoryVLA 是正面竞争者。** 它同样使用工作记忆、长期感知/认知记忆、固定容量合并和 diffusion action expert。VPWEM 不能再把“双记忆”或“固定容量”本身作为主新颖点。
2. **HAMLET 是最应复现的紧凑历史基线。** 它用 moment tokens + 两层 Transformer，在真实历史依赖任务上将 GR00T N1.5 从 29.2% 提到 76.4%，同时成本远低于多帧拼接。
3. **RoboMME 推翻了“一种记忆表示通吃”的假设。** 16 个任务上，视觉帧采样总体最强（44.51%），符号记忆擅长计数，视觉记忆擅长运动/时间敏感任务，循环记忆反而最弱。
4. **MemER 证明显式关键帧选择很强。** 三个真实长时任务平均进度 96%，长上下文基线仍低 34%；VPWEM 必须和 keyframe retrieval 在相同预算下比较。
5. **ALT、AFI、SemanticVLA 暴露“记忆即查表”的风险。** VPWEM 的提升可能来自更准确地索引训练轨迹，而非恢复不可观测任务状态。
6. **最有价值的 idea 不是再加一个 memory 名词，而是让记忆具有可证伪的功能：** 什么时候写、写了什么、是否可修改、是否编码任务状态而不是轨迹 ID、长度外推时是否仍有用。

## 1. 范围与证据等级

这里的 P0 复核标准是：论文至少满足以下一项。

- **P0-A 直接方法竞争**：机器人/VLA/visuomotor policy 在单次交互中读取历史并直接影响动作。
- **P0-B 核心评测**：专门测机器人记忆，能决定 VPWEM 的任务、指标或基线是否充分。
- **P0-D 机制诊断/反例**：直接挑战“长期记忆提升来自状态推理”的解释。
- **P0-T 同问题特化解法**：在明确的非马尔可夫操作任务上使用历史，虽不完全通用，但可形成强任务基线。

证据标记：

- **F**：阅读全文，核对方法、主表、消融和局限。
- **A**：会议官方摘要/聚合原始摘要；正文未公开或无稳定入口，只作边界判断。
- **X**：阅读全文或摘要后确认是关键词误检。

## 2. 26 篇原始 P0 的资格审计

| 年份 | 论文（简称） | 原级别 | 复核 | 证据 | 理由 |
|---|---|---:|---:|---:|---|
| 2024 | Embodied Active Defense | P0 | P2 | F | 用 recurrent feedback 防御对抗贴片，不研究任务历史记忆 |
| 2024 | READ-PVLA | P0 | P2 | F | PVLA 指 partial video-language alignment，不是 policy/VLA |
| 2024 | Offline Risk-sensitive RL with Partial Observability | P0 | P1 | F | POMDP 人机任务分配；不是视觉运动历史压缩 |
| 2024 | Social Robot in a Hospital Memory Clinic | P0 | 排除 | F | “Memory Clinic” 是医院科室，纯关键词误检 |
| 2025 | LRAM / xLSTM | P0 | P1 | F | 通用离线 action model 与高效 recurrence；不是专门的 episodic memory |
| 2025 | SAM2Act / SAM2Act+ | P0 | P1 | F | 空间记忆操作强基线，但依赖 SAM2 memory bank 和外部 motion planner |
| 2025 | 3D-Mem | P0 | P1 | F | 3D 探索、EQA、导航记忆，不是低层 visuomotor policy |
| 2025 | Spatial Feature Memory | P0 | P2 | F | 具身目标检测的 TSDF 特征地图，不生成操作动作 |
| 2026 | Demystifying Diffusion Policies / ALT | P0 | P0-D | F | 直接挑战 Diffusion Policy 的泛化解释 |
| 2026 | Embodied Agents Meet Personalization | P0 | P1 | F | 高层 agent 个性化知识记忆，非低层控制；其干扰结论很有用 |
| 2026 | HAMLET | P0 | P0-A | F | 压缩时刻 token 条件化 VLA 动作，直接竞争 |
| 2026 | MIKASA | P0 | P0-B | F | 记忆密集型机器人 benchmark，VPWEM 已使用 |
| 2026 | MemoryVLA | P0 | P0-A | F | 工作/长期双记忆 + diffusion action expert，最高重叠 |
| 2026 | MemER | P0 | P0-A | F | 关键帧经验检索 + 分层 VLA，真实分钟级任务 |
| 2026 | HiMe | P0 | P0-A | F | Executor/Sentry/Planner 分层记忆系统，直接处理长时控制 |
| 2026 | RoboMME | P0 | P0-B | F | 16 个机器人记忆任务和 14 种 memory-VLA 变体 |
| 2026 | SOMA / Out-of-Vision Spatial Memory | P0 | P1 | A | 持久空间地图，只处理视野外对象定位；论文页未挂正文 |
| 2026 | AFI / Memory Traps | P0 | P0-D | F | 直接诊断并修复 VLA 轨迹记忆陷阱 |
| 2026 | CycleManip | P0 | P0-T | F | 周期操作中的历史采样和进度监督 |
| 2026 | OptimusVLA | P0 | P0-A | F | 双记忆 VLA，但记忆主要是跨轨迹动作先验和近期动作块 |
| 2026 | SemanticVLA | P0 | P0-D | F | 明确区分语义推理与动作记忆，提供 OOD 诊断 |
| 2026 | TRM-VLA | P0 | P0-A | F | 缓存关键帧 CoT/视觉推理特征条件化 action expert |
| 2026 | RGMP | P0 | P1 | F | 循环几何先验主要服务人形操作泛化，不是长时记忆 |
| 2026 | Embodied-SlotSSM / LIBERO-Mem | P0 | P0-A/B | F | 对象级 transient memory + 新 benchmark，虽依赖 oracle subgoal |
| 2026 | Event-based Memory-Augmented Policy | P0 | P2 | F | 无人机事件流的低延迟累计状态，问题和模态差异较大 |
| 2026 | Conflict-Aware Memory | P0 | P2 | A | LLM planner 的向量知识库冲突清洗，不是 episode visuomotor memory |

## 3. 真 P0 精读

### 3.1 MemoryVLA：最危险的正面竞争者

**一句话。** 当前帧经 7B VLM 产生 256 个 perceptual tokens 和 1 个 cognitive token；它们既构成 working memory，又查询固定容量的 Perceptual–Cognitive Memory Bank（PCMB），最后条件化 300M 的 diffusion action expert。

**机制。** PCMB 分感知流和认知流，每个历史时刻保存一组 token；当前 token 通过带时间编码的 cross-attention 读取历史，门控融合后写回。当容量达到 `L`，分别寻找相邻条目中余弦相似度最高的一对并平均合并。它不是无限窗口，但每个 perceptual entry 有 256 tokens，容量和成本仍明显受 `L` 影响。

**关键证据。** SimplerEnv-Bridge 71.9%，比 CogACT-Large 高 14.6 点；LIBERO 平均 96.5%；MIKASA-Robo 平均 41.2%，比次优高 11.8 点；真实长时任务 83%，比 CogACT 高 26 点。双流记忆 71.9%，单独 cognitive/perceptual 只有 63.5/64.6%；token merge 71.9%，FIFO 66.7%。

**硬伤。** Backbone 和训练规模很大；不同任务用不同 memory length（常规 16，长时真实任务 256），不等于对任意长度都恒定；相邻相似条目平均可能抹掉稀有但关键事件；结果没有证明 memory 不在做 trajectory retrieval。

**对 VPWEM。** 不能再只说“固定容量双记忆”。应强调并验证：VPWEM 将所有越窗历史**递归压成固定数量 summary tokens**，每步成本不随 episode 或手工设定 `L` 增长；同时它可落在非 VLA 的低层 DP 上。但这必须用长度-成本曲线和同预算实验支撑。

### 3.2 HAMLET：最值得复现的轻量基线

**一句话。** 每个时刻用 4 个 learnable moment tokens 压缩图像和指令，以 time-contrastive learning 初始化，再用两层 causal Transformer 聚合最近 `T` 个时刻，输出历史增强特征给原 VLA action expert。

**关键证据。** 三个真实历史任务上 GR00T N1.5 为 29.2%，多帧为 45.8%，HAMLET 为 76.4%。RoboCasa-100 demos 为 65.4% vs. 62.6%，LIBERO 为 97.6% vs. 95.6%；CogACT/SimplerEnv 为 63.5% vs. 52.1%。历史 8 时，多帧延迟/额外显存为 193 ms/2023 MB，HAMLET 为 85.8 ms/578 MB。

**硬伤。** 它仍只保留固定 `T` 个时刻，不是 episode-wide 递归记忆；moment token 先做“同一时刻相似、异时刻分离”，可能过分强调变化而非任务因果性；在普通 benchmark 上增益小于真实手工记忆任务。

**对 VPWEM。** 必须有 `moment token + shallow Transformer` 的等 token、等历史、等 backbone 对照。VPWEM 的真正优势应出现在 `test horizon >> train horizon` 和关键事件早已超出 HAMLET 窗口时。

### 3.3 MemER：显式关键帧检索路线

**一句话。** Qwen2.5-VL 高层策略从最近 8 帧中选择关键帧并累计成视觉记忆，同时预测 subtask；低层 π0.5 执行动作。关键帧标签来自人工定义的每类 subtask 规则。

**关键证据。** 三个真实任务平均进度：无历史 26%、短历史 58%、32 帧长历史 62%、MemER 96%、人类高层 96%。32 帧已带来约 1 秒推理成本，仍比 MemER 低 34%。视觉记忆优于文本或视觉+文本；文本容易让模型依赖示范中的 canonical subtask order。

**硬伤。** 分层系统、7B VLM、人工关键帧规则和 subtask 标签较重；只测 3 个真实任务、20 trials；keyframe buffer 与任务段数一起增长，不是严格有界；高低层解耦使收益不能直接归因于记忆表示。

**对 VPWEM。** 加入均匀帧、相似度帧、change-point 帧和可学习关键帧四类基线。最有说服力的任务是“两个历史视觉上相似但任务变量不同”，证明 latent compression 不只是挑图。

### 3.4 RoboMME：最重要的新评测与负结果

**一句话。** 16 个任务覆盖 temporal/spatial/object/procedural memory，并在 π0.5 上统一比较 symbolic、perceptual、recurrent 三类表示和 context/modulator/expert 三种注入方式。

**关键证据。** 最强的 FrameSamp+Modulator 为 44.51%；MemER 为 42.38%；无记忆 π0.5 为 17.93%；循环变体约 18–22%，没有优势。真实机器人上 FrameSamp+Modulator 25/40，symbolic 19/40，π0.5 4/40。人类高层 + oracle control 仍只有 90.5%。视觉记忆约 6.27 TFLOPs，而 MemER 接近无记忆模型的 5 倍成本。

**最重要发现。** 计数/事件显著任务偏好符号记忆；运动、时间敏感、长视频推理偏好视觉记忆；动态场景变化偏好 MemER。没有一种表示通吃。MemoryVLA 和 DP 在该 benchmark 的补充表中总体都低于 10%，说明跨 benchmark 排名会反转。

**对 VPWEM。** RoboMME 应成为主实验或至少强补充。必须按任务类型报告，而非只给平均数；同时加入 FrameSamp+Modulator，因为它是“简单但强”的最低门槛。

### 3.5 MIKASA：证明任务真的需要记忆

**一句话。** 32 个桌面操作任务按 memory length、capacity、object、spatial、sequential 等维度组织；区分“长 horizon”和“当前帧不可判定”。

**关键证据。** 在 RememberColor3，OpenVLA-K8 为 0.59，但 Color5/9 降到 0.16/0.06；π0 在 5 个代表任务只有 0.15–0.42。作者发现较长 action chunk 能绕过部分记忆需求，但复杂度增加后失效；常见 offline RL 和序列模型在高 capacity/sequential memory 上也基本失败。

**硬伤。** 大量任务的视觉和动力学比真实世界简单；RL、VLA、控制接口不完全统一；成功率低时容易混入低层操作失败。

**对 VPWEM。** 每个任务都应标注“不可观测变量是什么、最小必要延迟是多少、随机猜测率是多少”。把 `RememberColor3/5/9` 做成 memory capacity scaling，而非只选一个容易任务。

### 3.6 HiMe：系统级分层记忆

**一句话。** 高频 Executor 执行，Sentry 用最近 8 帧判断 subtask 是否完成，低频 Planner 读取可增删改的图文长期记忆并制定策略。

**关键证据。** Object Search、Counting、Rearrangement 平均任务进度 90%；仅图像 80%，仅文本 89%，交错图文最好。FIFO-8 为 68%，只追加不管理为 86%，Add/Update/Delete 为 90%。相对 flat memory，Planner API calls 约降 3 倍，memory hit 从约 61–76% 提到 92–98%。

**硬伤。** “infinite memory” 会随任务增长，不满足 VPWEM 的固定容量目标；依赖 Planner API 和人工系统分层；Sentry recall 仅约 35%，依靠定时 Planner fallback 避免死循环。

**对 VPWEM。** HiMe 的真正启示不是复制三层架构，而是**记忆要能修正旧状态**。VPWEM 当前递归摘要主要做 consolidation，没有显式 contradiction/update/delete 机制。

### 3.7 Embodied-SlotSSM / LIBERO-Mem：对象级记忆

**一句话。** 用 slots 保持对象身份和局部时序状态，SSM 同时重建过去并预测未来 slot，再输出 16 个 relation tokens 条件化动作；提出重复、交换、多对象依赖的 LIBERO-Mem。

**关键证据。** LIBERO-Goal 上 SlotSSM 83.0%，长窗口 SlotVLA 75.5%，短窗口 32%。但更难的 LIBERO-Mem 上平均 subgoal completion 只有 14.8%，π0/SlotVLA 约 0–5%。

**硬伤。** 当前实现使用 oracle subgoal text embedding；仅仿真、固定场景；完整任务成功几乎为零。论文自己承认这只是弱 empirical baseline。

**对 VPWEM。** 它指出 Q-Former summary 的信息干扰问题：目标、物体身份、空间和阶段全塞进同一组 token。可以引入“对象 slots + 全局 residual tokens”，但不能依赖 oracle segmentation/subgoal。

### 3.8 TRM-VLA：记忆关键推理而非原始历史

**一句话。** Keyframe Temporal Reasoning 在关键帧产生短 CoT，Global Context Memorization 缓存过去关键推理特征，当前动作通过 cross-attention 读取并用 FiLM 调制 diffusion expert。

**关键证据。** SimplerEnv 72.9%，CogACT 51.3%；LIBERO-90 94.8%；真实机器人 69%，CogACT 50%。同时移除 KTR/GCM 后 SimplerEnv 约从 0.73 降到 0.54，真实场景从 0.69 降到 0.50。

**硬伤。** 需要 VLM 语义推理和关键帧机制，成本、标注和基础模型先验强；缓存规模与关键帧数相关；性能提升可能来自额外 CoT 监督而非 memory。

**对 VPWEM。** 给 memory token 加 task-state probe 是低成本高收益实验：物体是否移动、哪个 subgoal 完成、计数值、目标先前位置。若 probe 不可读，认知记忆叙事很难成立。

### 3.9 OptimusVLA：双记忆叙事竞争

**一句话。** Global Prior Memory 从训练示范检索任务级动作先验以初始化 flow sampling；Local Consistency Memory 维护已执行 action chunks，改善局部连续性。

**关键证据。** LIBERO 平均 98.6%，Long 96.4%；CALVIN 平均完成长度 4.45。移除两类 memory 后 LIBERO-Long 96.4→92.4，真实泛化 85→75；平均 NFE 3.2，并报告真实推理约比 π0.5 快 2.9 倍。

**硬伤。** GPM 是跨 episode 的训练轨迹库，LCM 主要记动作而非不可见环境状态；“dual memory”与 VPWEM 名称相似但功能边界不同。

**对 VPWEM。** Related Work 必须按**记忆的时间边界和对象**区分：跨轨迹动作先验、当前 episode 动作史、当前 episode 观测史。建议用 history/action 互换实验分别测依赖。

### 3.10 CycleManip：历史采样 + 进度监督

**一句话。** 针对敲击、摇晃、擦拭等当前帧高度相似的周期任务，用成本感知历史采样和周期/进度理解监督恢复动作方向与完成度。

**关键证据。** 仿真 Block Hammering 86%（最佳基线 23%），Bottle Shaking 95%（19%）；六类真实任务完整方法显著优于 DP3 和移除 task-understanding 模块的版本。

**硬伤。** 任务族高度特化，显式周期/进度监督占重要作用；不能说明一般遮挡、身份、空间记忆也能解决。

**对 VPWEM。** 加一个显式 progress/count head，检查简单充分统计量是否已经解释全部收益。若是，则不应把它包装成通用 episodic memory。

### 3.11 ALT：Diffusion Policy 可能只是动作查表

**一句话。** 将图像编码到对比空间，最近邻检索训练 trajectory ID 和 frame index，再直接取对应动作；作者认为 DP 在小数据机器人任务上本质是隐式 lookup table。

**关键证据。** 单个真实任务中 ALT 和 DP 都达到 100% ID 成功；ALT 模型 45.5 MB vs. DP 5.3 GB，推理 0.009 s vs. 2.65 s。作者报告 ALT 只需 DP 的 0.34% 推理时间、0.85% 内存。

**硬伤。** 结论来自小数据、少任务，最近邻规模扩展性未知，OOD threshold 高度敏感，不能外推到大规模 DP/VLA。

**对 VPWEM。** 这是必须做的反证：对 memory token 训练 trajectory-ID probe；报告最近邻相似度和成功率相关性；做 train-scene identity holdout；加入 ALT 或 observation-nearest action chunk 基线。

### 3.12 AFI：更强记忆也可能更难纠错

**一句话。** 检测 VLA 是否陷入重复训练轨迹，回滚到历史低代价位置，再以 3D affordance field 采样新 waypoint。

**关键证据。** 真实任务将 π0 的 Place Carrot 61→87%、Remove Lid 63→80%、Slot Pen 60→82%、Stack Tape 64→86%；LIBERO-Spatial 54.0→75.7%。

**对 VPWEM。** 插入外力扰动、错误动作、物体重定位后，比较有/无 episodic memory 的 recovery time。若 memory 让策略坚持旧轨迹，说明它记的是动作惯性而非可更新状态。

### 3.13 SemanticVLA：语义推理和动作记忆要拆开

**一句话。** 显式坐标轨迹与隐式 action tokens 协同，目标是避免“见过指令就复现动作”的 shortcut。

**关键证据。** LIBERO 平均 97.0%，Long 94.4%；真实机器人 62.3%，π0 为 48.0%；指令改写下报告性能下降 9.4%。

**对 VPWEM。** 联合做 instruction paraphrase、history shuffle、同当前帧异历史和同历史异当前帧四象限测试，才能区分语义理解、当前视觉、历史状态和训练轨迹索引。

## 4. 降级论文：仍可利用什么

### 4.1 2025 的四篇

**LRAM/xLSTM（P1）。** 在 432 tasks、6 domains、894M transitions 上比较 xLSTM、Mamba、Transformer，核心价值是 recurrent inference 的常数状态与长度外推。论文还发现把 past actions 放进输入会产生 copycat shortcut，去掉 actions 后长历史反而更有用。可作为 VPWEM 的高效 recurrent baseline 和“禁止动作泄漏”的依据，但它并未证明长期不可观测状态恢复。

**SAM2Act+（P1）。** 把 SAM2 的视觉 memory bank 用于机器人空间记忆，MemoryBench 三任务平均 94.3%，比下一基线高 39.3 点；真实任务 75% vs. RVT-2 43%。强项是对象重现和空间记忆，弱点是 keyframe action + 外部 motion planner，与连续闭环 diffusion policy 不同。适合做显式对象 memory 上界。

**3D-Mem（P1）。** 将探索过程压成多视角 3D memory snapshots，用 VLM 选 frontier，服务 EQA/GOAT navigation。它说明“保存结构化视角快照”比 caption 或全帧更有效，但任务输出是问答/导航，不是低层操作。

**Spatial Feature Memory（P2）。** 把 2D detector 的中间特征投影到 TSDF，之后从地图读回以增强低质量视角下的 embodied detection。可借鉴 read/write 几何对齐，但没有动作记忆，也假定静态场景。

### 4.2 2026 的邻近工作

**Personalization Memory（P1）。** MEMENTO 研究 object semantics 和 user pattern 的跨情景记忆。最有用的负结果是 top-k 增大反而引入噪声，多个相关记忆会产生 coordination failure；GPT-4o 在组合记忆条件下成功率可下降 30.5%。这支持“memory quality > memory quantity”和干扰消融。

**SOMA（P1，当前仅官方摘要）。** 扫描多视角形成全局 spatial-semantic memory，并动态更新、按指令检索，解决 out-of-vision manipulation。它是空间地图路线，不是通用情景压缩；应在正文公开后补主表和成本核验。

**RGMP（P1）。** recurrent geometric-prior multimodal policy 在人形操作泛化中报告 87% unseen-object success，重点是几何引导和数据效率，recurrent 不是为 episode-wide memory 设计。

**Event Memory（P2）。** 异步事件逐个更新固定状态，用于高频 UAV 控制，报告约 30% latency reduction。可借鉴“只在信息变化时更新”，但模态、任务和时间尺度均不同。

**Conflict-Aware Memory（P2，当前仅官方摘要）。** 用规则检测向量库中的语义冲突，LLM planner 的语法/解释准确率平均提高 15.25/14.25 点。对 VPWEM 的启发仅是 stale/conflicting memory 管理，不能作为低层策略基线。

### 4.3 2024 的四篇为何不应占精读资源

**Embodied Active Defense。** recurrent feedback 用于移动机器人主动改变视角，以减弱 adversarial patch；历史只服务鲁棒感知闭环，不是 episodic task state。

**READ-PVLA。** 全称中的 PVLA 是 Partial Video-Language Alignment；研究低资源视频语言模型的参数高效适配，不输出机器人动作。

**Offline Risk-sensitive RL with Partial Observability。** 从有限 HRI 数据学习 POMDP 并按模型不确定性选风险敏感任务分配策略，26 名参与者验证；和视觉操作记忆的共同点只有 POMDP 形式化。

**Hospital Memory Clinic Robot。** 多人多模态对话系统部署在医院 memory clinic；“memory”是临床场景名。应从索引删除或永久标记 false positive。

## 5. 竞争格局：到底在记什么

| 路线 | 代表 | 保存内容 | 容量 | 最大优势 | 主要漏洞 |
|---|---|---|---|---|---|
| 递归潜在摘要 | VPWEM | 越窗视觉 observation tokens | 固定 | 每步成本有界、端到端 | 摘要漂移、不可解释、可能索引轨迹 |
| 感知/认知条目库 | MemoryVLA | 每时刻 256 感知 + 1 认知 token | 固定 `L`，但较大 | 双流信息丰富 | `L` 需按任务调，合并稀释事件 |
| 时刻 token 窗口 | HAMLET | 最近 `T` 个 moment tokens | 固定窗口 | 简单、高效、易接 VLA | 超窗即忘 |
| 显式关键帧 | MemER | 任务相关历史图像 | 随 subtask 增长 | 可解释、真实长时任务强 | 需标签/高层 VLM、非严格有界 |
| 均匀/稀疏视觉历史 | RoboMME FrameSamp | 采样帧 tokens | 固定预算 | 简单且总体最强 | 任务无关，可能漏掉稀有事件 |
| 符号/场景图 | HiMe、SOMA、SlotSSM | 物体、位置、计数、subgoal | 结构相关 | 可更新、可解释 | 感知/grounding 误差、依赖先验 |
| 动作记忆 | OptimusVLA、ALT | 示范动作先验/trajectory ID | 库大小相关 | 快、数据内强 | OOD memory trap |
| 推理轨迹 | TRM-VLA | 关键 CoT/视觉推理特征 | 关键帧相关 | 语义强 | 成本和监督重 |

## 6. 从缺口反推的 Idea Backlog

### Idea 1（最高优先）：Causal Memory Stress Test，不先改模型

**问题。** 当前 VPWEM 的成功率不能证明 episodic token 被因果使用，更不能证明其中是任务状态而不是 trajectory ID。

**实验包。** 对每个 evaluation state 构造：正确历史、清空记忆、时间打乱、同任务另一轨迹记忆、同当前帧但不同 latent state 的反事实记忆、只保留关键事件、只保留最近窗口。再报告动作分布变化、成功率、恢复时间和 memory attention。

**新增指标。** Memory Necessity（正确→清空下降）、Memory Specificity（正确→错误历史下降）、State Sufficiency（正确 memory + 模糊当前帧能否保持）、Trajectory Leakage（trajectory-ID probe accuracy）。

**价值。** 2–4 天能完成第一版，却能同时回答 ALT、AFI、SemanticVLA 和 reviewer 对机制解释的质疑。即使不成为新方法，也会显著抬高 VPWEM 论文证据等级。

### Idea 2（最高优先）：同预算强基线 + 长度外推

**问题。** 现在最危险的替代解释是“任意历史 token 都有效”或“只是多了参数”。

**基线。** HAMLET-style moment tokens、FrameSamp+Modulator、uniform frames、change-point/keyframe、xLSTM state、MemoryVLA-style adjacent merge。统一 token 数、backbone 参数和训练步数。

**外推设计。** train horizon 为 32/64，test horizon 为 1×/2×/4×/8×；关键 cue 固定在 episode 前 10%，让最近窗口必然看不到。横轴同时画 success、latency、peak memory、FLOPs。

**价值。** 这会把 VPWEM 的定位从“又一个 memory module”收敛为“在固定预算下对任意长 episode 的状态保真”。

### Idea 3：Event-Gated Write，避免每步递归污染

**问题。** 当前越窗 observation 进入 compressor 的节奏主要由窗口移动决定，不由信息价值决定。静态/重复帧反复写入会造成 summary drift，稀有事件则可能被平均掉。

**方法。** 学一个轻量 write gate，输入当前 working-memory、episodic tokens、动作变化和 prediction error，输出 `skip / update / allocate`。训练信号可来自 action loss + 稀疏写入正则 + memory reconstruction/probe loss。

**关键消融。** 固定周期、RGB difference、feature surprise、learned gate；相同写次数下比较。特别测计数、物体被遮挡前状态和长静止区间后的稀有事件。

**来源。** RoboMME 的 event-salient/task-dependent 结论、MemER 的关键帧、Event Memory 的信息变化更新、MemoryVLA 的相邻合并共同指向这一缺口。

### Idea 4：Revisable Episodic Memory，允许纠正旧状态

**问题。** 递归压缩擅长“保留”，不擅长“先前事实已失效”。对象移动、任务回滚或外力扰动后，旧记忆可能继续控制动作。

**方法。** 将 episodic tokens 分成 `entity/state slots + residual slots`；新 observation 先检索相关 slot，再以 contradiction gate 执行 keep/update/reset。无需文本规则，可用 next-observation consistency 或前后对象匹配训练。

**评测。** 中途移动已见物体、撤销已完成 subgoal、交换同类物体、插入错误动作；报告 stale-state recovery steps。

**来源。** HiMe 的 Add/Update/Delete 比 append-only 高 4 点，AFI 的 memory trap，以及 Personalization/Conflict-Aware Memory 的干扰结论。

### Idea 5：Type-Routed Memory Tokens，而非一锅 Q-Former

**问题。** RoboMME 表明 symbolic/perceptual memory 各有所长；SlotSSM 表明对象身份需要独立持续状态。单一 token 池可能发生计数、空间、阶段和视觉细节的表征冲突。

**方法。** 保持总 token budget 不变，划分或动态路由为 `event/count`、`object/spatial`、`motion/procedural`、`residual visual` 四类 slots；当前 query 决定读写配额。不要引入 LLM 文本 memory，以免重演 MemER 文本过度依赖。

**最低可行版本。** 先只做 2 路：object slots + free latent slots；用 slot diversity、temporal consistency 和 action loss 联训。

**风险。** 容易把系统做复杂；必须坚持固定总预算，并证明不是参数量收益。

### Idea 6：Anti-Lookup Episodic Representation

**问题。** memory token 越能定位训练轨迹，ID 成功率可能越高，但 OOD 越脆。

**方法。** 主任务之外加两个头：任务 latent-state probe 要预测成功；trajectory-ID adversary 通过 gradient reversal 要预测失败。再加 history mixup：保持 latent state 一致时跨轨迹交换历史片段。

**评测。** 同义指令、物体纹理/位置变化、轨迹速度变化、不同示范者、错误动作恢复。看 ID probe 降低是否伴随 OOD 成功率上升。

**价值。** 这是最能把 ALT 反例转成新贡献的方法 idea，但需要确保 latent-state 标签在 MIKASA/RoboMME 可获得。

## 7. 推荐执行顺序

### 第一阶段：一周内，先把论文站稳

1. 实现 Idea 1 的 memory intervention harness。
2. 加 ALT/nearest-action 和 FrameSamp 两个最低成本强基线。
3. 在 RememberColor3/5/9、ShellGame、RoboMME 至少一项 counting 和一项 motion task 上跑 train/test horizon scaling。
4. 输出 success–latency–memory 三轴表，不再只报成功率。

### 第二阶段：决定是否形成新方法

优先实现 **Event-Gated Write**。它最贴合现有 VPWEM compressor，改动小于对象 slots 或分层 Planner，又能直接形成新的算法贡献和写入率/保真度实验。

如果实验发现主要失败来自环境状态被更新后记忆仍旧：转向 **Revisable Episodic Memory**。

如果 RoboMME 分任务结果呈明显互补：再做 **Type-Routed Memory Tokens**，不要一开始同时堆三种机制。

### 第三阶段：论文叙事

建议主张收敛为：

> VPWEM studies bounded within-episode observation memory for low-level visuomotor control. Unlike finite history windows, growing keyframe banks, symbolic planners, or cross-trajectory action retrieval, it recursively compresses arbitrarily long observation histories into a fixed token budget. We verify not only task success but also causal memory use, horizon extrapolation, and resistance to trajectory lookup shortcuts.

对应中文：VPWEM 研究低层视觉运动控制中的**有界单回合观测记忆**。它不同于有限历史窗口、随任务增长的关键帧库、符号规划器和跨轨迹动作检索，将任意长观测史递归压缩到固定 token 预算；并通过因果干预、长度外推和反查表测试验证记忆确实保存了任务状态。

## 8. 最终优先级

| 优先级 | 动作 | 新颖性收益 | 实现成本 | 失败风险 |
|---|---|---:|---:|---:|
| S | Causal Memory Stress Test | 高 | 低 | 低 |
| S | HAMLET/FrameSamp/ALT 同预算基线 | 高 | 中 | 低 |
| S | Horizon extrapolation + 成本曲线 | 高 | 低 | 低 |
| A | Event-Gated Write | 高 | 中 | 中 |
| A | Revisable Episodic Memory | 高 | 中高 | 中 |
| A | Anti-Lookup adversarial representation | 高 | 中 | 中高 |
| B | Type-Routed/Object slots | 很高 | 高 | 高 |
| C | 再加一个 LLM Planner 或文本 memory | 低 | 很高 | 高 |

最务实的主线是：**先证明 VPWEM 的 memory 真在记任务状态，再优化写入；不要先扩大模型。**

## 9. 本地材料与来源

- 本报告对应 PDF：`literature/vpwem/vpwem_p0_2024_2026/pdfs/`
- PDF 抽取文本：`literature/vpwem/vpwem_p0_2024_2026/text/`
- CVPR 2026 全文核验与图表：`literature/vpwem/cvpr2026_vpwem_related/report.md`
- 三年原始分级索引：`literature/vpwem/vpwem_2024_index/`、`vpwem_2025_index/`、`vpwem_2026_index/`

主要官方入口：

- MemoryVLA: https://arxiv.org/abs/2508.19236
- HAMLET: https://arxiv.org/abs/2510.00695
- MemER: https://arxiv.org/abs/2510.20328
- RoboMME: https://arxiv.org/abs/2603.04639
- HiMe: https://arxiv.org/abs/2607.03449
- MIKASA: https://arxiv.org/abs/2502.10550
- ALT: https://arxiv.org/abs/2505.05787
- Embodied-SlotSSM: https://arxiv.org/abs/2511.11478
- SAM2Act: https://arxiv.org/abs/2501.18564
- LRAM: https://arxiv.org/abs/2410.22391

## 10. 数据质量备注

- Connected Researchers 用于候选发现，不作为技术事实的最终来源；关键数字均优先核对论文正文。
- ICML 2026 虚拟页当时未给 HiMe、RoboMME、SOMA 的论文按钮；HiMe/RoboMME 已从 arXiv 正文核验，SOMA 仅按官方摘要判断。
- ACL 2026 Conflict-Aware Memory 当前无 `pdf_url`，只使用其原始英文摘要和公开数值，不把它计为全文精读。
- 2024 的 P0 误检说明原索引的关键词规则需要加入负面规则：`memory clinic`、`video-language alignment`、`adversarial patch defense` 不应自动进入机器人策略记忆 P0。
