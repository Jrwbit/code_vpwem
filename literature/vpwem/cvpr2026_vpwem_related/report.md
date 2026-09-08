# CVPR 2026 × VPWEM：记忆、历史与长时程机器人策略论文雷达

> 扫描日期：2026-07-21  
> 检索入口：[Connected Researchers · CVPR 2026](https://connectedresearchers.top/guide/cvpr/2026)  
> 研究锚点：VPWEM——面向非马尔可夫视觉运动策略的、固定容量的单回合历史压缩与工作/情景记忆。

## 一页结论

这不是一份按关键词堆出来的论文清单，而是一份以 VPWEM 的研究主张为坐标系、逐篇回到官方 PDF 核验后的竞争格局报告。

本轮完整读取了 Connected Researchers 的 CVPR 2026 目录：**4,071 条论文记录 → 99 篇宽召回 → 24 篇标题/摘要高相关 → 8 篇原文精读**。最终得到四篇必须优先处理的直接相关工作，以及四篇用于补实验边界和反例的工作。

![CVPR 2026 × VPWEM 相关论文定位图](assets/relevance_map.png)

最重要的判断有四点：

1. **“机器人策略中的记忆研究很少”已经不是安全表述。** CVPR 2026 同时出现了双记忆 VLA、关键推理轨迹记忆、动态场景图记忆和运动历史压缩。VPWEM 应把贡献收窄并说准：**将单回合、长时视觉观测历史压缩为固定数量的潜在情景 token，为视觉运动策略恢复被当前帧遮蔽的任务状态。**
2. **最直接的竞争者是 OptimusVLA。** 它也使用“双记忆”叙事，但其 Global Prior Memory 是跨示范轨迹检索，Local Consistency Memory 主要建模已执行动作块；VPWEM 的核心则应明确为单回合观测史的在线压缩，二者保存的信息和使用边界不同。
3. **HiF-VLA 是最有价值的轻量基线。** 它证明把历史压缩成运动向量，比直接堆叠历史图像更有效且更省显存。VPWEM 必须回答：当外观、物体身份、遮挡前状态等非运动信息决定动作时，学习到的情景 token 是否优于运动流摘要。
4. **AFI 与 SemanticVLA 提供了必要反例。** 长历史并不天然等于更强泛化；策略可能只是更牢固地复现训练轨迹。因此实验不能只测 in-distribution 成功率，还要加入指令改写、物体位置变化、错误动作回滚、历史置换等诊断。

## 筛选流程与可靠性

### 检索漏斗

| 阶段 | 数量 | 做法 |
|---|---:|---|
| CVPR 2026 全量目录 | 4,071 | 遍历 Connected Researchers 的 CVPR 2026 全部分页并去重 |
| 宽召回 | 99 | 机器人/VLA/模仿学习，与 memory、history、temporal、partial observability、long-horizon、non-Markov 等交叉 |
| 高相关候选 | 24 | 人工复核标题与摘要，排除纯视频记忆、导航专用、仅名称含 memory 的工作 |
| 原文精读 | 8 | 下载 CVF 或 arXiv 官方 PDF，核对方法图、主表、消融和运行代价 |

### 数据质量说明

Connected Researchers 只用于**发现和建立候选池**。核对中发现：

- Scene Graph 论文在聚合页中题名为 “Scaling Spatial and Temporal Context…”，官方预印本题名为 **“Expanding Spatial and Temporal Context for Robotic Imitation Learning With Scene Graphs”**；
- PALM 的聚合摘要出现了与正文无关的混杂内容；
- 个别自动生成的结果描述与最终表格不完全一致。

因此，本文的技术描述、页码和数字均来自官方 PDF，不转述聚合页的自动摘要。

## 论文优先级总览

| 优先级 | 论文 | 保存的“过去” | 使用方式 | 与 VPWEM 的关系 |
|---|---|---|---|---|
| P0 | OptimusVLA | 跨轨迹动作先验 + 近期动作块 | 检索初始化流采样；局部一致性调节 | 双记忆叙事最直接竞争，但不是单回合观测记忆 |
| P0 | TRM-VLA | 关键帧推理特征 | 动态缓存、跨注意力检索、FiLM 调制 | 记的是推理轨迹，VPWEM 记的是环境/交互状态 |
| P0 | Scene Graph Memory | 任务相关对象、位置、可见性 | 动态场景图条件化扩散策略 | 显式、可解释的情景记忆替代方案 |
| P0 | HiF-VLA | 历史运动向量 | 用紧凑 motion stream 代替历史帧 | 最关键的轻量历史压缩基线 |
| P1 | CycleManip | 成本感知抽样的历史帧 | 周期/进度监督 | 同样针对非马尔可夫操作，但更任务特化 |
| P1 | PALM | 预测的任务进度与 affordance | 进度查询调制动作生成 | 说明部分长时程问题可由进度建模解决 |
| P1 | AFI | 历史可回滚位置 | 检测 memory trap 后回滚并重规划 | 反例：动作记忆可能导致 OOD 失败 |
| P1 | SemanticVLA | 显式空间轨迹 + 潜在动作 token | 语义推理与动作生成协同 | 反例：需要区分推理能力与动作记忆 |

![七篇论文的代表性结果快照](assets/results_snapshot.png)

> 图中每张卡片只在同一论文、同一任务内比较作者方法与相应基线，用来感知证据强弱；不同任务之间不能直接排名。Scene Graph 工作主要以曲线图报告多阶段任务结果，因此没有硬塞进数值卡片。

---

## P0-1：OptimusVLA——最直接的“双记忆”竞争者

**论文：** [Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Global_Prior_Meets_Local_Consistency_Dual-Memory_Augmented_Vision-Language-Action_Model_for_CVPR_2026_paper.html)

### 30 秒判断

必须作为 Related Work 与实验讨论中的第一优先级。它和 VPWEM 都把“记忆”作为 VLA 的核心机制，但两者的记忆对象不同：OptimusVLA 更像**跨轨迹动作先验 + 近期动作一致性**，VPWEM 应坚持并验证**单回合观测历史中的潜在任务状态**。

![OptimusVLA 方法图（官方论文 Figure 2）](assets/optimusvla_method.png)

### 它具体做了什么

1. **Global Prior Memory（GPM）**从训练示范中建立记忆库；面对当前任务时，检索任务级动作先验，用它初始化 flow-based action sampling。
2. **Local Consistency Memory（LCM）**维护已经执行的动作块，通过一致性层和类似 Mamba 的动态状态建模，感知局部执行进度并修正后续动作。
3. 两类记忆分别处理“全局动作方向”和“局部动作连续性”，目标还包括降低采样步数和推理延迟。

### 论文证据

- LIBERO 平均成功率 **98.6%**，LIBERO-Long **96.4%**；同表 OpenVLA-OFT 在 Long 上为 **94.5%**（PDF p.5, Table 1）。
- CALVIN 平均完成长度 **4.45**；π0.5 为 **4.26**（PDF p.5）。
- 同时移除 GPM 与 LCM 后，LIBERO-Long 从 **96.4% 降至 92.4%**，CALVIN 从 **4.45 降至 4.26**，真实场景泛化成功率从 **85% 降至 75%**（PDF p.6, Table 4）。
- 平均 NFE 为 **3.2**，论文报告相对 π0.5 的 10 次采样明显更低；真实机器人推理速度约提升 **2.9×**（PDF p.7, Figure 5 与正文）。

### 对 VPWEM 的压力与机会

- 不要只用 “dual memory” 做核心新颖性表述，这一标签已被占据。
- 必须画清数据边界：**GPM 跨 episode 读取训练轨迹；VPWEM memory 在当前 episode 内由观测在线更新。**
- 建议增加“跨回合示范被替换，但当前回合历史不变”与“当前回合历史被置换，但检索库不变”两组干预，证明策略到底依赖哪一种记忆。
- 如果 VPWEM 不需要外部轨迹库，应把这是其部署优势说清楚：固定容量、在线更新、无训练示范检索延迟。

---

## P0-2：TRM-VLA——把关键推理过程存进动态记忆

**论文：** [TRM-VLA: Temporal-Aware Chain-of-Thought Reasoning and Memorization for Vision-Language-Action Models](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TRM-VLA_Temporal-Aware_Chain-of-Thought_Reasoning_and_Memorization_for_Vision-Language-Action_Models_CVPR_2026_paper.html)

### 30 秒判断

TRM-VLA 与 VPWEM 都强调不能只看当前帧，但它保存的是**关键时刻的语言/视觉推理特征**。这是一个更语义化、更稀疏的记忆路线，也提醒 VPWEM 需要解释：潜在 token 中到底保留了任务状态、物体状态还是动作过程。

![TRM-VLA 方法图（官方论文 Figure 2）](assets/trm_vla_method.png)

### 它具体做了什么

1. **Keyframe Temporal Reasoning（KTR）**只在关键帧生成分层 CoT，避免每个时间步都输出长推理链。
2. **Global Context Memorization（GCM）**把过去关键推理特征放入动态缓存。
3. 当前决策通过跨注意力检索缓存，再用 FiLM 调制 diffusion action expert。

### 论文证据

- SimplerEnv 平均成功率 **72.9%**，CogACT 为 **51.3%**，提升 **21.6 个百分点**；π0 为 69.2%（PDF p.7, Table 1）。
- LIBERO-90 成功率 **94.8%**，CogACT-ECoT 为 92.1%（PDF p.7, Table 2）。
- 真实机器人平均成功率 **69%**，CogACT 为 50%；平均生成 token 数 **4.3**，ECoT 为 26.8（PDF p.7–8, Table 3）。
- 同时去掉 KTR 与 GCM，SimplerEnv 从 **0.73 降到 0.54**，真实场景从 **0.69 降到 0.50**（PDF p.8, Table 4–5）。

### 对 VPWEM 的压力与机会

- 加入一个“关键帧稀疏记忆”或 TRM 风格的语义摘要基线，避免只和无记忆 policy 比。
- 对 memory token 做可诊断实验：线性探测物体是否移动、子任务是否完成、当前阶段等变量。
- 如果 VPWEM 无需 CoT 标注或显式关键帧标签，应将“端到端、连续更新”作为与 TRM-VLA 的核心差异，而不是笼统声称更通用。

---

## P0-3：Scene Graph Memory——显式、可解释的情景记忆

**论文：** [Expanding Spatial and Temporal Context for Robotic Imitation Learning With Scene Graphs](https://arxiv.org/abs/2606.01072)

### 30 秒判断

这是 VPWEM 最应该正面对照的**显式情景记忆**方案。它把任务相关物体、二维框、三维位置和可见性组织成动态图；物体离开视野后仍保留其空间状态。其可解释性强，但依赖检测、跟踪与结构化场景。

![Scene Graph Memory 方法概览（官方论文 Figure 1）](assets/scene_graph_method.png)

### 它具体做了什么

1. LLM 在每个任务开始时枚举任务相关对象。
2. GroundingDINO 在 episode 开始时检测对象，XMem 在后续帧中跟踪。
3. 每个图节点保存 DINOv2 外观特征、可见时的 2D 边界框和世界坐标系中的 3D 中心；物体离开视野后仍保留最后的 3D 位置。
4. 场景图与当前视觉输入共同条件化 self-attention 和 diffusion policy。

![Scene Graph Memory 真实机器人多阶段结果（官方论文 Figure 6）](assets/scene_graph_results.png)

### 论文证据

- 模拟和真实机器人曲线均显示：在物体离开视野、后续子任务依赖先前状态时，完整 scene-graph memory 优于仅当前 RGB、仅可见对象图和 point-to-point 等对照（PDF p.6, Figure 2；p.8, Figure 6）。
- 系统在 NVIDIA 3090 上运行约 **3.5–3.8 Hz**，即每步约 260–285 ms；XMem 跟踪约 0.19 s/帧。LLM 每任务调用一次，GroundingDINO 每 episode 调用一次（PDF p.8）。

### 对 VPWEM 的压力与机会

- 这篇工作可以作为“显式结构化记忆”基线；VPWEM 则是“隐式、固定 token、无需对象表”的路线。
- 建议设计两类场景：一类物体清晰、结构化，scene graph 占优；另一类状态由接触、形变、容器内部或微小视觉变化决定，检验 VPWEM 是否能保存图结构难表达的信息。
- 报告模型延迟随 episode 长度变化。VPWEM 的固定 token 应体现为近似恒定的推理成本。

---

## P0-4：HiF-VLA——把历史帧压缩成运动流

**论文：** [HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_HiF-VLA_Hindsight_Insight_and_Foresight_through_Motion_Representation_for_Vision-Language-Action_CVPR_2026_paper.html)

### 30 秒判断

这是最值得复现的轻量历史基线。论文显示直接堆叠多帧不但昂贵，还可能降低成功率；将过去压缩为 motion vector stream 才得到增益。VPWEM 若主张“学到的压缩记忆更强”，就必须在相同历史窗口和算力预算下超过它。

![HiF-VLA 方法图（官方论文 Figure 2）](assets/hif_vla_method.png)

### 它具体做了什么

1. 将稠密历史帧编码为紧凑的运动向量序列，作为 **hindsight**。
2. VLM 同时预测未来运动表示和动作 token，形成 **foresight**。
3. 联合 action expert 融合 hindsight、当前观测和 foresight 生成动作。

### 论文证据

- 多视角 LIBERO-Long 成功率 **96.4%**，OpenVLA-OFT 为 94.0%；第三视角设置为 **94.4% vs. 91.0%**（PDF p.6, Table 1）。
- CALVIN 平均完成长度 **4.35**，OpenVLA-OFT 为 4.10（PDF p.6, Table 2）。
- 直接加入历史帧：显存从 **30.8 GB 增至 63.6 GB**，时延从 **72.9 ms 增至 229.5 ms**，成功率从 91.0% 降到 90.4%；只用 hindsight motion 时为 31.4 GB、117.7 ms、92.2%（PDF p.7, Table 3）。

### 对 VPWEM 的压力与机会

- 必加 motion-history baseline，比较成功率、显存、每步延迟和随历史长度的扩展性。
- 单独构造“运动相同但语义状态不同”的任务，例如相同轨迹经过不同物体、容器状态或先前操作结果，验证 VPWEM 保存了超出 optical flow 的信息。
- 报告 memory token 数量与性能的 Pareto 曲线，而不是只给一个配置。

---

## P1-1：CycleManip——成本感知历史采样解决周期性非马尔可夫任务

**论文：** [CycleManip: Enabling Cycle-based Manipulation via Effective History Perception and Understanding](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_CycleManip_Enabling_Cycle-based_Manipulation_via_Effective_History_Perception_and_Understanding_CVPR_2026_paper.html)

![CycleManip 方法图（官方论文 Figure 2）](assets/cyclemanip_method.png)

CycleManip 明确把反复敲击、摇晃、擦拭等任务表述为非马尔可夫问题：当前画面相似，但动作方向或完成度取决于过去。它使用成本感知历史采样，并增加任务进度/周期理解监督。

- Block Hammering 成功率 **86%**，最佳基线 DP3 为 23%；Bottle Shaking 为 **95% vs. 19%**（PDF p.6, Table 1）。
- 真实机器人中，完整方法在六类任务分别达到 93.75%、68.75%、90%、100%、50%、75%，显著高于 DP3 和移除任务理解模块的版本（PDF p.7, Table 2）。

**给 VPWEM 的启示：** 同样的非马尔可夫论点已经有强结果，但 CycleManip 偏向周期任务和显式进度监督。VPWEM 应在非周期、遮挡、状态不可逆变化等任务上证明更通用，并与“历史采样 + 进度头”对照。

---

## P1-2：PALM——有时需要的不是记忆，而是任务进度

**论文：** [PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_PALM_Progress-Aware_Policy_Learning_via_Affordance_Reasoning_for_Long-Horizon_Robotic_CVPR_2026_paper.html)

![PALM 方法图（官方论文 Figure 2）](assets/palm_method.png)

PALM 预测 global/local/spatial/dynamic 等细粒度 affordance foresight，并让 progress query 与 diffusion transformer 共同预测动作和任务进度。它不是持久记忆模块，却能解决大量长时程任务。

- CALVIN 平均完成长度 **4.48**，五任务链成功率 82%；Seer 为 3.98/64.3%，π0 为 3.92/59.9%（PDF p.6, Table 1）。
- LIBERO 平均成功率 **94.5%**，LIBERO-Long **91.8%**；CoT-VLA 的 Long 为 69.0%（PDF p.7, Table 2）。

**给 VPWEM 的启示：** 需要把“记住过去”与“知道当前进度”拆开。增加从 VPWEM token 预测任务进度的 probe，并与显式 progress head 比较；若简单进度监督已经解决任务，就不能把收益全部归因于情景记忆。

---

## P1-3：AFI——动作记忆也可能把策略困住

**论文：** [Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulation](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Affordance_Field_Intervention_Enabling_VLAs_to_Escape_Memory_Traps_in_CVPR_2026_paper.html)

![AFI 方法图（官方论文 Figure 3）](assets/afi_method.png)

AFI 将 memory trap 定义为：分布变化后，VLA 仍反复执行训练中记住的轨迹。系统根据末端位移和目标距离检测陷阱，回滚到历史上代价最低的位置，再以 Spatial Affordance Field 采样 waypoint 并选择轨迹。

- 真实机器人 Place Carrot：π0 从 **61% 提升到 87%**；Remove Lid 63%→80%，Slot Pen 60%→82%，Stack Tape 64%→86%（PDF p.6, Table 1）。
- LIBERO-Spatial 从 **54.0% 提升到 75.7%**，LIBERO-Object 从 56.4% 提升到 73.2%（PDF p.7, Table 2）。

**给 VPWEM 的启示：** 更强的历史依赖可能放大轨迹复现偏差。应增加物体初始位置变化、错误动作插入、已完成子任务被外力改变等 OOD 测试，并检查 memory token 是否让策略更难脱离旧轨迹。

---

## P1-4：SemanticVLA——区分语义推理和动作记忆

**论文：** [SemanticVLA: Towards Semantic Reasoning over Action Memorization via Synergistic Explicit-Implicit Visual Representations](https://openaccess.thecvf.com/content/CVPR2026/html/Ni_SemanticVLA_Towards_Semantic_Reasoning_over_Action_Memorization_via_Synergistic_Explicit_CVPR_2026_paper.html)

![SemanticVLA 方法图（官方论文 Figure 1）](assets/semanticvla_method.png)

SemanticVLA 将显式坐标轨迹与隐式动作 token 协同建模，目标是减少“见过指令→复现动作”的捷径，提升语义与空间推理。

- LIBERO 平均成功率 **97.0%**，LIBERO-Long **94.4%**；UniVLA 的 Long 为 92.0%（PDF p.6, Table 1）。
- 真实机器人平均成功率 **62.3%**，π0 为 48.0%，MolmoAct 为 40.3%（PDF p.7, Table 2）。
- 对指令进行改写时，论文报告 LIBERO 性能下降仅 **9.4%**（PDF p.7, Figure 5 与正文）。

**给 VPWEM 的启示：** 增加指令同义改写、语序变化和未见表达；同时做 history shuffle。这样可以区分模型是在理解任务并利用历史，还是把语言和历史一起当作训练轨迹索引。

---

## 对 VPWEM 的具体修改建议

### 1. 重新写核心主张

不再写：

> Existing visuomotor policies rarely incorporate memory / memory remains largely unexplored.

建议写成：

> Recent policies incorporate action priors, reasoning traces, motion histories, or explicit object graphs. VPWEM instead targets **bounded, within-episode observation memory**: it continuously compresses arbitrarily long interaction history into a fixed set of latent episodic tokens, preserving task-relevant state that is absent from the current observation without relying on an external trajectory bank or hand-constructed scene graph.

中文含义：已有工作分别记动作先验、推理轨迹、运动历史或对象图；VPWEM 关注的是**有界的单回合观测记忆**，把任意长的交互史持续压缩成固定潜在 token，保留当前观测中缺失的任务相关状态，且不依赖外部示范库或手工场景结构。

### 2. 必须补的基线

1. **No-history / frame stacking**：确认增益确实来自历史。
2. **HiF 风格 motion history**：检验是否只需运动摘要。
3. **Scene graph / object-centric memory**：检验结构化显式记忆是否足够。
4. **Progress-aware head**：排除“只是学会了任务进度”的解释。
5. **Recurrent/action-history memory**：与 OptimusVLA 的局部动作记忆、TRM 的关键帧记忆形成概念对照。

### 3. 最有信息量的诊断实验

| 实验 | 要回答的问题 | 预期观测 |
|---|---|---|
| 历史顺序打乱 | 模型是否真的利用时序 | VPWEM 明显下降，当前帧基线近似不变 |
| 跨 episode 历史替换 | 是否把历史当任务/轨迹索引 | 替换后不应盲目复现另一条轨迹 |
| 相同运动、不同语义状态 | 是否超越 motion-only 摘要 | VPWEM 优于 HiF 风格基线 |
| 物体离开视野再使用 | 是否保留被遮挡状态 | 与 Scene Graph 对照，按场景结构分析 |
| 指令同义改写 | 是否依赖语言记忆捷径 | 相对退化受控，并优于纯动作记忆路线 |
| 错误动作/外力干预 | 是否陷入 memory trap | 能覆盖旧记忆并恢复，而非重复轨迹 |
| episode 长度扩展 | 是否真正有界 | token 数、显存、延迟基本恒定 |
| memory token probe | token 中保存了什么 | 可读出阶段、对象状态、完成事件，但不过拟合动作 ID |

### 4. 推荐的结果组织方式

- 主表按 **短任务 / 长任务 / 部分可观测 / OOD 干预** 四组报告，不把“长任务”自动等同于“需要记忆”。
- 每个主结果同时给成功率、每步延迟、峰值显存、memory 容量。
- 画性能—容量 Pareto：token 数为 0/2/4/8/16，并加入 motion vector 与 scene graph 基线。
- 做三类可视化：memory attention 随时间、关键状态被写入/覆盖的时刻、失败案例中的错误记忆传播。

## 未列入精读主表但值得留档的候选

- **Explore with Long-term Memory**：关注带长期记忆的探索/导航，更接近 embodied navigation 与 MLLM-RL，任务形态与操作策略有距离。
- **AstraNav-Memory、LASAR、WeaveTime**：涉及长期记忆或时间建模，但主要面向导航、空间推理或通用视频，不宜直接充当 manipulation policy 基线。
- 大量题名含 “long-horizon” 的操作论文只做层次规划、技能分解或数据生成，没有显式读取历史；它们可放长任务背景，不应混进记忆方法对比。

## 最终结论

CVPR 2026 对 VPWEM 不是“没有相关工作”，而是已经形成四条清楚的邻近路线：

1. **动作先验/动作块记忆**——OptimusVLA；
2. **关键推理轨迹记忆**——TRM-VLA；
3. **显式对象图记忆**——Scene Graph Memory；
4. **运动历史压缩**——HiF-VLA。

VPWEM 最有希望站住的位置，是这些路线没有同时覆盖的交叉点：**单回合、观测驱动、固定容量、端到端学习、能保存运动之外的隐含任务状态。** 接下来工作的重点不应是继续扩大论文数量，而是用上面的基线和干预实验，把这个位置证实。

---

## 审计记录

- 检索入口：[Connected Researchers · CVPR 2026](https://connectedresearchers.top/guide/cvpr/2026)
- 全量快照：4,071 条去重记录；其中 4,064 条带站内 guide 内容。
- 本地保留：全量 JSON、8 篇官方 PDF、纯文本、候选记录和方法图。
- 所有结果数字由官方 CVF/arXiv PDF 的表格或正文人工核对；聚合站自动摘要未作为证据来源。
- Scene Graph 论文按官方预印本题名记录，以避免沿用聚合页的题名差异。
