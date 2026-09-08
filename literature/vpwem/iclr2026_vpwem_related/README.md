# ICLR 2026 Memory 专题中与 VPWEM 相关的论文

整理时间：2026-07-21

来源：[ICLR 2026 专题 — Memory（记忆机制）](https://jcn8jlt2fong.feishu.cn/docx/ArXqd5BlVosgdxxg6iEcLTKznIg)

筛选对象：原文档全部 259 篇论文。本文保留 22 篇与 VPWEM 具有直接或可迁移关系的工作，重点关注视觉机器人策略、非马尔可夫任务、工作记忆与情景记忆、长历史压缩、Diffusion Policy，以及持续学习。

> 说明：下述摘要和判断基于原飞书文档提供的论文导读整理，尚不等同于逐篇阅读全文后的最终 related-work 结论。正式写论文前应再核对原论文正文、实验设置与最终版本。

## 一、结论先行

### 最优先阅读的 6 篇

1. **MemoryVLA**：与 VPWEM 的概念重叠最高，均涉及工作记忆、长期记忆、非马尔可夫机器人任务和扩散动作生成，是首要相关工作与潜在直接竞争者。
2. **HAMLET**：用 moment token 和轻量记忆模块把 VLA 改造成历史感知策略，是“压缩历史后条件化动作预测”的强基线。
3. **Scaling up Memory for Robotic Control via Experience Retrieval（MemER）**：从长历史中检索任务相关关键帧，用层级策略完成分钟级机器人任务，直接对应 VPWEM 的“选择性保留历史”问题。
4. **Recurrent Action Transformer with Memory（RATE）**：面向 POMDP 和离线 RL 的循环记忆 Transformer，可作为方法设计与长程决策基线参考。
5. **Memory, Benchmark & Robots（MIKASA）**：VPWEM 已使用的记忆密集型机器人基准，其任务分类和评测逻辑应进入论文的问题定义与实验设计。
6. **Demystifying Robot Diffusion Policies**：认为 Diffusion Policy 可能主要在做动作记忆/查表，直接影响 VPWEM 对性能提升来源的解释。

### 对 VPWEM 最重要的判断

- **新颖性风险**：MemoryVLA 与 VPWEM 的认知叙事高度相似。VPWEM 需要更明确地突出自身在策略类型、记忆更新、计算有界性、训练目标、实验任务或效率证据上的独特贡献。
- **基线风险**：仅与短上下文 DP、DP-PTP、RNN 和 MaIL 比较可能不够。应至少讨论 HAMLET、MemoryVLA、MemER 和 RATE；无法复现时，也要解释架构与实验设置差异。
- **机制解释风险**：需要排除“更大模型”“更长上下文”以及“动作查表”三种替代解释。
- **实验机会**：加入最近邻动作检索/ALT 类基线、历史打乱、关键帧遮蔽、记忆 token probing，以及计算量随 episode 长度变化的曲线，会显著增强证据链。

## 二、P0：直接竞争、关键基线与核心问题

### 1. MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation

- 作者：Hao Shi, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu et al.
- 原文：[OpenReview](https://openreview.net/forum?id=54U3XHf7qq)
- 核心方法：以感知 token 和认知 token 构成工作记忆；长期记忆库存储低层细节与高层语义，通过检索、融合和去冗余持续更新；最终由记忆条件化的扩散动作专家生成动作。
- 结果概况：在 SimplerEnv-Bridge、Fractal、LIBERO-5、MIKASA-Robo 以及真实机器人长时程任务上报告了明显提升。
- 与 VPWEM 的关系：**最高度重叠**。两者都以非马尔可夫机器人操控为动机，采用短期/长期记忆分工，并用长期信息条件化扩散式动作生成。
- VPWEM 应对：逐项对比记忆输入、压缩机制、更新方式、token 数量是否固定、是否依赖 VLM/VLA、训练目标、推理复杂度和所用基准；不要只在 related work 中一句带过。

### 2. HAMLET: Switch Your Vision-Language-Action Model into a History-Aware Policy

- 作者：Myungkyu Koo, Daewon Choi, Taeyoung Kim, Kyungmin Lee, Changyeon Kim et al.
- 原文：[OpenReview](https://openreview.net/forum?id=KcJ9U0x6kO)
- 核心方法：用 time-contrastive learning 初始化 moment token，以紧凑表示编码每个时间步，再由轻量记忆模块聚合为用于动作预测的 memory features。
- 结果概况：报告 GR00T N1.5 在历史依赖真实任务上的成功率大幅提升，并在 RoboCasa 与 LIBERO 上继续改进。
- 与 VPWEM 的关系：同样研究“如何把原本只看当前观测的策略变成历史感知策略”，并强调紧凑历史表征。
- VPWEM 应对：突出 Q-Former 情景压缩与 moment-token 聚合的差异；建议增加相同历史预算下的压缩方式对比。

### 3. Scaling up Memory for Robotic Control via Experience Retrieval

- 作者：Ajay Sridhar, Jennifer Pan, Satvik Sharma, Chelsea Finn
- 原文：[OpenReview](https://openreview.net/forum?id=1dH4ARGdwD)
- 核心方法：MemER 用高层策略从自身经验中选择并追踪任务相关关键帧，再结合近期观测生成文本指令，驱动低层 VLA 策略。
- 结果概况：在三项需要分钟级记忆的真实世界机器人操作任务上超过已有方法。
- 与 VPWEM 的关系：直接回答“长历史不应无差别保留，应该选择任务相关证据”。它采用显式关键帧检索，VPWEM 采用学习型固定 token 压缩。
- VPWEM 应对：增加 keyframe retrieval、均匀采样或相似度检索基线，以证明端到端压缩优于显式选择。

### 4. Recurrent Action Transformer with Memory

- 作者：Egor Cherepanov, Aleksei Staroverov, Alexey Kovalev, Aleksandr Panov
- 原文：[OpenReview](https://openreview.net/forum?id=kByN4v0M3e)
- 核心方法：RATE 在 Action Transformer 中加入循环记忆，显式调节信息保留与遗忘，面向部分可观测的离线强化学习。
- 结果概况：覆盖 T-Maze、Memory Maze、POPGym、Atari 和 MuJoCo 等记忆型及标准任务。
- 与 VPWEM 的关系：是“固定历史窗口不够，策略需要跨窗口记忆”的直接架构基线，但动作学习范式与 VPWEM 的行为克隆扩散策略不同。
- VPWEM 应对：在方法对比中说明 recurrent state 与 episodic token 的容量、可解释性、并行训练和长期信息保真度差异。

### 5. Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning

- 作者：Egor Cherepanov, Nikita Kachaev, Alexey Kovalev, Aleksandr Panov
- 原文：[OpenReview](https://openreview.net/forum?id=9cLPurIZMj)
- 核心贡献：提出记忆密集型 RL 任务分类，构建 MIKASA-Base 与包含 32 个桌面机器人记忆任务的 MIKASA-Robo。
- 与 VPWEM 的关系：这是 VPWEM 实验问题成立的重要依据，尤其适合说明“长 horizon”与“真正需要历史信息”并不等价。
- VPWEM 应对：将 MIKASA 的任务分类映射到 VPWEM 的实验；明确每个任务依赖哪种过去不可见变量，而不只汇报平均成功率。

### 6. Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative

- 作者：Chengyang He, Xu Liu, Gadiel Mark Sznaier Camps, Joseph Bruno, Guillaume Adrien Sartoretti et al.
- 原文：[OpenReview](https://openreview.net/forum?id=PL0tJOfm7I)
- 核心观点：Diffusion Policy 在低数据场景下可能主要通过视觉近邻召回训练 action chunk，而不是学习平滑的动作泛化；论文提出显式 Action Lookup Table（ALT）作为简单替代。
- 与 VPWEM 的关系：这是 VPWEM 最重要的机制反例。长期记忆带来的提升可能来自更容易定位训练轨迹，而不一定来自抽象的情景记忆。
- VPWEM 应对：加入最近邻动作检索、训练轨迹相似度、OOD 观测、历史打乱和动作片段来源分析；证明记忆 token 编码了任务变量，而不只是样本索引。

### 7. Planning with an Embodied Learnable Memory

- 作者：Priyam Parashar, Jacob Krantz, Matthew Chang, Kavit Shah, Xavier Puig et al.
- 原文：[OpenReview](https://openreview.net/forum?id=79BOATBal9)
- 核心方法：Embodied Perception Memory 使用统一 VLM，根据自我中心视觉维护、更新文本环境表征，并结合模仿学习与难度感知强化学习进行长时域规划。
- 与 VPWEM 的关系：任务层级更偏高层规划，但同样体现“记忆表示必须随环境变化更新”，可帮助界定 VPWEM 的低层控制定位。

## 三、P1：可迁移的记忆压缩与时序建模方法

### 8. Composition of Memory Experts for Diffusion World Models

- 作者：Sebastian Stapf, Pablo Acuaviva, Aram Davtyan, Paolo Favaro
- 原文：[OpenReview](https://openreview.net/forum?id=sUEdpZCHdp)
- 核心方法：组合短期、长期和空间长期记忆专家，用 diffusion product-of-experts 兼顾局部细节、情景历史与空间一致性。
- 对 VPWEM 的启示：可以把工作记忆与情景记忆视为具有不同职责的专家，而不是简单拼接 token；适合启发独立 dropout、门控融合或专家消融。

### 9. Task-Related Token Compression in Multimodal Large Language Models from an Explainability Perspective

- 作者：Lei Lei, Jie Gu, Xiaokang Ma, Chu Tang, Jingmin Chen et al.
- 原文：[OpenReview](https://openreview.net/forum?id=YULeQtSyiW)
- 核心方法：根据指令评估视觉 token 的全局重要性，在模型输入阶段执行任务相关压缩，并用轻量网络近似完整可解释性计算。
- 对 VPWEM 的启示：可将“哪些历史值得记住”显式变成任务相关重要性问题；适合作为 memory token 可解释性和压缩质量的分析参照。

### 10. MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding

- 作者：Peiran Wu, Zhuorui Yu, Yunze Liu, Chi-Hao Wu, Enmin Zhou et al.
- 原文：[OpenReview](https://openreview.net/forum?id=5gLgjn3MVT)
- 核心方法：先用视觉记忆检索器选择事件片段，再用强化学习和知识蒸馏训练压缩模型。
- 结果概况：报告视觉 token 减少 95%，同时明显降低显存和延迟。
- 对 VPWEM 的启示：提示应同时评估任务性能、token 压缩率、显存和延迟；仅报告成功率不足以证明记忆压缩有效。

### 11. VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling

- 作者：Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu et al.
- 原文：[OpenReview](https://openreview.net/forum?id=MUjdNcfNPv)
- 核心方法：HiCo 从 clip 级逐层压缩到 video 级表示，并配合由短到长的训练课程。
- 对 VPWEM 的启示：可参考分层压缩与 short-to-long curriculum；也应测试历史长度超出训练分布时的泛化。

### 12. Short Window Attention Enables Long-Term Memorization

- 作者：Loïc Cabannes, Maximilian Beck, Maria Lomeli, Gergely Szilvasy, Matthijs Douze et al.
- 原文：[OpenReview](https://openreview.net/forum?id=btgVfhudI1)
- 核心发现：过大的局部 attention 窗口可能让模型逃避使用长期记忆；随机改变窗口大小可同时训练短程建模与长期记忆。
- 对 VPWEM 的启示：VPWEM 的 working-memory 长度可能决定模型是否真正使用 episodic memory。建议随机化工作窗口，或至少报告不同窗口下长期记忆利用率。

### 13. MoM: Linear Sequence Modeling with Mixture-of-Memories

- 作者：Jusen Du, Weigao Sun, Disen Lan, Jiaxi Hu, Tao Zhang et al.
- 原文：[OpenReview](https://openreview.net/forum?id=3PdOq8Rgue)
- 核心方法：维护多个独立 memory state，通过路由网络动态分配输入，减少单一固定状态中的记忆干扰。
- 对 VPWEM 的启示：当一个 Q-Former summary 同时承担目标、阶段、空间和物体状态时可能发生干扰；可探索分槽记忆或路由式 query token。

### 14. JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation

- 作者：Shuang Zeng, Dekang Qi, Xinyuan Chang, Feng Xiong, Xie Shichao et al.
- 原文：[OpenReview](https://openreview.net/forum?id=RnuB0Nlbd5)
- 核心方法：将空间几何记忆与视觉语义记忆建模为两个固定大小的隐式状态，并通过保留初始窗口和滑动窗口 KV 实现增量更新。
- 对 VPWEM 的启示：记忆可按信息类型解耦；VPWEM 可分析不同 query token 是否自然形成目标、空间或阶段专门化。

### 15. StreamingVLM: Real-Time Understanding for Infinite Video Streams

- 作者：Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng et al.
- 原文：[OpenReview](https://openreview.net/forum?id=gVbPWbA97s)
- 核心方法：复用 attention sink 状态，并分别维护近期视觉和文本窗口，使长视频流的延迟和内存保持稳定。
- 对 VPWEM 的启示：可借鉴“无限流式输入下资源不随时间增长”的评测方式，明确给出显存、延迟与 episode 长度的关系。

## 四、P2：持续学习、VLA 泛化与长期记忆扩展

### 16. Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting

- 作者：Asher James Hancock, Xindi Wu, Lihan Zha, Olga Russakovsky, Anirudha Majumdar
- 原文：[OpenReview](https://openreview.net/forum?id=sFO9d6XSlf)
- 核心方法：将低层动作表示为自然语言，减小 VLM 预训练语料与机器人微调数据之间的分布差异，再用 LoRA 训练 VLA。
- 与 VPWEM 的关系：重点不在 episode 内记忆，但可帮助区分“情景记忆”与“跨任务知识保留”，也提醒视觉编码器微调可能导致基础能力退化。

### 17. Robust Fine-tuning of Vision-Language-Action Robot Policies via Parameter Merging

- 作者：Yajat Yadav, Zhiyuan Zhou, Andrew Wagenmaker, Karl Pertsch, Sergey Levine
- 原文：[OpenReview](https://openreview.net/forum?id=uWJwQ5SZoM)
- 核心方法：融合预训练与微调模型权重，使单一策略兼顾新技能和原有通用能力，并扩展到终身学习。
- 与 VPWEM 的关系：若 VPWEM 后续扩展到跨任务持续学习，这是一条比情景缓存更直接的知识保持基线。

### 18. Vision-Language-Action Instruction Tuning: From Understanding to Manipulation

- 作者：Shuai Yang, Hao Li, Bin Wang, Yilun Chen, Yang Tian et al.
- 原文：[OpenReview](https://openreview.net/forum?id=tsxwloasw5)
- 核心方法：以混合专家适配联合训练具身推理与动作生成，在保留 VLM 能力的同时提升操控性能。
- 与 VPWEM 的关系：可作为“更强基础策略本身是否已经解决长期任务”的背景；VPWEM 应证明收益不是弱 backbone 特有现象。

### 19. Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning

- 作者：Ke Sun, Hongming Zhang, Jun Jin, Chao Gao, Xi Chen et al.
- 原文：[OpenReview](https://openreview.net/forum?id=loNTDX3wTn)
- 核心方法：受海马体—皮层分工启发，以 fast learner 负责迁移、meta learner 负责知识整合与抗遗忘。
- 与 VPWEM 的关系：与“工作记忆—情景记忆”具有相似的认知分工，但发生在跨任务持续学习层面，可用于澄清 VPWEM 只处理单 episode 内记忆。

### 20. Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory

- 作者：Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin et al.
- 原文：[OpenReview](https://openreview.net/forum?id=PMz29A7Muq)
- 核心方法：M3-Agent 从连续视听输入构建实体中心的情景与语义记忆，通过多轮检索和推理回答长视频问题。
- 与 VPWEM 的关系：动作控制并非核心，但其情景/语义记忆分层和机器人视角长视频评测可用于记忆表征分析。

### 21. Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization

- 作者：Taeyoon Kwon, Dongwook Choi, Hyojun Kim, Sunghwan Kim, Seungjun Moon et al.
- 原文：[OpenReview](https://openreview.net/forum?id=E5L43l5EIu)
- 核心方法：研究具身智能体如何利用历史交互中的物体语义和用户行为序列，并用层级知识图谱缓解信息过载与多记忆协调失败。
- 与 VPWEM 的关系：提示长期记忆不仅有容量问题，还有检索协调问题；可作为未来面向个性化或跨 episode 记忆的扩展方向。

### 22. Building Spatial World Models from Sparse Transitional Episodic Memories

- 作者：Zizhan He, Maxime Daigle, Pouya Bashivan
- 原文：[OpenReview](https://openreview.net/forum?id=w3w7WVG4ks)
- 核心方法：Episodic Spatial World Model 从稀疏、不连续的情景转移记忆中恢复空间结构，并用于适应、探索和导航。
- 与 VPWEM 的关系：从机制角度支持“稀疏关键情景足以恢复任务结构”，可为 memory probing 或稀疏历史采样提供理论动机。

## 五、建议写入 VPWEM 论文的 related-work 结构

### 1. Memory-aware robot policies

重点讨论 MemoryVLA、HAMLET、MemER、RATE，并用以下维度进行统一比较：

- 策略类别：Diffusion Policy、VLA、offline RL Transformer 或层级策略；
- 记忆来源：原始视觉帧、moment token、关键帧、循环隐状态或学习型 summary token；
- 更新机制：追加、检索、循环更新、去冗余或 Q-Former 压缩；
- 资源复杂度是否随 episode 长度增长；
- 是否专门评估非马尔可夫机器人控制。

### 2. Efficient long-history compression

讨论任务相关 token 压缩、MARC、VideoChat-Flash、SWAX、MoM 和 StreamingVLM，说明 VPWEM 的位置不是通用长上下文模型，而是由动作预测目标监督的机器人历史压缩器。

### 3. Memory-intensive benchmarks and mechanisms

以 MIKASA 说明问题定义，以 Demystifying Robot Diffusion Policies 说明动作记忆替代解释，再用 Memory Experts 与 Episodic Spatial World Model 支撑短期/长期及稀疏情景分工。

## 六、建议补充的实验

1. **直接竞争基线**：MemoryVLA、HAMLET、MemER；若工程上无法完整复现，至少实现统一 backbone 下的 moment pooling、关键帧检索与 recurrent memory 近似版本。
2. **查表基线**：最近邻视觉 embedding 检索训练 action chunk，报告与 Diffusion Policy、VPWEM 的差异。
3. **记忆真实性测试**：打乱历史顺序、替换其他 episode 历史、遮蔽初始目标帧、仅保留伪相关动作历史。
4. **记忆表征 probing**：从 memory token 预测目标位置、任务阶段、已完成子目标或初始隐藏变量。
5. **计算有界性**：横轴使用 episode 长度，报告显存、单步延迟和成功率；与完整上下文及固定滑窗对比。
6. **工作窗口依赖**：改变 working-memory 长度，检查窗口过大时模型是否不再使用 episodic memory。
7. **压缩方式消融**：FIFO、均匀采样、相似度检索、关键帧选择、Q-Former，以及固定/动态 memory token 数量。
8. **正负对照**：在真正需要记忆的 POMDP、自然长时程任务和近似 Markov 任务上分别报告结果。

## 七、推荐阅读顺序

1. MemoryVLA
2. HAMLET
3. Scaling up Memory for Robotic Control via Experience Retrieval
4. Demystifying Robot Diffusion Policies
5. Memory, Benchmark & Robots
6. Recurrent Action Transformer with Memory
7. Task-Related Token Compression
8. Short Window Attention Enables Long-Term Memorization
9. Composition of Memory Experts for Diffusion World Models
10. 其余 P1、P2 论文按实验需求查阅
