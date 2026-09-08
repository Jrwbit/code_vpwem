# 2026 顶会与 VPWEM 相关论文：图文导读版

更新时间：2026-07-22

共 122 篇。图片和中文导读来自 Connected Researchers；没有站内素材的论文仅保留文字和原文链接。

说明：本版用于快速阅读与筛选，方法和结果数字在正式引用前仍需回到官方论文核验。

## ICLR 2026

本轮收录 27 篇。

### P0

#### Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative

![Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-PL0tJOfm7I-90e4971575.webp)

- 导读：针对视觉运动机器人操作任务的 Diffusion Policy 在仅使用少量任务演示数据进行训练的情况下，展现出卓越的灵巧性与鲁棒性。然而，这一优异表现背后的原因至今仍是一个谜。本文提出一个令人意外的假说：Diffusion Policy 本质上是在记忆一张动作查找表——而这恰恰是有益的。我们认为，在推理阶段，Diffusion Policy 会在潜在空间中找到与测试图像最接近的训练图像，并召回与之关联的训练动作（即 action chunk），从而实现无需动作泛化的反应式控制。在数据稀疏的场景下，由于数据密度不足以让模型习得动作泛化能力，这一机制反而行之有效。我们通过系统性的实证证据支撑上述论断，结果显示：即便以高度分布外（out-of-distribution，OOD）的图像作为条件输入，Diffusion Policy 输出的 action chunk 仍来自训练数据。我们在同一数据集上对三类具有代表性的策略族进行评估与比较：Diffusion Policy、Action Chunking with Transformers（ACT），以及预训练通用型 Vision-Langu…
- 方法：提出假设：扩散策略通过记忆动作查找表实现高性能。；引入动作查找表（ALT）策略作为扩散策略的简单替代。
- 结果：扩散策略在OOD情况下表现出强烈的动作记忆性和鲁棒性。；ACT在OOD情况下表现出动作插值但鲁棒性较差。
- 对 VPWEM：关键反例：用于检验策略是否只是复现动作或轨迹。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-PL0tJOfm7I-90e4971575) · [原文](https://openreview.net/forum?id=PL0tJOfm7I)

#### Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization

![Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-E5L43l5EIu-76e1d62d28.webp)

- 导读：基于大语言模型（LLM）的具身智能体（embodied agents）已在传统物体重排任务中取得成功，但如何提供个性化辅助——即利用来自历史交互的用户特定知识——仍面临新的挑战。我们从智能体记忆利用（memory utilization）的视角对这些挑战展开研究，聚焦两个关键维度：物体语义（object semantics，即根据个人含义识别物体）与用户模式（user patterns，即从行为习惯中回溯行动序列）。为评估上述能力，我们构建了 Memento——一个端到端的两阶段评估框架，涵盖单记忆（single-memory）与联合记忆（joint-memory）两类任务。实验结果表明，现有智能体能够回忆简单的物体语义，但在将序列化用户模式应用于规划（planning）方面存在明显不足。通过深入分析，我们识别出两个关键瓶颈：信息过载（information overload）以及处理多条记忆时出现的协调失效（coordination failures）。基于上述发现，我们探索了应对这些挑战的记忆架构方案。鉴于我们观察到情节记忆（episodic memory）同时提供个性化知识与…
- 方法：构建Memento评估框架，包含单内存和联合内存任务。；通过物体语义和用户模式两个维度研究内存利用。
- 结果：当前代理能够回忆简单的物体语义。；代理在将序列用户模式应用于规划时存在困难。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-E5L43l5EIu-76e1d62d28) · [原文](https://openreview.net/forum?id=E5L43l5EIu)

#### HAMLET: Switch Your Vision-Language-Action Model into a History-Aware Policy

![HAMLET: Switch Your Vision-Language-Action Model into a History-Aware Policy](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-KcJ9U0x6kO-5e81815031.webp)

- 导读：机器人操控任务本质上依赖于历史信息：充分利用过去的上下文往往能带来显著收益。然而，现有的大多数 Vision-Language-Action 模型（VLAs）在设计之初并未考虑这一点，即它们仅依赖当前观测，而忽略了先前的上下文信息。本文提出 HAMLET——一个可扩展的框架，用于使 VLAs 在预测动作时能够关注历史上下文。具体而言，我们引入了 moment tokens，以紧凑的方式对每个时间步的感知信息进行编码。这些表示通过 time-contrastive learning 进行初始化，使其能够更好地捕捉时间上具有区分性的特征。随后，我们采用一个轻量级的记忆模块，将跨历史时间步的 moment tokens 整合为 memory features，并将其用于动作预测。通过大量实验评估，我们证明 HAMLET 能够成功地将最先进的 VLA 转化为具备历史感知能力的策略，在需要历史上下文的长时程任务上尤为突出。具体而言，基于 GR00T N1.5，HAMLET 在依赖历史信息的真实世界任务上实现了 76.4% 的平均成功率，较基线性能提升了 47.2%。此外，HAMLET 在 Ro…
- 方法：引入时刻令牌紧凑编码每个时间步的感知信息。；通过时间对比学习初始化时刻令牌表示以捕捉时间独特性。
- 结果：在GR00T N1.5上实现76.4%的平均成功率，超越基线47.2%。；将RoboCasa Kitchen（100-demo设置）性能从64.1%提升至66.4%。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-KcJ9U0x6kO-5e81815031) · [原文](https://openreview.net/forum?id=KcJ9U0x6kO)

#### Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning

![Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-9cLPurIZMj-9f2377caa1.webp)

- 导读：记忆能力对于使智能体能够处理具有时序与空间依赖关系的复杂任务至关重要。尽管许多强化学习（RL）算法已集成了记忆机制，但该领域仍缺乏一套通用基准来评估智能体在多样化场景下的记忆能力。这一空白在桌面机器人操作任务中尤为突出——在此类任务中，记忆能力对于解决部分可观测条件下的任务以及确保鲁棒性能不可或缺，然而目前尚无标准化基准可供使用。为填补这一空白，我们提出 MIKASA（Memory-Intensive Skills Assessment Suite for Agents，面向智能体的记忆密集型技能评估套件），这是一个面向记忆强化学习的综合性基准，包含三项核心贡献：（1）我们提出了一套针对记忆密集型 RL 任务的系统性分类框架；（2）我们构建了 MIKASA-Base——一个统一基准，能够对记忆增强型智能体在多样化场景下进行系统性评估；（3）我们开发了 MIKASA-Robo——一个包含 32 个精心设计的记忆密集型任务的新型基准，专门用于评估智能体在桌面机器人操作场景中的记忆能力。本研究提出了一套统一框架，以推动记忆强化学习领域的研究进展，从而为现实世界应用构建更为鲁棒的系统。
- 方法：提出了记忆密集型RL任务的综合分类框架。；收集了MIKASA-Base，用于系统评估记忆增强智能体。
- 结果：提出了MIKASA，首个记忆强化学习的综合基准。；MIKASA-Base实现了记忆增强智能体的系统评估。
- 对 VPWEM：直接评测依据：可用于补充记忆任务与指标。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-9cLPurIZMj-9f2377caa1) · [原文](https://openreview.net/forum?id=9cLPurIZMj)

#### MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation

![MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-54U3XHf7qq-981996a4fc.webp)

- 导读：时序上下文对于机器人操作至关重要，因为此类任务本质上是非马尔可夫的（non-Markovian），然而主流的 VLA（Vision-Language-Action）模型通常忽视这一点，在长时域、具有时序依赖性的任务上表现欠佳。认知科学表明，人类依赖工作记忆（working memory）来缓存短时表征以支持即时控制，同时依赖海马体系统（hippocampal system）保留过去经验的逐字情节细节（verbatim episodic details）与语义要旨（semantic gist），以实现长时记忆。受上述机制启发，我们提出 MemoryVLA——一个面向长时域机器人操作的 Cognition-Memory-Action 框架。预训练的 VLM 将观测编码为感知 token 和认知 token，共同构成工作记忆；与此同时，一个感知-认知记忆库（Perceptual-Cognitive Memory Bank）从工作记忆中整合并存储低层细节与高层语义。工作记忆从记忆库中检索与决策相关的条目，将其与当前 token 进行自适应融合，并通过合并冗余信息来更新记忆库。基于这些 tok…
- 方法：仿生记忆架构：工作记忆处理即时控制，海马式记忆库存储长期细节与语义；感知-认知双通道编码：VLM将观测分解为感知令牌与认知令牌
- 结果：仿真任务成功率：SimplerEnv-Bridge 71.9%、Fractal 72.7%、LIBERO-5 96.5%、Mikasa-Robo 41.2%；显著超越基线：Bridge任务+14.6提升，Mikasa-Robo+11.8提升
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-54U3XHf7qq-981996a4fc) · [原文](https://openreview.net/forum?id=54U3XHf7qq)

#### Scaling up Memory for Robotic Control via Experience Retrieval

![Scaling up Memory for Robotic Control via Experience Retrieval](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-1dH4ARGdwD-0aa94ae7c2.webp)

- 导读：人类依赖记忆来完成任务；我们的目标是赋予机器人策略同等的能力。直接基于长时间观测历史进行条件化计算代价高昂，且在协变量偏移下表现脆弱；而对历史信息进行无差别下采样则会引入无关或冗余的信息。为此，我们提出一种层级式策略框架，其中高层策略被训练为从自身经验中选取并追踪与当前任务相关的历史关键帧（keyframes）。高层策略利用所选关键帧与最近帧，为低层策略生成文本指令并驱动其执行。该设计与现有的视觉-语言-动作（vision-language-action，VLA）模型兼容，使系统能够高效地推理长时程依赖关系。在实验中，我们分别以 Qwen2.5-VL-7B-Instruct 和 作为高层策略与低层策略进行微调，所用训练数据为演示样本辅以少量语言标注。我们的方法 MemER 在三项真实世界长时程机器人操作任务上超越了此前方法，这些任务需要数分钟量级的记忆能力。相关视频与代码可见于 https://jen-pan.github.io/memer/。
- 方法：提出分层策略框架：高层策略选择任务相关关键帧，低层策略执行指令；高层策略结合关键帧与最近帧生成低层指令
- 结果：在三个需要分钟级记忆的真实机器人任务上超越基线方法；使用Qwen2.5-VL-7B-Instruct和π0.5分别作为高/低层策略
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-1dH4ARGdwD-0aa94ae7c2) · [原文](https://openreview.net/forum?id=1dH4ARGdwD)

### P1

#### Ctrl-World: A Controllable Generative World Model for Robot Manipulation

![Ctrl-World: A Controllable Generative World Model for Robot Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-748bHL2BAv-84770bc914.webp)

- 导读：通用机器人策略如今已能执行多种操控技能，但如何评估和提升其在面对陌生物体与新指令时的能力，依然是一项重大挑战。严格的评估需要大量真实世界的rollout，而系统性的改进则需要额外的带专家标注的纠正数据。这两个过程都耗时、高成本，且难以扩展。 世界模型通过允许策略在想象空间中执行rollout，提供了一种极具前景的可扩展替代方案。然而，构建一个能够支持通用机器人策略进行多步交互的可控世界模型，仍是核心挑战所在。这要求世界模型兼容现代通用策略，具体而言需支持多视角预测、细粒度动作控制以及一致的长时域交互——而这些能力均未被此前的工作所实现。在本文中，我们向前迈进一步，提出了一个可控的多视角世界模型，可用于评估和提升通用机器人策略的指令跟随能力。我们的模型通过基于位姿条件的记忆检索机制来维持长时域一致性，并通过帧级动作条件化实现精确的动作控制。该模型在DROID数据集（95k条轨迹，564个场景）上进行训练，能够在新颖场景和新相机位置下生成时空一致的轨迹，持续时长超过20秒。我们证明，该方法无需真实世界机器人rollout即可准确地对策略性能进行排名。此外，通过在想象空间中合成成功轨迹并将…
- 方法：设计可控多视角世界模型，支持多视角预测和细粒度动作控制；采用姿态条件记忆检索机制保持长时交互一致性
- 结果：在新场景和相机布局下生成时空一致轨迹超过 20 秒；无需真实 rollout 即可准确排名策略性能
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-748bHL2BAv-84770bc914) · [原文](https://openreview.net/forum?id=748bHL2BAv)

#### DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving

![DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-plrGn3RdzN-602fb368ee.webp)

- 导读：Scaling Vision-Language-Action (VLA) 模型于大规模数据之上，为实现更具泛化能力的驾驶智能提供了一条颇具前景的路径。然而，VLA 模型受制于一种"监督赤字"（supervision deficit）：庞大的模型容量仅由稀疏、低维的动作信号加以监督，导致其大量表征能力被闲置浪费。为弥补这一缺陷，我们提出 DriveVLA-W0——一种以世界建模（world modeling）为核心、通过预测未来图像来驱动训练的范式。该任务能够生成密集的自监督信号，迫使模型深入学习驾驶环境的内在动态规律。我们通过将该范式实例化于两类主流 VLA 架构，充分展示其通用性：针对使用离散视觉 token 的 VLA，采用自回归世界模型（autoregressive world model）；针对基于连续视觉特征运行的 VLA，则采用扩散世界模型（diffusion world model）。在世界建模所习得的丰富表征基础之上，我们进一步引入一个轻量级动作专家（action expert）模块，以降低推理延迟，满足实时部署需求。在 NAVSIM v1/v2 benchmark…
- 方法：提出DriveVLA-W0训练范式：用世界建模预测未来图像生成密集自监督信号；为离散视觉标记VLA构建自回归世界模型，为连续特征VLA构建扩散世界模型
- 结果：在NAVSIM v1/v2基准测试显著超越BEV和VLA基线方法；在680倍大的内部数据集验证有效性
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-plrGn3RdzN-602fb368ee) · [原文](https://openreview.net/forum?id=plrGn3RdzN)

#### Empowering Multi-Robot Cooperation via Sequential World Models

![Empowering Multi-Robot Cooperation via Sequential World Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-IvUM6UwYCJ-8bad520056.webp)

- 导读：基于模型的强化学习（MBRL）凭借其高样本效率和规划能力，在机器人领域取得了卓越成果。然而，将 MBRL 扩展至物理多机器人协作仍面临重大挑战，根本原因在于联合动力学的建模复杂度极高。为此，我们提出 Sequential World Model（SeqWM），一种将序列化范式融入多机器人 MBRL 的全新框架。SeqWM 采用独立的、自回归的智能体级世界模型来表征联合动力学：每个智能体在其前驱智能体预测结果的基础上生成自身的未来轨迹并规划动作。这一设计降低了建模复杂度，并通过显式的意图共享机制使高级协作行为得以自然涌现。在 Bi-DexHands 和 Multi-Quadruped 两个基准上的实验表明，SeqWM 在整体性能和样本效率两个维度上均优于现有最先进的有模型和无模型基线方法，并展现出预测性自适应、时序对齐和角色分工等高级协作行为。此外，SeqWM 已成功部署于真实四足机器人上，验证了其在现实多机器人系统中的有效性。演示视频与代码已开源发布。
- 方法：提出顺序世界模型（SeqWM），将顺序范式融入多机器人MBRL；采用独立自回归的智能体级世界模型表示联合动力学
- 结果：在Bi-DexHands和Multi-Quadruped上超越现有基于模型和无模型基线；样本效率显著优于对比方法
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-IvUM6UwYCJ-8bad520056) · [原文](https://openreview.net/forum?id=IvUM6UwYCJ) · [代码](https://github.com/zhaozijie2022/seqwm)

#### H$^3$DP: Triply‑Hierarchical Diffusion Policy for Visuomotor Learning

![H$^3$DP: Triply‑Hierarchical Diffusion Policy for Visuomotor Learning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-Q1CP0iAmOb-1adba80f6f.webp)

- 导读：视觉运动策略学习（Visuomotor policy learning）在机器人操作领域取得了显著进展，近期方法主要依赖生成模型对动作分布进行建模。然而，这些方法往往忽视了视觉感知与动作预测之间至关重要的耦合关系。在本工作中，我们提出三重层级扩散策略（Triply-Hierarchical Diffusion Policy，H³DP），这是一种新颖的视觉运动学习框架，通过显式引入层级结构来强化视觉特征与动作生成之间的整合。H³DP 包含 3 个层级：（1）基于深度信息对 RGB-D 观测进行组织的深度感知输入分层（depth-aware input layering）；（2）以不同粒度级别编码语义特征的多尺度视觉表征（multi-scale visual representations）；（3）将由粗到精的动作生成与相应视觉特征对齐的层级条件扩散过程（hierarchically conditioned diffusion process）。大量实验表明，H³DP 在 44 个仿真任务中相较于基线方法实现了平均 +27.5% 的相对性能提升，并在 4 个具有挑战性的双臂真实世界操作任…
- 方法：深度感知输入分层：基于深度信息组织RGB-D观测数据。；多尺度视觉表征：编码不同粒度的语义特征。
- 结果：在44个仿真任务中平均相对基线提升27.5%。；在4项双手现实操作任务中表现优越。
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-Q1CP0iAmOb-1adba80f6f) · [原文](https://openreview.net/forum?id=Q1CP0iAmOb)

#### Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control

![Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-OeDwYtp8n1-64271e301d.webp)

- 导读：近期视频扩散模型（video diffusion models）的进展在生成机器人决策数据方面展现出良好前景，而轨迹条件控制（trajectory conditions）进一步赋予了对生成过程的精细化操控能力。然而，现有方法主要聚焦于单一物体的运动建模，难以捕捉复杂操作任务中至关重要的多物体交互关系。这一局限源于物体重叠区域中特征的相互耦合，导致生成视觉质量下降。为解决上述问题，我们提出 RoboMaster——一种新颖的框架，通过协作轨迹公式化（collaborative trajectory formulation）对物体间的动态关系进行建模。与此前将多个物体分别解耦的方法不同，我们的核心思路是将交互过程拆解为三个子阶段：交互前（pre-interaction）、交互中（interaction）与交互后（post-interaction），并在各阶段以主导物体为核心分别建模——具体而言，交互前与交互后阶段以机械臂为主导物体，交互阶段则以被操作物体为主导物体。这一设计有效缓解了先前工作中多物体特征融合所带来的问题。为进一步保证视频中各帧的主体语义一致性，我们引入了兼顾外观与形状的隐…
- 方法：将交互过程分解为三阶段：交互前（机械臂主导）、交互中（被操作物体主导）、交互后（机械臂主导）；采用协作轨迹公式建模物体间动力学
- 结果：在Bridge数据集及RLBench、SIMPLER基准上达到最先进性能；有效缓解多物体特征融合问题
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-OeDwYtp8n1-64271e301d) · [原文](https://openreview.net/forum?id=OeDwYtp8n1)

#### Real-Time Robot Execution with Masked Action Chunking

![Real-Time Robot Execution with Masked Action Chunking](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-r0RGJ1j9on-6789bd231b.webp)

- 导读：实时执行对于机器人等信息物理系统至关重要。这类系统运行于动态的真实世界环境中，即便是微小的延迟也会削弱系统的响应能力并损害整体性能。异步推理近年来作为一种系统级范式兴起，用于实现机器人操作的实时响应——它允许在当前 action chunk 执行期间同步预测下一个 action chunk。尽管这一方法能够实现实时响应，但朴素的集成方式往往导致执行失败。 先前的方法将此类失败归因于 chunk 间的不连续性（inter-chunk discontinuity），并开发了测试阶段算法来平滑 chunk 边界。与之不同，我们识别出另一个关键却长期被忽视的因素：chunk 内不一致性（intra-chunk inconsistency），即机器人实际执行的 action chunk 与其当前感知之间存在部分错位。为解决这一问题，我们提出 REMAC，该方法通过 masked action chunking 在预训练策略的基础上学习纠正性调整，使策略在异步推理过程中面对预期动作与实际执行之间的不匹配时保持鲁棒性。此外，我们还引入了一种 prefix-preserved 采样流程，以强化 ch…
- 方法：提出REMAC方法，通过掩码动作块学习对预训练策略进行校正调整。；引入前缀保留采样程序以增强块间连续性。
- 结果：在仿真和真实环境中实现更快的任务执行。；在不同延迟下保持鲁棒性。
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-r0RGJ1j9on-6789bd231b) · [原文](https://openreview.net/forum?id=r0RGJ1j9on)

#### SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation

![SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-aemqAxScl9-d56aaf448a.webp)

- 导读：大规模机器人学习在复杂操作任务上已取得进展，然而长时域、富含接触的问题——尤其是涉及可变形物体的任务——由于示范数据质量参差不齐，仍面临严峻挑战。本文提出一种阶段感知的、基于视频的奖励建模框架（SARM），该框架联合预测任务阶段与细粒度进度，利用自然语言子任务标注在长度不一的示范序列中导出一致的标签。这一方法规避了基于帧索引标注的脆弱性，即便在折叠T恤衫等任务中也能提供稳定的监督信号。所提出的奖励模型对示范数据的多样性具有鲁棒性，能够泛化至分布外场景，并有效提升下游策略的训练效果。在此基础上，本文进一步提出奖励对齐行为克隆（Reward-Aligned Behavior Cloning，RA-BC），通过依据奖励估计对示范数据进行过滤与重新加权来改进策略学习。实验结果表明，本方法在真实机器人部署和人工验证两个层面均显著优于各基线方法。在折叠T恤衫任务中，从展平初始状态出发的成功率达到83%，从揉皱初始状态出发的成功率达到67%，而原始行为克隆（vanilla BC）在相同条件下分别仅为8%与0%。综合来看，本文的结果表明奖励建模是一种可扩展、标注高效的解决方案，适用于长时域机器人操作…
- 方法：提出基于视频的阶段感知奖励框架，联合预测任务阶段与细粒度进度；利用自然语言子任务注释生成跨可变长度演示的 consistent 标签
- 结果：T恤折叠任务中：平整状态成功率83%（原始BC仅8%），皱褶状态67%（原始BC为0%）；奖励模型对演示变异性具有鲁棒性且能泛化至分布外场景
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-aemqAxScl9-d56aaf448a) · [原文](https://openreview.net/forum?id=aemqAxScl9)

#### Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments

![Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-LQD1MrnbxH-1f89c32997.webp)

- 导读：基于语言模型（LM）的具身智能体正被越来越广泛地部署于真实世界场景之中。然而，在动态环境中，这类智能体的适应能力依然有限——而构建准确且灵活的世界模型对于有效推理与决策至关重要。为应对这一挑战，我们将 Mixture-of-Experts（MoE）范式扩展至具身智能体领域。传统 MoE 架构将知识模块化为专家组件并采用预训练路由机制，但一旦部署便趋于僵化，难以适应动态环境中未曾见过的领域。为此，我们提出 Test-time Mixture of World Models（TMoW）框架，以增强智能体对未知领域及持续演变领域的适应能力。与传统 MoE 中路由函数在部署后保持固定不同，TMoW 在测试阶段动态更新其针对世界模型的路由函数，使智能体能够重组现有模型并整合新模型，从而实现持续适应。具体而言，TMoW 通过以下三项机制实现上述目标：（i）多粒度基于原型的路由（multi-granular prototype-based routing），可跨越从物体级到场景级的相似性层次动态调整混合方式；（ii）测试时精化（test-time refinement），在推理阶段将未知领域特征与…
- 方法：多粒度原型路由机制: 根据物体到场景级别的相似性动态调整模型混合；测试时精炼: 在推理过程中对齐未知领域特征与原型
- 结果：在VirtualHome、ALFWorld和RLBench基准测试中表现优异；在零样本适应和小样本扩展场景均展现强大性能
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-LQD1MrnbxH-1f89c32997) · [原文](https://openreview.net/forum?id=LQD1MrnbxH)

#### WMPO: World Model-based Policy Optimization for Vision-Language-Action Models

![WMPO: World Model-based Policy Optimization for Vision-Language-Action Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-qE2FyvRvuF-12550585ae.webp)

- 导读：Vision-Language-Action（VLA）模型在通用机器人操控领域展现出强劲潜力，但其对专家示范数据的依赖限制了模型从失败中学习以及执行自我纠错的能力。强化学习（RL）通过与物理环境的自我改进式交互来克服上述局限，然而在真实机器人上部署时面临极高的样本复杂度问题。为此，我们提出基于世界模型的策略优化方法 World-Model-based Policy Optimization（WMPO）——一个无需与真实环境交互即可实现 on-policy VLA RL 训练的原则性框架。与广泛使用的隐空间世界模型不同，WMPO 专注于基于像素的预测，使"想象"出的轨迹与利用大规模网络图像预训练所得的 VLA 特征相对齐。至关重要的是，WMPO 使策略能够执行 on-policy GRPO 优化，相较于常用的 off-policy 方法取得了更强的性能表现。我们在仿真环境与真实机器人两种设置下开展了大量实验，结果表明 WMPO 能够：（i）大幅提升样本效率；（ii）实现更强的整体性能；（iii）涌现出自我纠错等突现行为；（iv）展现出鲁棒的泛化能力与终身学习能力。
- 方法：提出基于世界模型的策略优化（WMPO），不与真实环境交互。；专注于基于像素的预测，将想象轨迹与预训练的VLA特征对齐。
- 结果：显著提高了样本效率。；实现了更强的整体性能。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-qE2FyvRvuF-12550585ae) · [原文](https://openreview.net/forum?id=qE2FyvRvuF)

#### villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models

![villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-y5CaJb17Fn-6ac90d96f8.webp)

- 导读：Vision-Language-Action (VLA) 模型已成为学习机器人操作策略的一种主流范式，能够遵循语言指令并泛化到新颖场景。近期研究开始探索将潜在动作（latent actions）——即两帧之间运动的抽象表示——引入 VLA 预训练过程。在本文中，我们提出 villa-X，一种新颖的 Vision-Language-Latent-Action (ViLLA) 框架，旨在推进潜在动作建模，以学习可泛化的机器人操作策略。 我们的方法在两个维度上进行了改进：其一是潜在动作的学习方式，其二是将潜在动作融入 VLA 预训练的方式。我们证明，villa-X 能够以 zero-shot 方式生成潜在动作规划，即便面对未见过的机器人本体（unseen embodiments）和开放词汇的符号理解（open-vocabulary symbolic understanding）任务，该能力同样适用。这一特性使得 villa-X 在多样化仿真任务中取得了卓越表现——具体体现在 SIMPLER 基准测试上——同时在两个真实机器人平台上（分别涉及夹爪操作和灵巧手操作）也展现出优异性能。上述结果表…
- 方法：改进潜在动作的表征学习方式；优化潜在动作与VLA预训练的融合机制
- 结果：在SIMPLER仿真平台实现卓越性能；在夹爪和灵巧手两类真实机器人任务中表现优异
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-y5CaJb17Fn-6ac90d96f8) · [原文](https://openreview.net/forum?id=y5CaJb17Fn)

### P2

#### COMI: Coarse-to-fine Context Compression via Marginal Information Gain

![COMI: Coarse-to-fine Context Compression via Marginal Information Gain](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-OGDIXDfaN4-7c6c53e9fc.webp)

- 导读：大型语言模型（LLMs）已在多样化任务中展现出卓越能力。然而，其在长上下文场景中的部署仍受制于计算效率低下与信息冗余问题。上下文压缩方法通过大幅缩减输入长度、消除冗余信息来应对上述挑战。我们提出 COMI——一种由粗到细的自适应上下文压缩框架，在高压缩率条件下联合优化语义相关性与多样性。我们引入边际信息增益（Marginal Information Gain，MIG）这一度量指标，其定义为某一单元与输入查询的相关性减去其与其他单元之间的语义冗余度，以此引导压缩过程优先保留既具相关性又低冗余的信息。该框架分两个阶段运行：（1）粗粒度组重分配（Coarse-Grained Group Reallocation）——将上下文划分为若干组，并依据组间 MIG 动态分配各组的压缩率，确保压缩预算与信息价值分布相匹配；（2）细粒度 Token 融合（Fine-Grained Token Merging）——利用基于组内 MIG 的加权机制对每组内的 token 进行融合，从而在保留关键语义的同时避免冗余信息的累积。我们在问答任务（如 NaturalQuestions、2WikiMQA、Hotpo…
- 方法：引入边际信息增益（MIG）作为压缩过程的指导指标。；粗粒度组重分配：根据组间MIG动态分配压缩率。
- 结果：在NaturalQuestions任务上，Qwen2-7B模型在32倍压缩约束下实现了约25分的精确匹配（EM）提升。；COMI在多种任务和骨干模型上大幅优于现有基线。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-OGDIXDfaN4-7c6c53e9fc) · [原文](https://openreview.net/forum?id=OGDIXDfaN4)

#### Composition of Memory Experts for Diffusion World Models

![Composition of Memory Experts for Diffusion World Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-sUEdpZCHdp-e92e8e2b7e.webp)

- 导读：世界模型旨在预测与过去观测一致的合理未来，这一能力是强化学习中规划与决策的核心。然而，现有架构面临一个根本性的记忆权衡困境：Transformer 能够保留局部细节，但受制于二次方复杂度的 attention 机制；而循环模型和状态空间模型虽然扩展效率更高，却以牺牲保真度为代价对历史信息进行压缩。为了克服这一权衡，我们提出将未来—过去一致性从单一架构中解耦，转而借助一组专业化的专家模型来实现这一目标。我们引入了一种基于 diffusion 的框架，通过对比式 product-of-experts 构造将异构记忆模型进行整合。我们的方法实例化了三种互补角色：一个短期记忆专家，用于捕捉精细的局部动态；一个长期记忆专家，通过轻量级测试时微调将情节性历史存储于外部 diffusion 权重之中；以及一个空间长期记忆专家，用于强化几何与空间一致性。这种组合式设计避免了模式坍塌（mode collapse），并能在不引入二次方代价的情况下扩展至长上下文场景。在仿真与真实世界 benchmark 上，我们的方法在时序一致性、过去观测的回溯召回以及导航性能方面均取得了提升，为构建和运行记忆增强型 d…
- 方法：采用扩散框架集成异构记忆模型，通过对比性专家乘积公式协调；设计短期记忆专家捕捉精细局部动态
- 结果：避免模式坍塌同时支持长上下文（无二次方计算成本）；在模拟和真实基准测试中提升时间一致性
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-sUEdpZCHdp-e92e8e2b7e) · [原文](https://openreview.net/forum?id=sUEdpZCHdp)

#### FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding

![FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-tPhcrP75OP-df41e33f50.webp)

- 导读：近期关于长视频理解的研究借助大型多模态模型（Large Multimodal Models，LMMs）先进的视觉-语言推理能力，推动了专门用于处理超长视频序列的 video-LMMs 的快速发展。然而，这类模型的可扩展性受到严重制约——超长视频序列所生成的海量视觉 token 给计算带来了难以承受的压力。为应对这一挑战，我们提出 FLoC，一种基于 facility location 函数的高效视觉 token 压缩框架。Facility location 函数是一种有原则的方法，能够在预定义的视觉 token 数量预算范围内，快速从原始 token 集合中筛选出一个紧凑而高度具有代表性且多样性充分的子集。通过集成 lazy greedy 算法，我们的方法在快速选取紧凑 token 子集的过程中实现了显著的效率提升，在大幅削减视觉 token 数量的同时，能够保证接近最优的性能表现。值得特别指出的是，我们的方法无需训练（training-free）、与模型无关（model-agnostic）、与查询无关（query-agnostic），提供了一种通用性极强的解决方案，能够与各类 vi…
- 方法：提出基于设施位置函数的视觉标记压缩框架FLoC。；在预定义预算内快速选择紧凑、代表性和多样化的视觉标记子集。
- 结果：大幅减少视觉标记数量，同时保证接近最优的性能。；在大规模基准测试上始终优于最近的压缩技术。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-tPhcrP75OP-df41e33f50) · [原文](https://openreview.net/forum?id=tPhcrP75OP)

#### MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding

![MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-5gLgjn3MVT-3bae051ec4.webp)

- 导读：大型语言模型（LLMs）的快速发展为多模态模型奠定了基础。然而，视觉语言模型（VLMs）在从图像扩展至视频领域时，仍面临显著的计算开销问题。当视频数据由于高帧率和较长时长而体积过大时，模型的推理成本将急剧上升，这严重阻碍了其在需要快速响应且计算资源有限的环境中的部署与应用。 对输入视频进行 token 压缩是一个颇具前景的研究方向，因为有效的压缩方案能够显著降低计算开销。现有的大多数压缩方法基于无需训练的 token 合并策略，在空间维度或时间维度上分别展开。尽管这些方法能够降低计算开销，但其无需训练的本质在 token 压缩过程中不可避免地导致信息损失，进而造成性能的显著下降。为应对上述挑战，我们提出了一种面向高效视频理解的基于记忆增强强化学习的 token 压缩方法——MARC（Memory-Augmented Reinforcement Learning-based Token Compression），该方法将结构化检索与基于强化学习（RL）的知识蒸馏融为一体。MARC 采用先检索后压缩的处理范式，结合了视觉记忆检索器（Visual Memory Retriever，VMR）…
- 方法：提出检索-压缩框架：先用视觉记忆检索器(VMR)分割视频并筛选关键片段；设计压缩组相对策略优化(C-GRPO)：通过强化学习实现教师网络到学生网络的蒸馏
- 结果：在6个基准测试中保持与64帧基线相当精度，视觉token减少95%；GPU内存使用降低72%，生成延迟减少23.9%
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-5gLgjn3MVT-3bae051ec4) · [原文](https://openreview.net/forum?id=5gLgjn3MVT)

#### Memento: Toward an All-Day Proactive Assistant for Ultra-Long Streaming Video

![Memento: Toward an All-Day Proactive Assistant for Ultra-Long Streaming Video](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-FtdbdoGbk3-73ac99ea3c.webp)

- 导读：多模态大型语言模型在视觉-语言理解领域展现出令人瞩目的能力，尤其在离线视频任务中表现突出。近年来，在线视频建模的兴起带来了主动交互的早期形态。然而，现有模型通常仅能处理数十分钟长度的视频，尚无法对超长视频流实现全天候主动理解。这些模型在在线场景下难以维持长期上下文，原因在于其面临 token 持续累积的问题，且缺乏可扩展的记忆机制。上述局限阻碍了一系列关键任务的实现——例如，提醒用户数小时前已服药这一能力，正是从被动响应式助手向具备长期推理能力的记忆导向型助手转变的典型体现。 为弥合这一差距，我们提出 Memento，这是首个面向超长流式视频的主动视觉-语言框架。为避免 token 数量随时间线性增长并支持可扩展的长时程理解，我们引入了 Dynamic Memory（动态记忆）与 Query-related Memory Selection（查询相关记忆选取）机制，实现稀疏记忆保留与高效检索。针对基于记忆的建模所带来的训练挑战，我们提出 Step-Aware Memory Attention（步骤感知记忆注意力），通过将记忆访问与时间步骤对齐来实现稳定的监督信号。为支持主动、长时程行…
- 方法：引入动态记忆和查询相关记忆选择，实现稀疏记忆保留和高效检索。；提出步骤感知记忆注意力，将记忆访问与时间步骤对齐以实现稳定的监督。
- 结果：Memento实现了卓越的性能，为可靠的全天候主动视频助手铺平了道路。；Memento-54K和MementoBench数据集-基准套件涵盖长达7小时视频流的多样化任务。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-FtdbdoGbk3-73ac99ea3c) · [原文](https://openreview.net/forum?id=FtdbdoGbk3)

#### Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity

![Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-i36E5Ezm0H-0c4d7af46a.webp)

- 导读：视觉语言模型（VLMs）因大量生成视觉 token 而面临严重的计算效率问题。尽管已有研究表明相当大比例的视觉 token 是冗余的，但现有压缩方法难以在重要性保留与信息多样性之间取得平衡。为此，我们提出 PruneSID——一种无需训练的协同重要性-多样性（Synergistic Importance-Diversity）方法，采用两阶段流水线：（1）主语义成分分析（Principle Semantic Components Analysis，PSCA），将 token 聚类为语义连贯的组，确保概念覆盖的全面性；（2）组内非极大值抑制（Intra-group Non-Maximum Suppression，NMS），在每组内剪除冗余 token 的同时保留关键的代表性 token。此外，PruneSID 还引入了一种信息感知动态压缩比机制，根据图像复杂度自适应地优化 token 压缩率，从而在多样化场景中实现更有效的平均信息保留。大量实验表明，该方法达到了最先进的性能：在 LLaVA-1.5 上仅保留 11.1% 的 token 即可实现 96.3% 的准确率；在 LLaVA-Ne…
- 方法：语义成分分析(PSCA)：通过聚类生成语义连贯的令牌组；组内非极大值抑制(NMS)：保留每组关键代表令牌并剪枝冗余
- 结果：LLaVA-1.5上仅保留11.1%令牌时准确率96.3%；LLaVA-NeXT极端压缩率(5.6%)下准确率92.8%
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-i36E5Ezm0H-0c4d7af46a) · [原文](https://openreview.net/forum?id=i36E5Ezm0H) · [代码](https://github.com/ZhengyaoFang/PruneSID)

#### QueryStream: Advancing Streaming Video Understanding with Query-Aware Pruning and Proactive Response

![QueryStream: Advancing Streaming Video Understanding with Query-Aware Pruning and Proactive Response](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-738HjJEbml-c010eb64ca.webp)

- 导读：在线视频场景中对实时交互日益增长的需求，促使研究者需要开发新一类高效的流式视频理解模型。然而，现有方法通常依赖于一种与查询无关的"变化即重要"假设，这一假设将视觉动态与语义相关性混为一谈，导致计算冗余以及响应时机不当。为此，我们提出 QueryStream——一种将查询感知能力融入视频处理与响应调度核心的新型框架。QueryStream 包含两个协同工作的模块：（1）Query-Aware Differential Pruning（QDP），该策略通过联合评估输入 token 流与查询之间的语义相关性，以及其相对于动态平滑历史的时序新颖性，对 token 流进行过滤；（2）Relevance-Triggered Active Response（RTAR），一种双门控机制，依据高查询相关性与显著信息密度两个条件共同触发响应调度。QueryStream 作为一个轻量级的免训练模块，在适度剪枝条件下，于 StreamingBench 和 OVO-Bench 等 benchmark 上实现了当前最优性能；在剪枝超过 70% 视觉 token 的情况下，仍能达到与保留全部 token 的基线模…
- 方法：查询感知差分剪枝(QDP)：联合评估语义相关性和时间新颖性来动态过滤令牌流；相关性触发主动响应(RTAR)：基于查询相关性和信息密度的双门控响应调度机制
- 结果：在StreamingBench和OVO-Bench基准测试中达到最先进性能；剪枝超过70%视觉令牌的同时匹配全令牌基线性能
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-738HjJEbml-c010eb64ca) · [原文](https://openreview.net/forum?id=738HjJEbml)

#### REMem: Reasoning with Episodic Memory in Language Agent

![REMem: Reasoning with Episodic Memory in Language Agent](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-fugnQxbvMm-e62f715ff0.webp)

- 导读：人类擅长在时空背景下记忆具体经历，并对这些事件进行跨时间的推理，这种能力即为情节记忆（episodic memory）。相比之下，语言智能体中的记忆主要停留在语义层面，现有智能体尚无法有效地回溯交互历史并对其进行推理。本文识别并形式化了这一差距所带来的情节回溯与推理的核心挑战，并发现现有工作往往忽视情节性特征、缺乏显式的事件建模，或过度强调简单检索而非复杂推理。我们提出 REMem——一个用于构建和推理情节记忆的两阶段框架：1）离线索引阶段，REMem 将经历转化为混合记忆图（hybrid memory graph），灵活地将具有时间感知的要点（gist）与事实（fact）关联起来；2）在线推理阶段，REMem 采用配备精心设计工具集的自主检索器（agentic retriever），对记忆图进行迭代式检索。在四个情节记忆 benchmark 上的综合评估表明，REMem 显著优于 Mem0 和 HippoRAG 2 等最先进的记忆系统，在情节回溯和推理任务上分别取得 3.4% 和 13.4% 的绝对性能提升。此外，REMem 针对无法回答的问题也展现出更为稳健的拒答行为。
- 方法：离线索引：将经历转换为混合记忆图，链接时间感知的要点和事实。；在线推理：采用代理检索器，配备工具进行迭代检索。
- 结果：REMem在情景回忆任务上比Mem0和HippoRAG 2提升了3.4%。；REMem在推理任务上比Mem0和HippoRAG 2提升了13.4%。
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-fugnQxbvMm-e62f715ff0) · [原文](https://openreview.net/forum?id=fugnQxbvMm)

#### RMAAT: Astrocyte-Inspired Memory Compression and Replay for Efficient Long-Context Transformers

![RMAAT: Astrocyte-Inspired Memory Compression and Replay for Efficient Long-Context Transformers](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-sTkJdbVxsI-e73aff2703.webp)

- 导读：自注意力机制的二次方复杂度对将 Transformer 模型应用于长序列构成了重大障碍。本研究探索从星形胶质细胞——一类对生物记忆和突触调制至关重要的胶质细胞——中提炼出的计算原理，将其作为高效自注意力传统架构改进方案的互补途径。我们提出循环记忆增强星形态 Transformer（Recurrent Memory Augmented Astromorphic Transformer，RMAAT），这是一种集成了抽象化星形胶质细胞功能的架构。RMAAT 采用基于片段的循环处理策略，通过持久记忆 token 传播上下文信息。一种自适应压缩机制对这些 token 进行调制，该机制由一个新颖的保留因子（retention factor）所控制，该因子源自对星形胶质细胞长时程增强（long-term plasticity，LTP）的模拟。片段内的注意力计算采用受星形胶质细胞短时程可塑性（short-term plasticity，STP）启发的高效线性复杂度机制。训练过程采用星形胶质细胞记忆回放反向传播算法（Astrocytic Memory Replay Backpropagation，AM…
- 方法：提出循环记忆增强星形胶质Transformer（RMAAT），整合抽象化的星形胶质细胞功能。；采用基于片段的循环处理策略，持久记忆令牌传播上下文信息。
- 结果：在长距离竞技场（LRA）基准测试中表现出竞争力的准确性。；计算和内存效率有显著提升。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-sTkJdbVxsI-e73aff2703) · [原文](https://openreview.net/forum?id=sTkJdbVxsI)

#### UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning

![UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-QWuXU0qNX0-521375583d.webp)

- 导读：Mixture of Experts（MoE）模型通过仅激活部分参数实现了显著的计算效率，但在推理阶段面临较高的内存访问开销。Memory-layer 架构以极低的内存访问次数提供了一种颇具吸引力的替代方案，然而此前的尝试（如 UltraMem）仅能匹配 2-expert MoE 模型的性能，与最先进的 8-expert 配置相比仍存在明显差距。本文提出 UltraMemV2，一种经过重新设计的 memory-layer 架构，旨在弥合这一性能差距。 我们的方法引入了五项关键改进：将 memory layer 集成至每个 Transformer block 中；采用单一线性投影简化 value expansion；从 PEER 借鉴基于 FFN 的 value processing 方式；实施有原则的参数初始化策略；以及重新调整 memory 与 FFN 之间的计算比例。通过大量实验评估，我们证明 UltraMemV2 在相同计算量和参数量的前提下，能够与 8-expert MoE 模型达到性能持平，同时内存访问开销显著更低。值得注意的是，UltraMemV2 在内存密集型任务上表现…
- 方法：在每个Transformer块中集成内存层；采用单线性投影简化值扩展
- 结果：达到与8专家MoE模型相当的性能（相同计算/参数）；长上下文记忆任务提升+1.6分
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-QWuXU0qNX0-521375583d) · [原文](https://openreview.net/forum?id=QWuXU0qNX0)

#### VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling

![VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/iclr2026-MUjdNcfNPv-658cf6598e.webp)

- 导读：长上下文视频建模对多模态大语言模型（MLLMs）至关重要，使其能够处理电影、在线视频流等长视频内容。尽管该领域已取得一定进展，但处理长视频依然面临严峻挑战——核心难点在于如何高效理解极长的视频上下文。本文从模型架构、训练数据、训练策略和评测基准四个维度出发，系统性地解决上述问题。 首先，我们提出一种新颖的分层视频token压缩方法（Hierarchical video token Compression，HiCo），该方法充分利用长视频中的视觉冗余性，将长视频上下文从Clip级逐步压缩至Video级，在显著降低计算量的同时保留关键细节，实现了约1/50的极端压缩比，且几乎不造成性能损失。其次，我们引入一套多阶段由短到长的学习方案（multi-stage short-to-long learning scheme），并构建了一个大规模真实世界长视频数据集LongVid，以及一个具有较高挑战性的评测基准——"Multi-Hop Needle-In-A-Video-Haystack"。该基准旨在考察模型在长视频中进行多跳信息检索与推理的能力。最后，我们基于上述方法构建了强大的视频多模态大语…
- 方法：分层视频令牌压缩(HiCo)：利用视觉冗余实现Clip级到Video级的50倍压缩；多阶段短到长学习方案：渐进式提升模型长视频处理能力
- 结果：实现约1/50的极端压缩比且几乎无性能损失；在10,000帧NIAH测试中达到99.1%准确率（开源模型首次）
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-MUjdNcfNPv-658cf6598e) · [原文](https://openreview.net/forum?id=MUjdNcfNPv) · [代码](https://github.com/OpenGVLab/VideoChat-Flash)

## ICML 2026

本轮收录 32 篇。

### P0

#### HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control

![HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-60897.webp)

- 导读：当前的视觉-语言-动作(VLA)模型在机器人操作方面表现出色，但在需要长期记忆和推理的非马尔可夫任务中往往表现不佳，因为它们依赖于即时观察。现有解决方案面临频率-能力悖论：高性能模型因速度过慢而无法实时控制，而快速模型又缺乏足够的推理能力。为解决这一架构错配问题，我们提出了HiMe（分层具身记忆框架），将具身智能解耦为高频执行器(Executor)、工作记忆哨兵(Sentry)和长期策略规划器(Planner)。我们还引入了一个基于跨模态语义模式和主动管理机制的动态知识系统，使机器人通过'增删改'操作保持记忆可塑性。这种分层设计有效平衡了实时执行与慢思考规划之间的冲突，显著提高了长周期任务的成功率。实验表明，该方法不仅优于平面记忆基线，还展现出基于人类偏好自我修正内部知识的新能力。
- 方法：三层架构解耦：高频执行器(Executor)、工作记忆哨兵(Sentry)、长期策略规划器(Planner)；基于跨模态语义模式的动态知识系统
- 结果：显著提升长周期任务成功率（优于平面记忆基线）；首次实现基于人类偏好的内部知识自我修正能力
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-60897) · [原文](https://icml.cc/virtual/2026/poster/60897)

#### RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies

![RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-65933.webp)

- 导读：记忆对于长期依赖历史的机器人操作至关重要。这类任务通常涉及重复动作计数或临时遮挡物体的操作。当前的视觉-语言-动作(VLA)模型开始引入记忆机制，但其评估仍局限于狭窄的非标准化场景，限制了系统性理解、比较和进展衡量。为此，我们推出**RoboMME**：一个用于评估和推进VLA模型在长期依赖历史场景中的大规模标准化基准。该基准包含16个操作任务，基于精心设计的分类法构建，评估时间、空间、对象和程序记忆。我们进一步开发了14种基于π₀.₅架构的记忆增强VLA变体，系统探索不同记忆表示与集成策略。结果表明，记忆表示的有效性高度依赖任务，每种设计在不同任务中展现出独特优势和局限。视频与代码详见https://anonymtest1.github.io
- 方法：构建包含16个任务的标准化基准RoboMME，覆盖时空/对象/程序记忆；开发14种基于π₀.₅架构的记忆增强VLA变体
- 结果：记忆有效性高度依赖任务类型（16类任务验证）；不同记忆设计在不同任务中表现优劣各异
- 对 VPWEM：直接评测依据：可用于补充记忆任务与指标。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65933) · [原文](https://icml.cc/virtual/2026/poster/65933)

#### Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action

![Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-66214.webp)

- 导读：我们提出了SOMA（空间记忆框架），用于增强视觉-语言-动作（VLA）模型在目标物体超出视野时的操作能力。现有VLA模型通常假设任务相关物体始终可见，导致目标移出摄像头视野时行为脆弱且被动。SOMA通过为VLA配备基于可移动头部摄像头多视角观测构建的持久空间记忆，实现了超越当前视锥的空间推理。该框架包含三个组件：通过扫描将角度观测聚合为统一空间语义表征的『空间记忆构建』、保持全局一致性的『动态记忆优化』，以及在操作时激活指令相关空间线索的『上下文记忆检索』。我们在五个真实世界非视野操作任务（包括目标初始不可见的多步骤和双臂场景）上评估SOMA。实验表明SOMA不仅提高任务成功率，还引发质变的行为模式：目标定位更快、视角搜索减少，在部分可观测条件下接近一次性抓取。在RoboCasa GR1和SimplerEnv上的附加实验进一步验证了SOMA记忆设计在常规全观测场景下的有效性。
- 方法：通过可移动头部摄像头扫描构建统一空间语义表征；动态优化机制保持记忆的全局一致性
- 结果：在五个非视野操作任务中显著提高成功率；目标定位速度提升（定量结果未明确）
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-66214) · [原文](https://icml.cc/virtual/2026/poster/66214)

### P1

#### Cross-Embodiment Robot Foundation World Models with Latent Actions

![Cross-Embodiment Robot Foundation World Models with Latent Actions](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-63978.webp)

- 导读：机器人形态和动作空间的多样性使得构建能够泛化到不同形态的世界模型具有挑战性。我们提出了一种潜在动作条件机器人世界模型(LAC-WM)，该模型在学习的统一潜在动作空间中运行，该空间在不同形态间共享。我们展示了这种统一动作空间如何提升世界模型在适应未见过的机器人形态时的性能。我们将LAC-WM与基于显式运动标签的基准模型(EAC-WM)进行比较。结果表明，基于显式标签的条件化会在不同形态间创建分离的动作空间，从而限制适应新机器人时的下游任务性能。我们在灵巧操作任务上评估了两种模型。潜在动作条件模型LAC-WM比EAC-WM性能提升高达46.7%。关键的是，统一的潜在动作空间使得LAC-WM的下游性能随着预训练中使用的形态数量增加而正向提升。相比之下，EAC-WM中分离的动作空间随着预训练形态数量的增加导致性能下降。这些结果突显了统一动作空间对于高效跨形态学习的重要性，解决了机器人学中的一个关键挑战。
- 方法：提出潜在动作条件机器人世界模型(LAC-WM)；在学习的统一潜在动作空间中运行
- 结果：LAC-WM比EAC-WM性能提升高达46.7%；统一动作空间使下游性能随预训练形态数量增加而提升
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-63978) · [原文](https://icml.cc/virtual/2026/poster/63978)

#### Sample from What You See: Visuomotor Policy Learning via Diffusion Bridge with Observation-Embedded Stochastic Differential Equation

![Sample from What You See: Visuomotor Policy Learning via Diffusion Bridge with Observation-Embedded Stochastic Differential Equation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-61138.webp)

- 导读：基于扩散模型的模仿学习通过捕捉多模态动作分布推动了机器人控制的发展。然而现有方法通常仅将观测作为去噪网络的高层条件，而未将其整合到扩散过程的随机动力学中，导致采样被迫从随机噪声开始，削弱了感知与控制的耦合，并常导致次优性能。我们提出BridgePolicy，一种通过扩散桥公式直接将观测整合到随机动力学中的视觉运动生成策略。通过构建观测信息化的轨迹，BridgePolicy使采样能从丰富信息化的先验开始而非随机噪声，显著提升了控制精度和可靠性。关键难点在于扩散桥通常连接维度匹配的分布，而机器人观测具有异构性且与动作自然不对齐。为此，我们引入多模态融合模块和语义对齐器来统一视觉与状态输入，并将观测与动作表示对齐，使扩散桥能适用于异构机器人数据。在三个基准的52项仿真任务和5项现实任务中的大量实验表明，BridgePolicy始终优于最先进的生成策略。
- 方法：提出扩散桥公式BridgePolicy，将观测直接整合到随机动力学中；构建观测信息化的轨迹作为采样起点（替代随机噪声）
- 结果：在3个基准的52项仿真任务中持续超越现有最佳生成策略；5项现实世界任务验证了方法的有效性
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61138) · [原文](https://icml.cc/virtual/2026/poster/61138)

#### WestWorld: A Knowledge-Encoded Scalable Trajectory World Model for Diverse Robotic Systems

![WestWorld: A Knowledge-Encoded Scalable Trajectory World Model for Diverse Robotic Systems](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-61703.webp)

- 导读：轨迹世界模型在机器人动力学学习、规划与控制中扮演关键角色。尽管近期研究探索了面向多样化机器人系统的轨迹世界模型，但它们难以扩展到大量不同系统动力学，且忽视了物理结构的领域知识。为解决这些局限，我们提出了WestWorld——一种面向多样化机器人系统的知识编码可扩展轨迹世界模型。针对可扩展性挑战，我们提出了一种新颖的系统感知混合专家模型（Sys-MoE），通过可学习的系统嵌入动态组合和路由针对不同机器人系统的专用专家。为进一步增强零样本泛化能力，我们通过引入结构嵌入将轨迹表征与形态信息对齐，从而融入机器人物理结构的领域知识。在涵盖仿真和真实场景的89个复杂环境上预训练后，WestWorld在零样本和少样本轨迹预测上显著超越竞争基线，同时展现出对广泛机器人环境的强大可扩展性，并显著提升不同机器人下游基于模型的控制性能。最后，我们将模型部署在真实世界的Unitree Go1机器人上，其表现出稳定的运动性能（详见网站演示）。
- 方法：提出系统感知混合专家模型（Sys-MoE），通过可学习系统嵌入动态路由专用专家；引入结构嵌入对齐轨迹表征与机器人形态信息以增强零样本泛化
- 结果：在零样本和少样本轨迹预测上显著超越竞争基线；展现出对广泛机器人环境的强大可扩展性
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61703) · [原文](https://icml.cc/virtual/2026/poster/61703)

#### DreamDojo: A Real-Time Robot World Model from Large-Scale Human Videos

![DreamDojo: A Real-Time Robot World Model from Large-Scale Human Videos](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-65193.webp)

- 导读：能够模拟不同环境中的动作结果将彻底改变规模化通用智能体的开发。然而，由于数据覆盖有限和动作标签稀缺，对这些世界动态的建模（尤其是灵巧机器人任务）提出了重大挑战。为此，我们推出DreamDojo——一个从4.4万小时人类第一视角视频中学习多样化交互与灵巧控制的基础世界模型。我们的混合数据集是迄今最大的世界模型预训练视频数据集，涵盖广泛日常场景与多样物体技能。针对动作标签稀缺问题，我们引入连续潜在动作作为统一代理动作，增强未标注视频的交互知识迁移。经过小规模目标机器人数据微调后，DreamDojo展现出强大的物理理解与精确动作控制能力。我们还设计了蒸馏流程，将模型加速至10.93 FPS实时速度并进一步提升上下文一致性。该工作实现了基于生成世界模型的多个重要应用，包括实时遥操作、策略评估和基于模型的规划。在多个挑战性分布外基准上的系统评估验证了该方法对开放世界高接触任务模拟的重要意义，为通用机器人世界模型铺平道路。
- 方法：从4.4万小时第一视角人类视频构建最大规模世界模型预训练数据集；引入连续潜在动作作为统一代理动作解决未标注视频的知识迁移
- 结果：在微调后展示出卓越的物理理解与精确动作控制能力；实现10.93 FPS实时推理速度（比原始模型快27倍）
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65193) · [原文](https://icml.cc/virtual/2026/poster/65193)

#### Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning

![Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-64055.webp)

- 导读：尽管视觉-语言-动作（VLA）模型在机器人控制方面取得了快速进展，但在长期任务中，指令漂移仍然是一个持续存在的失败模式。本文重新概念化了这一现象，认为指令漂移本质上是一种系统性采样误差：局部贪婪采样容易陷入“负关键窗口”——具有高局部概率的不可逆局部最优，从而切断全局成功路径。为了解决这一问题，我们提出了 extbf{上下文感知功率采样（CAPS）}，一种无需训练、在推理时计算框架。CAPS利用功率分布来锐化全局轨迹概率，有效激活模型的隐式世界模型以进行前瞻规划。此外，我们引入了一种基于信噪比（SNR）的元认知控制机制。该机制仅在检测到漂移风险时触发自适应MCMC搜索，从而实现从“直觉快速思考”到“理性慢速搜索”的动态过渡。在RoboTwin、Simpler-WindowX和Libero-long基准测试中的实验表明，CAPS显著优于SOTA基线，如OpenVLA和TACO，且无需参数更新。这些结果证实，自适应推理时计算是增强具身长期鲁棒性的有效途径。
- 方法：提出上下文感知功率采样（CAPS）框架。；利用功率分布锐化全局轨迹概率。
- 结果：CAPS显著优于SOTA基线，如OpenVLA和TACO。；无需参数更新即可实现性能提升。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-64055) · [原文](https://icml.cc/virtual/2026/poster/64055)

#### DriveWorld-VLA: Unified Latent-Space World Modeling with Vision–Language–Action for Autonomous Driving

![DriveWorld-VLA: Unified Latent-Space World Modeling with Vision–Language–Action for Autonomous Driving](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-61191.webp)

- 导读：端到端（E2E）自动驾驶最近引起了越来越多的关注，旨在将视觉-语言-动作（VLA）与世界模型统一起来，以增强决策和前瞻性想象力。然而，现有方法由于潜在状态共享不足，无法在单一架构内有效统一未来场景演变和动作规划，限制了视觉想象力对动作决策的影响。为了解决这一局限，我们提出了DriveWorld-VLA，这是一个新颖的框架，通过在表示层面紧密集成VLA和世界模型，将世界建模和规划统一在潜在空间中，使VLA规划器能够直接从整体场景演变建模中受益，并减少对密集注释监督的依赖。此外，DriveWorld-VLA将世界模型的潜在状态作为VLA规划器的核心决策状态，促进规划器评估候选动作如何影响未来场景演变。通过在潜在空间中完全进行世界建模，DriveWorld-VLA支持在特征级别上进行可控的动作条件想象，避免了昂贵的像素级展开。大量的开环和闭环评估证明了DriveWorld-VLA的有效性，在NAVSIMv1上达到了91.3 PDMS，在NAVSIMv2上达到了86.8 EPDMS，在nuScenes上达到了0.16的3秒平均碰撞率。代码和模型将公开发布。
- 方法：在表示层面紧密集成VLA和世界模型，将世界建模和规划统一在潜在空间中。；将世界模型的潜在状态作为VLA规划器的核心决策状态。
- 结果：在NAVSIMv1上达到了91.3 PDMS。；在NAVSIMv2上达到了86.8 EPDMS。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61191) · [原文](https://icml.cc/virtual/2026/poster/61191)

#### FocalPolicy: Frequency-Optimized Chunking and Locally Anchored Flow Matching for Coherent Visuomotor Policy

![FocalPolicy: Frequency-Optimized Chunking and Locally Anchored Flow Matching for Coherent Visuomotor Policy](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-66520.webp)

- 导读：视觉运动策略旨在从专家演示中学习复杂操作任务。然而，生成平滑连贯的轨迹仍具挑战性，因其需平衡近端精度与远端预见性。现有方法通常聚焦于优化块内动作分布，常忽略块间连贯性，导致块间不连续性严重阻碍连贯长程动作的学习。为突破此限制并实现精度与预见性的协同平衡，我们提出FocalPolicy——一种结合频率优化分块与局部锚定流匹配的预见感知视觉运动策略。我们引入预见性复合目标函数，在监督近端动作时域对齐的同时，通过正则化多未来动作块的频域结构来提升跨块连贯性。为高效学习复杂动作分布，我们设计局部锚定采样以增强一致性流匹配训练中的目标信号传播效率。大量实验表明，本方法持续优于现有方案，并进一步验证所提模块对其他基线模型的有效泛化性。项目将开源发布。
- 方法：频率优化分块：在频域正则化多未来动作块结构；局部锚定采样：提升流匹配训练中的目标信号传播效率
- 结果：在连贯长程动作生成上持续超越现有方法；模块可有效泛化至其他基线模型（实验验证）
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-66520) · [原文](https://icml.cc/virtual/2026/poster/66520)

#### From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation

![From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-66091.webp)

- 导读：视频生成模型通过预测长期未来观测为机器人操作提供了一种有前景的想象机制，但如何有效利用这些想象的未来进行动作执行仍具挑战性。现有方法要么基于预测帧条件化策略，要么直接将生成的视频解码为动作，两者都存在视觉真实性与控制相关性之间的不匹配问题。这导致预测观测更强调感知保真度而非状态转换的动作中心原因，从而产生间接且不稳定的控制。为解决这一问题，我们提出了MoLA（潜在动作混合），这是一种面向控制的接口，将想象的未来视频转化为可执行表示。MoLA不直接将预测帧传递给策略，而是利用多个预训练的逆动力学模型推断由生成视觉转换隐含的潜在动作混合。这些模态感知的逆动力学模型捕捉互补的语义、深度和流线索，提供了一种结构化和物理基础的动作表示，从而桥接视频想象与策略执行。我们在模拟基准（LIBERO、CALVIN和LIBERO-Plus）和真实世界机器人操作任务上评估了我们的方法，在任务成功率、时间一致性和泛化性方面取得了持续提升。代码将公开。
- 方法：提出MoLA接口，将生成的未来视频转化为可执行表示；利用多个预训练逆动力学模型推断潜在动作混合
- 结果：在LIBERO、CALVIN和LIBERO-Plus模拟基准上取得持续提升；提高真实世界机器人操作任务的成功率
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-66091) · [原文](https://icml.cc/virtual/2026/poster/66091)

#### From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models

![From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-63621.webp)

- 导读：潜在动作作为一种中间表示，能够在异构数据集上实现视觉-语言-动作（VLA）模型的一致性建模。然而，使用潜在动作监督VLA的方法较为分散，缺乏系统性比较。本研究从两个角度构建了潜在动作监督的研究框架：（i）通过基于图像的潜在动作对轨迹进行正则化，（ii）通过基于动作的潜在动作统一目标空间。在一个统一的VLA基线模型下，我们实例化并比较了四种代表性的集成策略。研究结果揭示了公式与任务的对应关系：基于图像的潜在动作有利于长时程推理，而基于动作的潜在动作在复杂运动协调方面表现更佳。此外，我们发现直接用离散的潜在动作标记监督VLM能够获得最有效的性能。最后，我们的实验初步揭示了潜在动作监督在混合数据中的优势，为VLA训练提供了一个有前景的方向。
- 方法：通过基于图像的潜在动作对轨迹进行正则化。；通过基于动作的潜在动作统一目标空间。
- 结果：基于图像的潜在动作有利于长时程推理。；基于动作的潜在动作在复杂运动协调方面表现更佳。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-63621) · [原文](https://icml.cc/virtual/2026/poster/63621)

#### LARA: Latent Action Representation Alignment for Vision-Language-Action Models

![LARA: Latent Action Representation Alignment for Vision-Language-Action Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-61232.webp)

- 导读：视觉语言动作（VLA）模型使机器人能够直接从观察和语言指令中预测动作，但其性能依赖于大规模、高质量的数据，并受到现实世界机器人动作数据集稀缺的限制。为了利用大量未标记的人类视频促进VLA模型学习，潜在动作模型（LAM）从视觉动态中学习潜在动作表示，为VLA学习提供额外的监督。然而，LAM和VLA通常分开训练，导致LAM在VLA训练期间缺乏基础，而VLA模型则受到冻结的LAM表示的限制。为了解决这些问题，我们提出了潜在动作表示对齐（LARA），一个即插即用的框架，通过表示对齐联合优化LAM和VLA。这使得LAM能够通过动作轨迹学习以避免虚假的视觉变化，而VLA则通过LAM中学习的前向动态进行正则化，以减少功能无效轨迹的幻觉。我们展示了LARA在预训练、预训练VLA模型的后训练增强以及LAM精炼中的多功能性和有效性，在3个模拟和1个精心设计的现实世界机器人操作基准上平均实现了约10%、5%和15%的改进。
- 方法：提出了潜在动作表示对齐（LARA）框架，联合优化LAM和VLA。；通过表示对齐，使LAM能够通过动作轨迹学习以避免虚假的视觉变化。
- 结果：在3个模拟和1个现实世界机器人操作基准上，LARA平均提升了约10%的性能。；在预训练VLA模型的后训练增强中，LARA平均提升了约5%的性能。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61232) · [原文](https://icml.cc/virtual/2026/poster/61232)

#### LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries

![LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-65457.webp)

- 导读：视觉-语言-动作(VLA)模型在机器人操作中展现出潜力，但常难以泛化至新指令或复杂多任务场景。我们发现当前训练范式存在一个关键缺陷：目标驱动的数据收集会导致数据集偏差。在此类数据集中，仅凭视觉观察即可高度预测语言指令，导致指令与动作间的条件互信息消失，这一现象我们称为'信息坍缩'。其结果是模型退化为仅依赖视觉的策略而忽略语言约束。为此，我们提出LangForce，通过贝叶斯分解强制遵循指令。通过引入可学习的潜在动作查询，我们构建了双分支架构来同时估计仅视觉先验$p(a \mid v)$和语言条件后验$\pi(a \mid v, \ell)$。随后优化策略以最大化动作与指令间的条件点互信息(PMI)，该目标有效惩罚视觉捷径，并奖励能显式解释语言命令的动作。在三个基准测试上的大量实验表明显著提升，包括在挑战性的OOD SimplerEnv基准上11.3%的改进，验证了LangForce稳健地将语言扎根于动作的能力。
- 方法：提出可学习的潜在动作查询构建双分支架构；同时估计视觉先验$p(a \mid v)$和语言条件后验$\pi(a \mid v, \ell)$
- 结果：在OOD SimplerEnv基准上提升11.3%；有效惩罚视觉捷径策略
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65457) · [原文](https://icml.cc/virtual/2026/poster/65457)

#### MVISTA-4D: View-Consistent 4D World Model with Test-Time Action Inference for Robotic Manipulation

![MVISTA-4D: View-Consistent 4D World Model with Test-Time Action Inference for Robotic Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-65571.webp)

- 导读：基于世界模型的‘想象后行动’范式在机器人操控领域展现出巨大潜力，但现有方法通常仅支持纯图像预测或部分3D几何推理，限制了其预测完整4D场景动态的能力。本研究提出了一种新颖的具身4D世界模型，能够实现几何一致的任意视角RGBD生成：仅需单视角RGBD观测作为输入，模型即可‘想象’剩余视角，并通过反向投影和融合构建跨时间的更完整3D结构。为高效学习多视角跨模态生成，我们显式设计了跨视角和跨模态特征融合机制，联合促进RGB与深度的一致性并强制多视角几何对齐。除预测外，将生成的未来转换为动作通常由逆动力学处理，但由于同一状态转移可能对应多种动作，该问题具有不适定性。我们通过测试时动作优化策略（通过生成模型反向传播以推断与预测未来最匹配的轨迹级潜变量）和残差逆动力学模型（将该轨迹先验转化为可执行动作）来解决此问题。在三个数据集上的实验证明了该方法在4D场景生成和下游操控任务中的优异性能，消融研究则为关键设计选择提供了实用见解。
- 方法：设计跨视角与跨模态特征融合机制强制RGB-D一致性；通过单视角RGBD输入生成任意视角并反向投影构建4D场景
- 结果：在三个数据集上验证了4D场景生成与操控任务的优越性能；消融实验证实跨模态特征融合提升几何一致性
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65571) · [原文](https://icml.cc/virtual/2026/poster/65571)

#### OXE-AugE: A Large-Scale Robot Augmentation of OXE for Scaling Cross-Embodiment Policy Learning

![OXE-AugE: A Large-Scale Robot Augmentation of OXE for Scaling Cross-Embodiment Policy Learning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-64619.webp)

- 导读：训练通用机器人策略需要大规模多样化数据集,以控制不同任务和环境下的各种机器人构型(机械臂与夹具组合)。由于为每个新硬件平台重新收集演示数据并重新训练成本过高,我们证明可以通过增强现有机器人数据实现迁移与泛化。Open X-Embodiment (OXE)数据集汇集了60多个机器人数据集,已被广泛用作训练通用策略的基础。但该数据集存在严重不平衡问题:前四种机器人类型占比超过85%,可能导致对特定机器人-场景组合的过拟合。我们提出AugE-Toolkit可扩展机器人增强流程,以及OXE-AugE高质量开源数据集,通过9种不同机器人构型扩展OXE。OXE-AugE提供超过440万条轨迹,是原始OXE的三倍多。我们系统研究了机器人增强规模对跨构型学习的影响。结果表明,用多样化机械臂和夹具增强数据集不仅能提升增强机器人的策略性能,还能提升未见机器人甚至原始机器人在分布变化下的性能。物理实验证明,OpenVLA和π0等通用策略通过OXE-AugE微调后,在四种现实操作任务中对未见机器人-夹具组合的成功率提升24-45%。
- 方法：提出AugE-Toolkit可扩展机器人增强流程；构建OXE-AugE数据集(9种新机器人构型)
- 结果：OXE-AugE包含440万条轨迹(原始OXE的三倍)；增强策略在未见机器人上成功率提升24-45%
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-64619) · [原文](https://icml.cc/virtual/2026/poster/64619)

#### Predicting What Matters: Robust Generalist Robot Policy Learning via Future Semantic Mask

![Predicting What Matters: Robust Generalist Robot Policy Learning via Future Semantic Mask](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-65548.webp)

- 导读：源自大规模视频生成预训练的世界模型已成为通用机器人策略学习的有前景范式。然而，标准方法通常专注于高保真RGB视频预测，这可能导致对无关因素（如动态背景和光照变化）的过拟合。这些干扰会降低模型的泛化能力，最终导致控制策略不可靠且脆弱。为解决这一问题，我们引入了掩码世界模型（MWM），该模型利用视频扩散架构预测语义掩码而非像素的演变。这一转变施加了几何信息瓶颈，迫使模型捕捉关键物理动态和接触关系，同时过滤视觉噪声。我们将此掩码动态主干与基于扩散的策略头无缝集成，以实现稳健的端到端控制。大量实验证明，MWM在LIBERO和RLBench仿真基准测试中优于现有基于RGB的世界模型。此外，真实世界实验和鲁棒性评估（通过随机令牌修剪）表明，MWM展现出卓越的泛化能力，并对纹理信息丢失具有强健的适应力。
- 方法：提出掩码世界模型（MWM），通过视频扩散架构预测语义掩码演变；利用几何信息瓶颈强制模型聚焦关键物理动态和接触关系
- 结果：在LIBERO和RLBench基准测试中显著优于现有基于RGB的世界模型；真实实验显示对纹理信息丢失具有强健适应力
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65548) · [原文](https://icml.cc/virtual/2026/poster/65548)

#### Reflex: Real-Time Vision-Language-Action Control through Streaming Inference

![Reflex: Real-Time Vision-Language-Action Control through Streaming Inference](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-63338.webp)

- 导读：流匹配视觉-语言-动作（VLA）模型有望实现精确的连续控制，但其迭代去噪特性与实时机器人技术存在根本性不兼容：全局时间步注入使KV缓存失效，迫使在缓慢的$O(N^2)$重新计算或数学上不正确的缓存重用之间做出选择。我们提出了\textbf{Reflex}框架，通过利用\textit{时间步不变性特性}（即感知编码器在功能上独立于去噪循环），实现了流匹配策略的\textit{实时流式推理}。Reflex将注意力上下文划分为静态、滑动和动态区域，实现$O(1)$增量缓存更新，保证输出与全批量推理完全相同。为确保连续高频推理下的稳定性，我们引入了\textit{AdaRMSNorm}，一种自适应归一化层，通过基于流相位门控防止BFloat16数值崩溃。我们进一步通过\textit{异步流水线}（将视觉编码与动作生成解耦）和\textit{算子融合}（减少内核开销）最大化吞吐量。在LIBERO和Kinetix基准测试中，Reflex实现了2.58$\times$的推理加速和50Hz稳定流式传输，反应延迟降低高达54\%，且无需性能损失即可高效部署。
- 方法：利用时间步不变性特性分离感知编码与去噪循环；将注意力上下文划分为静态/滑动/动态区域实现$O(1)$增量缓存
- 结果：推理速度提升2.58倍；实现50Hz稳定流式传输
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-63338) · [原文](https://icml.cc/virtual/2026/poster/63338)

#### RoboFlow4D: A Lightweight Flow World Model Toward Real-Time Flow-Guided Robotic Manipulation

![RoboFlow4D: A Lightweight Flow World Model Toward Real-Time Flow-Guided Robotic Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-62543.webp)

- 导读：在三维环境中规划和行动是实现现实世界机器人操控的基础能力。尽管先前研究已探索使用预测性流规划器指导3D操控，但现有方法通常依赖堆叠多个子模型的模块化流程，导致计算开销高且实时性能有限。为解决这些问题，我们提出了RoboFlow4D——一种轻量级流世界模型，通过估计物理3D空间的时序运动统一感知与规划。作为端到端框架，RoboFlow4D可直接从视觉观察和文本指令预测多帧3D流，提供显式的基于流规划来指导动作生成。该设计可与通用动作策略无缝集成，形成高效的观察-规划-执行闭环。通过流预测与动作控制的快慢协作，RoboFlow4D实现了实时且资源高效的操控。大量仿真和真实场景实验表明，RoboFlow4D持续提升了操控成功率和计算效率，推动了具身智能的流引导规划发展。
- 方法：构建端到端轻量级模型直接预测多帧3D流；通过视觉观察+文本指令联合输入实现显式流规划
- 结果：显著提升操控成功率（实验数据未具体说明）；计算效率优于模块化流程方法
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-62543) · [原文](https://icml.cc/virtual/2026/poster/62543)

#### Scaling Real-World Robot Policy Evaluation via Discrete Diffusion World Model

![Scaling Real-World Robot Policy Evaluation via Discrete Diffusion World Model](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-65898.webp)

- 导读：评估通用机器人操作策略在现实世界中成本高昂且难以扩展。虽然新兴的世界模型（如WorldEval、Ctrl-World）提供了一种有前景的替代方案，但其评估的可靠性仍是一个关键瓶颈。具体而言，它们的视觉预测可能通过将失败“自我纠正”为假阳性或在分布外控制下产生伪影，从而破坏策略评估。即使使用富含失败的数据，当前架构仍难以捕捉动作-因果动态，因为它们通常将动作视为被动条件而非因果驱动因素。为此，我们提出了dWorldEval，一种以动作为中心的离散扩散世界模型，它将视觉观察、语言指令和动作块映射到一个共享的统一标记空间，并通过单一自注意力主干进行去噪，其中动作作为一等标记。为实现可靠的策略-世界交互，dWorldEval引入了稀疏关键帧记忆，锚定全局场景状态的同时保留细粒度多视图交互线索，并利用“进展即文本”联合生成未来观察和成功指标。在LIBERO、RoboTwin和真实机器人任务上的大量实验表明，dWorldEval在动作可控性上显著优于视频扩散基线，稳定了长时程多视图推演，从而通过自动成功估计实现准确的策略排序。
- 方法：将视觉观察、语言指令和动作块映射到共享的统一标记空间，通过单一自注意力主干去噪。；引入稀疏关键帧记忆锚定全局场景状态，保留细粒度多视图交互线索。
- 结果：在LIBERO、RoboTwin和真实机器人任务上显著优于视频扩散基线。；动作可控性提升，稳定了长时程多视图推演。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65898) · [原文](https://icml.cc/virtual/2026/poster/65898)

#### Structured 4D Latent World Model for Robot Planning

![Structured 4D Latent World Model for Robot Planning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-63054.webp)

- 导读：学习型世界模型正成为机器人学中的一种强大范式，为任务泛化、长时程规划和灵活决策提供了有前景的路径。然而，现有方法通常基于2D视频序列运行，本质上缺乏精确空间推理和物理一致性所需的3D几何理解。我们提出了一种结构化4D潜在世界模型，该模型在观测和文本指令条件下，于结构化潜在空间中预测场景3D结构的演变。我们的表示对场景进行整体编码，并可解码为多种3D格式，实现更完整且物理一致的场景理解。该结构化4D潜在世界模型作为规划器，生成未来场景，并通过目标条件逆向动力学模块转化为可执行动作。实验表明，与最先进的基于视频的规划器相比，我们的模型生成的未来场景具有更优的视觉质量、物理一致性和多视角连贯性。因此，我们的完整规划流程在复杂操作任务中表现卓越，对新视觉条件展现出鲁棒泛化能力，并在真实机器人平台上验证有效。
- 方法：提出结构化4D潜在世界模型预测3D场景演变；通过文本指令和观测条件化潜在空间
- 结果：生成场景的视觉质量超越现有视频规划器；实现更好的物理一致性和多视角连贯性
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-63054) · [原文](https://icml.cc/virtual/2026/poster/63054)

#### Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Models

![Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-60629.webp)

- 导读：视觉-语言-动作（VLA）模型在机器人控制中展现出强大的泛化能力，但通过强化学习（RL）对其进行微调时，面临现实世界交互的高成本和安全风险限制。在交互式世界模型中训练VLA模型可避免这些问题，但也引入了像素级世界建模、多视角一致性以及稀疏奖励下的误差累积等挑战。基于大型多模态模型和基于模型的RL的最新进展，我们提出VLA-MBPO框架，以解决VLA微调中的这些问题。我们的方法包含三个关键设计：（i）采用统一多模态模型（UMMs）实现数据高效的世界建模；（ii）通过交错视角解码机制确保多视角一致性；（iii）采用分块级分支展开策略缓解误差累积。理论分析和仿真及真实任务实验表明，VLA-MBPO显著提升了策略性能和样本效率。值得注意的是，该方法在所有任务中保持统一的超参数集，突显了其在真实机器人部署中的鲁棒性和可扩展性。
- 方法：采用统一多模态模型（UMMs）实现数据高效的世界建模；设计交错视角解码机制确保多视角一致性
- 结果：显著提升策略性能与样本效率（实验验证）；所有任务使用统一超参数集，展现强鲁棒性
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-60629) · [原文](https://icml.cc/virtual/2026/poster/60629)

#### VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model

![VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-66169.webp)

- 导读：本文旨在通过迭代在线交互提升视觉-语言-动作（VLA）模型的性能与可靠性。由于在现实世界中收集策略部署数据成本高昂，我们探究是否可通过学习型模拟器（具体为动作条件视频生成模型）生成额外部署数据。然而，现有世界模型缺乏策略改进所需的物理保真度：它们主要基于缺乏多样化物理交互（尤其是失败案例）的演示数据集训练，且难以精准建模接触密集型物体操作中的微小但关键的物理细节。我们提出一种简单迭代改进算法，利用真实世界部署数据提升世界模型的保真度，进而生成合成数据以增强VLA模型。在真实机器人实验中，该方法将前沿VLA模型在多项下游任务中的性能提升39.2%（绝对成功率），其中合成部署数据贡献了11.6%的增益。视频见匿名网站：https://sites.google.com/view/vla-w。
- 方法：提出迭代改进算法：用真实部署数据提升世界模型保真度；利用改进后的世界模型生成合成部署数据
- 结果：VLA模型绝对任务成功率提升39.2%；合成部署数据贡献11.6%的性能增益
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-66169) · [原文](https://icml.cc/virtual/2026/poster/66169)

### P2

#### Episodic Memory-Guided Controllable Experience Synthesis for Reinforcement Learning

![Episodic Memory-Guided Controllable Experience Synthesis for Reinforcement Learning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-61811.webp)

- 导读：在现实场景中，强化学习（RL）的数据收集常受安全性和高成本限制，导致数据可用性不足。扩散模型（DM）近期在捕捉复杂分布方面展现出卓越能力，使数据增强成为可行方案。然而，现有基于DM的数据增强方法仍面临合成数据质量不足的问题。为此，我们提出了一种名为'情景记忆引导可控经验合成器（EMCES）'的新方法。EMCES通过情景记忆（EM）构建信息丰富且简洁的条件，结合基于情景记忆的可控DM。为引导合成高质量数据，我们提出了一种EM优先条件采样策略，利用基于EM的时间差分误差聚焦于对RL最有帮助的数据生成。此外，我们引入基于哈希的状态表示以提高EM效率并进一步提升合成数据质量。据我们所知，EMCES是首个将EM融入可控DM并利用EM引导RL数据合成的工作。多环境实验表明，EMCES显著提升了合成数据质量，从而改进了多种前沿RL算法的性能。
- 方法：提出情景记忆引导可控经验合成器（EMCES）框架；基于情景记忆构建信息丰富的可控扩散模型条件
- 结果：在多环境中显著提升合成数据质量；改进多种前沿RL算法性能
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61811) · [原文](https://icml.cc/virtual/2026/poster/61811)

#### Learning to Share: Selective Memory for Efficient Parallel Agentic Systems

![Learning to Share: Selective Memory for Efficient Parallel Agentic Systems](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-62890.webp)

- 导读：智能体系统通过协调多个智能体进行迭代推理、调用工具和交换中间结果来解决复杂任务。为了提高鲁棒性和解决方案质量，最近的方法部署了多个并行运行的智能体团队，以探索多样化的推理轨迹。然而，并行执行带来了显著的计算成本：当不同团队独立推理相似的子问题或执行类似的步骤时，它们会重复执行大量重叠的计算。为了解决这些限制，本文提出了学习共享（LTS），一种用于并行智能体框架的学习共享内存机制，能够在控制上下文增长的同时实现跨团队信息的选择性重用。LTS引入了一个所有团队都可以访问的全局内存库和一个轻量级控制器，该控制器决定是否将智能体的中间步骤添加到内存中。控制器通过逐步强化学习和使用感知的信用分配进行训练，使其能够识别在并行执行中全局有用的信息。在AssistantBench和GAIA基准测试上的实验表明，与无内存并行基线相比，LTS显著减少了总体运行时间，同时匹配或提高了任务性能，证明了学习内存准入是提高并行智能体系统效率的有效策略。
- 方法：提出学习共享（LTS）机制，实现跨团队信息的选择性重用。；引入全局内存库，所有团队均可访问。
- 结果：LTS显著减少了总体运行时间。；LTS在任务性能上匹配或优于无内存并行基线。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-62890) · [原文](https://icml.cc/virtual/2026/poster/62890)

#### Little By Little: Continual Learning via Incremental Mixture of Rank-1 Associative Memory Experts

![Little By Little: Continual Learning via Incremental Mixture of Rank-1 Associative Memory Experts](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-64295.webp)

- 导读：大规模预训练模型的持续学习（CL）面临任务干扰和灾难性遗忘的挑战。现有的基于LoRA的专家混合（MoE）方法通过添加新的任务特定适配器并冻结旧的适配器来缓解遗忘，但由于粗粒度专家和路由，常常出现冗余、干扰和模糊路由的问题。粗粒度专家（即具有大秩的完整LoRA适配器）编码低专业性信息。新添加的专家常常与现有专家重复或冲突，导致冗余和干扰。它们的低专业性进一步混淆了路由器，随着专家的积累加速了路由退化和遗忘。在本研究中，我们提出了MoRAM（Rank-1关联记忆混合）。基于权重矩阵作为线性关联记忆的观点，MoRAM通过逐步增加原子rank-1记忆专家来实现CL。每个rank-1适配器充当细粒度MoE专家或关联记忆单元。通过将rank-1适配器视为键值对，我们消除了MoE-LoRA中的显式路由器，使用一种自我激活机制，其中每个记忆原子通过其内在键评估其相关性。这将适应过程转化为稳健的、内容可寻址的检索。在CLIP和LLMs上的大量实验表明，MoRAM显著优于最先进的基线，实现了优越的可塑性-稳定性权衡，在提高泛化能力的同时缓解了遗忘。
- 方法：提出MoRAM（Rank-1关联记忆混合），通过逐步增加原子rank-1记忆专家来实现持续学习。；每个rank-1适配器充当细粒度MoE专家或关联记忆单元。
- 结果：MoRAM显著优于最先进的基线，实现了优越的可塑性-稳定性权衡。；在CLIP和LLMs上的实验表明，MoRAM在提高泛化能力的同时缓解了遗忘。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-64295) · [原文](https://icml.cc/virtual/2026/poster/64295)

#### NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating

![NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-60523.webp)

- 导读：音频提供了关键的情境线索，但当前的音频语言模型（ALMs）在长时录音中面临注意力瓶颈，即主导的背景模式会稀释罕见但显著的事件。我们提出了NAACA，一种无需训练的神经听觉注意认知架构，将注意力分配重新定义为听觉显著性过滤问题。其核心是OWM，一种受神经启发的振荡工作记忆，它维持稳定的类吸引子状态，仅在自适应能量波动感知到显著性时触发高级认知ALM处理，从而启动更高层次的推理。在XD-Violence数据集上，NAACA将AudioQwen的平均精度（AP）从53.50%提升至70.60%，同时减少了不必要的ALM调用。此外，在Urban Soundscapes of the World（USoW）数据集上的定性案例研究表明，OWM能够捕捉新事件和子类别变化，同时对瞬时停顿和环境城市噪声保持鲁棒性。
- 方法：提出无需训练的神经听觉注意认知架构（NAACA），将注意力分配转化为听觉显著性过滤问题。；设计振荡工作记忆（OWM），通过稳定类吸引子状态和自适应能量波动触发高级认知处理。
- 结果：在XD-Violence数据集上将AudioQwen平均精度（AP）从53.50%提升至70.60%。；显著减少不必要的ALM调用次数。
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-60523) · [原文](https://icml.cc/virtual/2026/poster/60523)

#### PRIM：Cooperative Dynamic Token Compression for Efficient Large Multimodal Models

![PRIM：Cooperative Dynamic Token Compression for Efficient Large Multimodal Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-60866.webp)

- 导读：大型多模态模型（LMMs）在视听理解方面展现出强大的能力，能够联合推理视觉、听觉和语言输入。然而，处理长格式的视听内容通常需要大量token，导致推理过程中计算和内存开销巨大。现有的效率导向方法通常采用统一的压缩或剪枝策略，忽视了多模态模型中模态特定的特性和指令依赖的推理行为。本文提出了PRIM，一个高效多模态推理的推理框架，基于注意力动态和指令相关性系统性地压缩视听表征。受注意力分析揭示的模态不平衡和层级冗余启发，PRIM引入了一个跨越多模态编码器和语言模型的协作压缩管道。具体而言，PRIM执行早期文本条件化的视听融合以外部化跨模态交互，应用注意力引导和频率感知策略去除冗余的音频和视频token，并根据任务需求进一步调整语言模型内部的token保留。在多个视听基准数据集上的广泛实验表明，PRIM在不同任务和数据集上始终实现稳定且优越的效率-准确性权衡。这些结果表明，PRIM作为一种多模态协作压缩方法，为可扩展和高效的视听推理提供了有效途径。
- 方法：PRIM执行早期文本条件化的视听融合以外部化跨模态交互。；应用注意力引导和频率感知策略去除冗余的音频和视频token。
- 结果：PRIM在不同任务和数据集上始终实现稳定且优越的效率-准确性权衡。；PRIM作为一种多模态协作压缩方法，为可扩展和高效的视听推理提供了有效途径。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-60866) · [原文](https://icml.cc/virtual/2026/poster/60866)

#### Progressive Cramming: Reliable Token Compression and What It Reveals

![Progressive Cramming: Reliable Token Compression and What It Reveals](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-66285.webp)

- 导读：令牌压缩技术通过学习嵌入将序列压缩至近乎完美的重构效果，但先前工作采用固定令牌预算和99%准确度阈值，难以区分残余错误是优化失败还是根本性限制。我们提出渐进式压缩方法，逐令牌增长目标前缀，仅在固定优化预算内无法实现重构时停止。通过渐进轨迹分析，发现优化路径在嵌入空间中呈现惊人的低维结构。注意力机制分析表明，压缩嵌入常成为特定中间层的注意力汇聚点，这与优化难度及下游性能退化相关。基于似然的多选评估中，前置压缩嵌入会使准确率降至接近随机猜测水平，即使上下文包含原始前缀。这些结果表明，完美重构可能源于脆弱的导向和注意力劫持，而非可迁移的语义表示。本研究将渐进式压缩定位为探索压缩极限的工具，同时证明完美重构不足以实现有意义的压缩。
- 方法：提出渐进式压缩方法：逐令牌扩展目标前缀直至重构失败；通过优化轨迹分析嵌入空间低维结构
- 结果：优化路径在嵌入空间中呈现显著低维结构（实验发现）；压缩嵌入成为特定中间层的注意力汇聚点（与优化难度正相关）
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-66285) · [原文](https://icml.cc/virtual/2026/poster/66285)

#### ProtoKV: Streaming Video Understanding under Delayed Evidence with Summary-State Memory

![ProtoKV: Streaming Video Understanding under Delayed Evidence with Summary-State Memory](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-63663.webp)

- 导读：流式视频理解(SVU)需要在视觉标记持续流动且GPU内存和查询延迟受限的情况下,异步响应查询请求。关键挑战在于延迟证据:决定性线索可能短暂出现,但在查询到达前会经历多次更新,增加了这些线索在有限内存中被驱逐或稀释的风险。我们提出ProtoKV,一种固定内存占用的SVU记忆机制,它将远历史表示为固定容量的摘要状态而非保留标记实例。ProtoKV维护精确的近窗口KV缓存,并将较旧内容聚合到带有残差统计量的语义-空间原型库中。在查询时,每个原型通过一个有限制的伪标记接口暴露,该接口与标准注意力机制完全兼容。在相同预算和可比查询时间成本下,ProtoKV在SVU基准测试中比标记保留基线准确率提升高达12.5个百分点,且证据延迟越大优势越明显。
- 方法：将远历史表示为固定容量的摘要状态而非原始标记；维护精确的近窗口KV缓存和残差统计的原型库
- 结果：在相同预算下比标记保留基线准确率提升高达12.5个百分点；证据延迟越大性能优势越显著
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-63663) · [原文](https://icml.cc/virtual/2026/poster/63663)

#### Context Forcing: Consistent Autoregressive Video Generation with Long Context

![Context Forcing: Consistent Autoregressive Video Generation with Long Context](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-61892.webp)

- 导读：近期实时长视频生成方法通常采用流式调优策略，试图用短上下文（无记忆）教师模型训练长上下文学生模型。这种框架下学生模型进行长序列展开，却只能获得局限于5秒短窗口的教师监督。这种结构性差异导致关键的师生不匹配问题：教师因无法获取长期历史信息而不能指导学生处理全局时间依赖关系，实质上限制了学生的上下文长度。为此，我们提出上下文强制(Context Forcing)框架，通过长上下文教师模型训练长上下文学生模型。通过确保教师感知完整生成历史，我们消除了监督不匹配问题，使模型能够稳健地保持长期一致性。为实现极端时长（如2分钟）的可行性，我们引入上下文管理系统，将线性增长的上下文转化为慢快记忆(Slow-Fast Memory)架构，显著降低视觉冗余。大量实验表明，本方法能实现超过20秒的有效上下文长度——比LongLive、Infinite-RoPE等前沿方法长6-10倍。利用这种扩展上下文，上下文强制在长时一致性上超越现有最优基线模型。
- 方法：提出上下文强制框架，使教师模型能感知完整生成历史；设计慢快记忆架构管理线性增长的上下文
- 结果：有效上下文长度突破20秒（比SOTA方法长6-10倍）；在2分钟极端时长下保持视频连贯性
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61892) · [原文](https://icml.cc/virtual/2026/poster/61892)

#### When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning

![When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/icml2026-61843.webp)

- 导读：尽管长上下文推理对众多实际应用至关重要，但大型语言模型(LLMs)在此方面仍面临挑战——随着上下文长度增加，其性能会显著下降。近期工作MemAgent尝试通过类RNN的循环逐块处理上下文并更新文本记忆库来解决问题，但该朴素循环记忆更新存在两大缺陷：(1) 记忆库可能因无差别更新（即便在无证据内容块上）而快速膨胀；(2) 循环缺乏退出机制，导致即便收集到充分证据后仍进行不必要计算。为此，我们提出GRU-Mem模型，通过引入两个文本控制门实现更稳定高效的长上下文推理。具体而言，GRU-Mem仅在更新门开启时更新记忆，并在退出门开启时立即终止循环。为赋予模型这种能力，我们在端到端强化学习中引入$r^{\text{update}}$和$r^{\text{exit}}$两种奖励信号，分别激励正确的更新和退出行为。在多种长上下文推理任务上的实验表明，GRU-Mem不仅效果优越，其推理速度较原始MemAgent最高可提升400%。
- 方法：引入更新门控制记忆写入时机（仅当检测到有效证据时）；设计退出门实现动态推理终止（满足条件立即停止）
- 结果：在多项长上下文任务中性能超越原始MemAgent；推理速度最高提升400%（相比基线）
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61843) · [原文](https://icml.cc/virtual/2026/poster/61843)

## CVPR 2026

本轮收录 35 篇。

### P0

#### Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulation

![Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-40221.webp)

- 导读：视觉语言动作（VLA）模型通过将视觉观察与语言指令直接映射到动作，在机器人操纵中展示强大性能。然而，它们在分布偏移下仍脆弱：当测试场景变化时，VLA常复现记忆轨迹而非适应更新场景，这是本文称为"记忆陷阱"的失败模式。此限制源于端到端设计，其缺乏显式3D空间推理并阻止在陌生环境中可靠识别可操作区域。为弥补此缺失的空间理解，3D空间可供性场（SAFs）可提供突出物理可行交互位置的几何表示，提供关于机器人应接近或避免区域的显式线索。因此本文引入可供性场干预（AFI），一种使用SAFs作为按需插件以引导VLA行为的轻量级混合框架。本文系统通过本体感觉检测记忆陷阱，将机器人重新定位到近期高可供性区域，并提议可供性驱动的waypoints以锚定VLA生成的动作。基于SAF的评分器随后选择具有最高累积可供性的轨迹。大量实验证明本文方法在不同VLA骨干（$\pi_{0}$与$\pi_{0.5}$）的分布外场景中真实世界机器人平台上实现平均23.5%的改善，且在LIBERO-Pro基准上实现20.2%，验证其在增强VLA对分布偏移鲁棒性方面的有效性。
- 方法：引入3D空间可供性场(SAFs)提供几何表示，标识物理可行的交互区域。；开发轻量级混合框架AFI，使用SAFs作为按需插件引导VLA行为。
- 结果：在真实世界机器人平台的分布外场景中，平均性能提升23.5%。；在LIBERO-Pro基准上性能提升20.2%。
- 对 VPWEM：关键反例：用于检验策略是否只是复现动作或轨迹。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-40221) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Affordance_Field_Intervention_Enabling_VLAs_to_Escape_Memory_Traps_in_CVPR_2026_paper.html)

#### CycleManip: Enabling Cycle-based Manipulation via Effective History Perception and Understanding

![CycleManip: Enabling Cycle-based Manipulation via Effective History Perception and Understanding](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-36298.webp)

- 导读：本文探索机器人操控中一个重要但尚未充分研究的任务：基于周期的操控，即机器人需要以预期终止时间执行周期性或重复性动作。这些任务在日常生活中至关重要，如摇瓶子或敲钉子。然而，先前很少有工作探索这一任务，导致两个主要挑战：1）模仿方法由于历史利用无效，往往无法在预期终止时间内完成这些任务；2）缺乏具有充足数据与自动评估工具的基准，阻碍了该领域有效解决方案的发展。为解决这些挑战，本文首先提出CycleManip框架，以端到端模仿方式实现基于周期的任务操控，无需任何额外模型、层次结构或显著计算开销。核心洞见是通过成本感知采样策略增强有效的历史感知，并通过多任务学习提升历史理解。其次，本文引入一个基于周期任务操控基准，提供多样化的基于周期的任务与自动评估方法。在仿真与真实世界设定中进行的大量实验表明，本文方法在基于周期任务操控中实现了高成功率。结果进一步展示了在通用操控中的强自适应性能，以及在视觉-语言-动作（VLA）模型等模仿策略上的即插即用能力。此外，结果表明本文方法可应用于多种机器人平台，包括双臂夹持器、灵巧手与人形机器人。
- 方法：提出CycleManip框架，以端到端模仿方式实现周期性操作；通过成本感知采样策略增强历史感知能力
- 结果：在周期性操作任务中实现高成功率；在通用操作任务中展现强大适应能力
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36298) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_CycleManip_Enabling_Cycle-based_Manipulation_via_Effective_History_Perception_and_Understanding_CVPR_2026_paper.html)

#### Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation

![Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-39665.webp)

- 导读：层级视觉-语言-动作（VLA）模型迅速成为机器人操作的主导范式。其通常包含用于感知与理解的视觉-语言骨干，以及用于动作生成的生成策略。然而，其性能日益受动作生成过程瓶颈限制。（i）低推理效率。各向同性噪声先验与目标动作分布之间的显著分布差距，增加去噪步数与不可行样本的发生率。（ii）差鲁棒性。现有策略仅条件于当前观测，忽视历史序列的约束，从而缺乏任务进度与时间一致性意识。为解决这些问题，本文引入OptimusVLA，一种具有全局先验记忆（GPM）与局部一致性记忆（LCM）的双记忆VLA框架。GPM用从语义相似轨迹检索的任务级先验替代高斯噪声，从而缩短生成路径并减少函数评估次数（NFE）。LCM动态建模执行的动作序列以推断任务进度并注入强制执行轨迹时间连贯性与平滑性的学习一致性约束。跨三个模拟基准，OptimusVLA一致超越强基线：在LIBERO上实现98.6%平均成功率，在CALVIN上较π_0改善13.5%，在RoboTwin 2.0 Hard上达到38%平均成功率。真实世界评估中，OptimusVLA在泛化与长程套件上排名最佳，分别超越π_0 42.9%与52.4%，同时提供2…
- 方法：引入全局先验记忆（GPM），用任务级先验替代高斯噪声，缩短生成路径。；引入局部一致性记忆（LCM），动态建模执行的动作序列，推断任务进展。
- 结果：在LIBERO上达到98.6%的平均成功率。；在CALVIN上比π0提高了13.5%。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39665) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Global_Prior_Meets_Local_Consistency_Dual-Memory_Augmented_Vision-Language-Action_Model_for_CVPR_2026_paper.html)

#### SemanticVLA: Towards Semantic Reasoning over Action Memorization via Synergistic Explicit Trace and Latent Action Planning

![SemanticVLA: Towards Semantic Reasoning over Action Memorization via Synergistic Explicit Trace and Latent Action Planning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-39352.webp)

- 导读：视觉语言动作（VLA）模型已成为有前景的范式，其中预训练视觉语言模型（VLMs）作为系统2用于高级推理，连接到作为系统1的动作专家用于低级运动控制。然而，当前工作未能真正利用VLM能力：VLMs产生缺乏语义可解释性的潜在嵌入，为下游策略提供模糊且不稳定的指导，而仅动作监督进一步导致VLMs退化为记忆动作模式而非执行泛化推理的参数重型融合编码器。为弥合此差距，本文引入SemanticVLA，其通过协同双路径设计利用VLM推理。显式轨迹推理通过VLM原生语言接口生成可解释的空间路径点作为文本坐标序列，直接重用其预训练空间落地为任务规划提供"思考过程"。潜在动作token通过补充轨迹推理学习基于视觉观察的紧凑视觉运动原语，提供超越纯坐标预测的更细粒度动作表征。此协同使轨迹推理能够利用VLM的多模态理解精化潜在token预测，而潜在token提供稳定且有根据的指导以补偿轨迹的数值敏感性。SemanticVLA在LIBERO上实现97.0%平均成功率与在SimplerEnv WidowX上实现65.1%， substantial 优于强基线。更重要的是，SemanticVLA在两个模拟套件中在…
- 方法：显式轨迹推理：通过VLM语言接口生成文本坐标序列作为可解释路径点；潜在动作标记：学习视觉观察基础的紧凑视觉运动基元
- 结果：LIBERO基准97.0%平均成功率，SimperEnv WidowX基准65.1%成功率；指令重述时性能稳定性显著优于基线
- 对 VPWEM：关键反例：用于检验策略是否只是复现动作或轨迹。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39352) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Ni_SemanticVLA_Towards_Semantic_Reasoning_over_Action_Memorization_via_Synergistic_Explicit_CVPR_2026_paper.html)

#### TRM-VLA: Temporal-Aware Chain-of-Thought Reasoning and Memorization for Vision-Language-Action Models

![TRM-VLA: Temporal-Aware Chain-of-Thought Reasoning and Memorization for Vision-Language-Action Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-36166.webp)

- 导读：视觉-语言-动作（VLA）模型已成为通用机器人操控的强大范式。然而，现有方法通常省略中间推理步骤并直接回归动作，限制了推理的可解释性以及在长程或组合任务中的性能。尽管近期研究将思维链（CoT）推理引入VLA模型，但其有效性仍不理想，主要存在两个关键问题：（1）在每个时间步生成完整推理轨迹会引入大量冗余，从而阻碍实时部署；（2）推理独立进行，忽视了时间一致性，导致规划冲突。本文提出TRM-VLA，一种时序感知的推理与记忆框架，将显式时序建模融入VLA推理过程。TRM-VLA包含两个核心组件：（1）关键帧触发推理（KTR），用于识别任务进度并仅在关键决策点执行分层CoT推理，以减少冗余推断；（2）粒度自适应上下文记忆（GCM），动态存储与检索历史推理轨迹，以保持帧间一致性与全局上下文。基于双系统架构——结合用于慢推理的多模态基础模型（系统2）与用于快速执行的扩散策略（系统1）——TRM-VLA以统一方式高效学习与执行规划。在LIBERO-90、SIMPLER及四项真实世界机器人任务上的大量实验表明，TRM-VLA在提升推理效率的同时达到了最先进的性能。
- 方法：引入关键帧触发推理（KTR），仅在关键决策点执行分层CoT推理。；采用粒度可适应的上下文记忆（GCM），动态存储和检索历史推理轨迹。
- 结果：在LIBERO-90、SIMPLER和四个真实世界机器人任务上实现了最先进的性能。；显著提高了推理效率。
- 对 VPWEM：关键反例：用于检验策略是否只是复现动作或轨迹。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36166) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TRM-VLA_Temporal-Aware_Chain-of-Thought_Reasoning_and_Memorization_for_Vision-Language-Action_Models_CVPR_2026_paper.html)

### P1

#### PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation

![PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-38810.webp)

- 导读：具有可验证奖励的强化学习 (RLVR) 已显示出 GUI 自动化的前景，使代理能够从二进制任务完成信号中学习。然而，当任务难度超过模型能力时，同策略探索无法发现正确的行动，从而产生消除学习信号的零优势陷阱。虽然合并非策略专家演示似乎很直观，但由于分布不匹配，它会导致持续的高熵状态，从而破坏有效的学习。我们提出了 GUI-SAGE，这是一个自解释框架，可以为 GUI 自动化生成分布内推理轨迹。通过以真实行为为条件，我们的方法产生分布内指导，避免分布外专家演示造成的混乱。我们进一步引入了熵调制信用分配，它通过共同考虑预测置信度和奖励信号来重新校准学习权重，从而能够放大自信正确行动的更新并减弱不确定探索的更新。在 AndroidControl 和 GUI-Odyssey 上进行的大量实验表明，GUI-SAGE-3B 实现了具有竞争力的性能，成功率为 81.1%，大大优于现有方法。我们的分析验证了自我解释保持了稳定的学习动态，而专家演示导致熵崩溃，并且熵调制对分布内样本提供了最大的改进。|||中断|||要在视觉和语言导航（VLN）中实现人类水平的表现，需要一个实体代理来共同理解多模态指令和视…
- 方法：构建基于交互中心可用性推理的策略学习框架；提炼捕捉物体相关性/接触几何/空间布局/运动动态的互补可用性表征
- 结果：LIBERO-LONG任务成功率91.8%；CALVIN ABC→D序列平均长度提升12.5%
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38810) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_PALM_Progress-Aware_Policy_Learning_via_Affordance_Reasoning_for_Long-Horizon_Robotic_CVPR_2026_paper.html)

#### AGiLe: Learning Robust Long-Horizon Manipulation via Affordance-Grounded Bidirectional Latent Planning

![AGiLe: Learning Robust Long-Horizon Manipulation via Affordance-Grounded Bidirectional Latent Planning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-39123.webp)

- 导读：长程操控任务的鲁棒执行仍是具身智能中的核心挑战，需要连贯高层规划与可靠低层控制。现有方法常遇到两个关键限制：子目标规划中预测误差的积累，导致随时间复合偏差；与规划-执行差距，其中高层抽象计划未能有效落地到连续感知-动作空间。为应对这些挑战，本文提出新颖统一框架Affordance-Grounded Bidirectional Latent Planning（AGiLe）。AGiLe引入双向潜规划机制，其联合优化后向规划器与前向评判器。后向规划器从最终目标生成目标导向子目标，而前向评判器评估其可达性，从而通过长程规划中持续一致性确保时间鲁棒性。此外，AGiLe通过利用可供性作为结构指导弥合规划-执行差距，将抽象子目标落地为驱动动作的稠密像素级视觉可供性。此方法增强空间鲁棒性，使系统能够有效适应语义与视觉干扰物。跨仿真与真实世界设置的广泛实证评估确认，AGiLe显著超越强基线，在仿真中实现较先前SOTA方法8.5%改进并展示其在长程操控任务中的有效性与鲁棒性。
- 方法：提出双向潜在规划机制：逆向规划器生成目标导向子目标，正向评价器评估可达性；利用功能可供性作为结构指导，将抽象子目标落地为像素级视觉可供性
- 结果：仿真实验中比先前最先进方法性能提升8.5%；在长时程操作任务中展现出卓越的有效性和鲁棒性
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39123) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_AGiLe_Learning_Robust_Long-Horizon_Manipulation_via_Affordance-Grounded_Bidirectional_Latent_Planning_CVPR_2026_paper.html)

#### Adaptive Action Chunking at Inference-time for Vision-Language-Action Models

![Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-39692.webp)

- 导读：在视觉-语言-动作（VLA）模型中，动作分块（即执行动作序列而无需中间重新规划）是提高机器人操作能力的关键技术。然而，大分块大小降低模型对新信息的响应性，而小分块增加模式跳跃的可能性，即由块间不连续性导致的急促行为。因此，选择最优分块大小以平衡模型反应性与一致性是迫切需求。不幸的是，当前VLA模型的主导趋势是推理时的经验固定分块长度，阻碍其在多样化操作任务中的优越性与可扩展性。为解决此问题，本文提出新颖的自适应动作分块（AAC）策略，其利用动作熵作为线索基于当前预测自适应确定分块大小。广泛模拟与真实世界机器人操作任务上的大量实验证明，本文方法较SOTA替代方案 substantial 改善性能。视频与源代码将公开发布。
- 方法：利用动作熵作为自适应调整分块尺寸的线索。；在推理时根据当前预测动态确定分块大小。
- 结果：在多种模拟和真实机器人操作任务上显著超越现有最优方案。；自适应分块策略有效平衡了反应性和一致性。
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39692) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.html)

#### ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving

![ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-36544.webp)

- 导读：自动驾驶需要从复杂多模态输入生成安全可靠的轨迹。传统模块化流水线将感知、预测与规划分离，而近期端到端（E2E）系统联合学习它们。视觉-语言模型（VLMs）通过引入跨模态先验与常识推理进一步丰富了该范式，但当前基于VLM的规划器面临三个关键挑战：（i）离散文本推理与连续控制之间的不匹配，（ii）自回归思维链解码的高延迟，以及（iii）低效或非因果规划器限制实时部署。本文提出ColaVLA，一种统一视觉-语言-动作框架，将推理从文本迁移到统一潜在空间并将其与层次化并行轨迹解码器耦合。认知潜在推理器通过自我中心自适应选择将场景理解压缩为紧凑的、面向决策的元动作嵌入，仅需两次VLM前向传递。层次化并行规划器随后在单次前向传递中生成多尺度、因果一致的轨迹。这些组件共同保留了VLMs的泛化性与可解释性，同时实现高效、准确且安全的轨迹生成。nuScenes基准上的实验表明，ColaVLA在开放环路与闭环设定中均达到最先进性能，具有有利的效率与鲁棒性。
- 方法：提出ColaVLA框架，将推理从文本转移到统一的潜在空间。；认知潜在推理器通过自我适应选择和仅两次VLM前向传递，压缩场景理解。
- 结果：在nuScenes基准测试中，ColaVLA在开环和闭环设置中均实现了最先进的性能。；ColaVLA具有较高的效率和鲁棒性。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36544) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Peng_ColaVLA_Leveraging_Cognitive_Latent_Reasoning_for_Hierarchical_Parallel_Trajectory_Planning_CVPR_2026_paper.html)

#### Dexterous World Models

![Dexterous World Models](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-37216.webp)

- 导读：三维重建的近期进展使从日常环境创建逼真数字孪生变得容易。然而，当前数字孪生 largely 保持静态——限于导航与视图合成而无具身交互性。为弥合这一差距，本文引入灵巧世界模型（DWM），一种场景-动作条件视频扩散模型，实现静态三维场景内的具身交互。给定静态三维场景渲染与第一人称手部运动序列，DWM生成描绘合理人机交互的时间连贯视频。本文方法将视频生成条件设定为（1）遵循指定相机轨迹的静态场景渲染以确保空间一致性，与（2）编码第一人称视图中几何与运动线索的第一人称手部网格渲染以直接建模动作条件动力学。本文在合成人-场景交互数据集与真实世界物体操控数据集上训练模型，然后跨合成与真实世界第一人称基准评估它。实验表明，DWM实现逼真、物理基础的交互，如抓取、打开或移动物体，同时保持相机与场景一致性。该框架建立基于视频扩散的交互数字孪生的第一步，实现从第一人称动作的具身模拟与三维场景交互性。
- 方法：结合静态3D场景渲染和指定相机轨迹，确保空间一致性。；利用自我中心手部网格渲染编码几何和运动线索，直接建模动作条件的动态。
- 结果：生成时间上连贯的视频，描绘合理的人-场景交互。；实现逼真、物理上合理的交互（如抓取、打开或移动物体）。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37216) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Dexterous_World_Models_CVPR_2026_paper.html)

#### Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics for Robot Manipulation

![Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics for Robot Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-37268.webp)

- 导读：虽然现有等变方法增强数据效率，但它们遭受高计算强度、单模态输入依赖以及与快速采样方法结合时的不稳定性。本文中，本文提出E3Flow，一种解决等变扩散策略关键局限的新颖框架。E3Flow成功克服这些挑战，首次统一高效rectified flow与稳定多模态等变学习。本文框架基于球谐表示构建以确保严格SO(3)等变性。本文引入新颖不变特征增强模块（FEM），动态融合混合视觉模态（点云与图像），将丰富视觉线索注入球谐特征。本文在MimicGen基准的8个操控任务上评估E3Flow，并进一步进行4个真实世界实验以验证其在物理环境中的有效性。模拟结果表明，E3Flow相比最先进水平球面扩散策略（SDP）平均成功率提高3.12%，同时实现7倍推理加速。E3Flow从而为机器人策略学习展示性能、效率与数据效率之间新的高效权衡。代码与视频将发布。
- 方法：基于球谐表示确保严格的SO(3)等变性；提出不变特征增强模块(FEM)动态融合点云与图像
- 结果：MimicGen基准8个任务平均成功率超越SDP方法3.12%；推理速度较SDP提升7倍
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37268) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Efficient_Hybrid_SE3-Equivariant_Visuomotor_Flow_Policy_via_Spherical_Harmonics_for_CVPR_2026_paper.html)

#### From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings

![From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-37209.webp)

- 导读：本文提出一种新颖无监督框架，解锁连续工业视频流中大量未标注人类演示数据用于视觉-语言-动作（VLA）模型预训练。本文方法首先训练轻量运动tokenizer以编码运动动力学，然后采用无监督动作分割器，利用新颖的「潜在动作能量」度量发现与分割语义连贯的动作基元。该流水线输出分割视频片段及其相应潜在动作序列，直接提供适合VLA预训练的结构化数据。公共基准与专有电机装配数据集上的评估证明人类在工作站执行的关键任务的有效分割。通过视觉-语言模型进一步聚类与定量评估确认发现的动作基元的语义一致性。据本文所知，这是首个用于从无结构工业视频中提取与组织VLA预训练数据的全自动端到端系统，为制造业中具身AI集成提供可扩展解决方案。
- 方法：训练轻量级运动编码器来编码运动动态。；利用“潜在动作能量”度量进行无监督动作分割。
- 结果：有效分割人类在工作站上执行的关键任务。；通过视觉-语言模型证实了所发现动作原语的语义一致性。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37209) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_From_Observation_to_Action_Latent_Action-based_Primitive_Segmentation_for_VLA_CVPR_2026_paper.html)

#### Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild

![Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-38737.webp)

- 导读：多模态学习系统的性能，特别是在自动抑郁症识别等高风险领域，从根本上受到从有限和复杂的临床数据中学习稳健的视觉表示的挑战的限制。为了克服这个问题，我们引入了跨模态引导视觉合成（CMG-VS），这是一种新颖的训练框架，可以通过合成新的、与任务相关的视觉特征来内部增强学习过程。CMG-VS 的核心是利用音频和文本模式的丰富上下文来指导条件生成模型。该模型学习从语音和语言到视觉表达的复杂映射，生成多种看似合理的视觉行为，以丰富训练分布。至关重要的是，这种合成不是一个单独的预处理步骤。通过任务引导的联合优化方案，生成过程由下游多模态识别器的性能动态控制。这种闭环反馈机制确保合成的视觉特征得到优化，以最大程度地区分识别任务，而不仅仅是现实的。对广泛使用的 DAIC-WOZ 和 E-DAIC 基准数据集进行的综合实验表明，CMG-VS 在所有标准回归和分类指标上均显着优于现有的最先进方法。消融研究进一步验证了我们的任务引导合成是这种性能提升的关键驱动因素，证明了其作为鲁棒多模态表示学习的新范例的有效性。|||中断|||长视频理解是困扰多模态大语言模型（MLLM）发展的一个关键挑战。在本文中，我们从…
- 方法：提出JALA框架，学习与逆向动力学和真实动作双重对齐的联合对齐潜在动作；绕过完整视觉动态重建，直接学习预测性动作嵌入
- 结果：在受控与非受控场景下生成更真实的手部动作；显著提升模拟和真实世界任务的机器人操作性能
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38737) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Joint-Aligned_Latent_Action_Towards_Scalable_VLA_Pretraining_in_the_Wild_CVPR_2026_paper.html)

#### PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation

![PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-38054.webp)

- 导读：人类通过一瞥和对身体动作的 contemplated 来预期3D世界将如何响应，这一能力对机器人操纵同样至关重要。我们引入PointWorld，一个大型预训练3D世界模型，在共享3D空间中将状态和动作统一为3D点流：给定一个或少数RGB-D图像和一系列低级机器人动作命令，PointWorld预测响应给定动作的每像素3D位移。通过将动作表示为3D点流而非具身特定的动作空间（如关节位置），这种公式直接以机器人的物理几何为条件，对接触推理至关重要，同时无缝集成跨具身学习。为训练我们的3D世界模型，我们策划了一个大规模数据集，涵盖开放世界环境中的真实和模拟机器人操纵，利用3D视觉和模拟环境的最新进展，总计约200万轨迹和500小时，涵盖单臂Franka和双手人形机器人。通过对骨干、动作表示、学习目标、部分可观测性、数据混合、域迁移和扩展的严格大规模实证研究，我们提炼出大规模3D世界建模的设计原则。以实时（0.1秒）推理速度，PointWorld可以高效集成到模型预测控制（MPC）框架中进行操纵。我们证明，单个预训练检查点使真实世界Franka机器人能够执行刚体推动、可变形和关节物体操纵以及工…
- 方法：将机器人动作建模为与物体几何交互的3D点流（而非关节空间）；构建跨真实/仿真的大规模异构机器人操控数据集（200万轨迹）
- 结果：单模型支持Franka机器人零样本完成刚体/可变形体/铰接体操控（无演示数据）；在500小时跨形态数据上验证（含双手机器人）
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38054) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_PointWorld_Scaling_3D_World_Models_for_In-The-Wild_Robotic_Manipulation_CVPR_2026_paper.html)

#### RehearseVLA: Simulated Post-Training for VLAs with Physically-Consistent World Model

![RehearseVLA: Simulated Post-Training for VLAs with Physically-Consistent World Model](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-37811.webp)

- 导读：通过模仿学习训练的视觉-语言-动作（VLA）模型在数据稀缺场景中遭受显著性能退化，由于它们依赖大规模演示数据集。虽然基于强化学习（RL）的后训练已证明有效解决数据稀缺，但其在VLA模型中的应用受真实世界环境不可重置性质阻碍。这一局限在工业自动化等高风险域中尤为关键，其中交互常引起状态变化，重置成本高昂或不可行。此外，现有VLA方法缺乏检测任务完成的可靠机制，导致降低整体任务成功率的冗余动作。为应对这些挑战，本文提出RehearseVLA，一种基于RL的后训练框架，用低成本世界模型虚拟模拟器替代物理交互。RehearseVLA由两个关键组件组成：（1）生成时间一致未来视觉观测的物理一致世界模拟器，与（2）提供连续奖励信号并预测动作终止的视觉语言模型（VLM）引导即时反射器。这种模拟环境使VLA模型能够安全探索并超越其初始模仿学习分布泛化。本文方法在每任务仅五个专家演示下即实现显著性能增益。复杂机器人操控任务上的实验证明RehearseVLA有效克服传统依赖真实世界交互的VLA模型的数据低效、安全约束与低效执行，为资源受限设置中的后训练提供实用且可扩展的解决方案。
- 方法：采用物理一致的世界模拟器生成时序一致的未来视觉观测；设计VLM引导的即时反射器提供持续奖励与动作终止预测
- 结果：仅需每任务5次专家演示即可显著提升性能；在复杂机器人操作任务中克服传统方法的数据低效问题
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37811) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_RehearseVLA_Simulated_Post-Training_for_VLAs_with_Physically-Consistent_World_Model_CVPR_2026_paper.html)

#### Scalable Trajectory Generation for Whole-Body Mobile Manipulation

![Scalable Trajectory Generation for Whole-Body Mobile Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-40034.webp)

- 导读：移动机器人需要协调全身运动以有效执行家务任务。当前移动操作数据集依赖昂贵的遥操作或慢速规划方法，将可用数据限制在数百个演示。此数据稀缺严重限制可泛化基于学习策略的开发。本文中，本文证明GPU加速规划每GPU小时生成多达5,000个片段，较现有方法快80倍以上。本文AutoMoMa流水线在300个家庭场景与多种机器人 embodiment 中产生500K多样化物理有效全身运动，而先前数据集限于数百个演示的狭窄机器人-场景对。下游验证证明大规模训练数据带来一致的策略改善。本文为移动操作数据瓶颈提供首个可扩展解决方案。通过实现大规模数据集生成，AutoMoMa加速朝向能够执行复杂协调任务的通用家用机器人的进展。
- 方法：使用GPU加速规划，每小时生成多达5,000个任务片段。；开发AutoMoMa管道，生成50万种多样且物理上有效的全身运动。
- 结果：GPU加速规划比现有方法快80倍以上。；AutoMoMa生成了50万种多样且物理上有效的全身运动。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-40034) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Niu_Scalable_Trajectory_Generation_for_Whole-Body_Mobile_Manipulation_CVPR_2026_paper.html)

#### Scaling Spatial and Temporal Context for Robotic Imitation Learning Policies With Scene Graphs

- 导读：模仿学习使机器人能够通过观察学习如何执行任务。然而，由于空间尺度较大，家庭和办公室等现实世界环境常常被严重部分观察到。此外，许多任务涉及执行一系列子任务，需要自主机器人在较长的时间范围内进行推理。为了应对这些挑战，我们建议使用场景图作为模仿学习中的显式结构化记忆机制。通过维护一个动态场景图来捕获以对象为中心的关系及其随时间的演变，我们的方法允许代理在任务执行期间保留相关的历史上下文，以有效地推理增量产生的场景信息。我们对模拟移动操作和现实世界桌面操作的实验表明，我们的方法大大提高了策略性能，特别是在部分可观察性下需要长期推理和稳健泛化的环境中。
- 方法：使用场景图作为显式和结构化的记忆机制。；维护动态场景图捕捉物体中心关系及其演变。
- 结果：显著提高了策略性能。；在需要长期推理的场景中表现优异。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38076)

### P2

#### An Efficient Token Compression Framework for Visual Object Tracking

![An Efficient Token Compression Framework for Visual Object Tracking](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-36338.webp)

- 导读：通过消除特征级冗余来精化视觉表示，对于同时优化视觉跟踪模型的性能与计算成本至关重要。为提升性能，许多当代基于Transformer的跟踪器利用更多历史模板帧以捕获更丰富的时空线索。然而，该策略导致大量输入视觉token，产生两个关键问题：带来二次计算负担，并可能降低跟踪器整体性能。为弥合这一差距，本文提出压缩-然后-交互跟踪框架ETCTrack，学习将来自历史模板帧的模板token高效压缩为鲁棒的目标表示，超越手工规则。本文方法首先采用自适应token压缩器，通过过滤冗余视觉token动态构建紧凑 yet highly discriminative的模板token。随后，这些精化token由本文的层次交互编码器处理，以实现与搜索特征的深度自适应交互。该融合通过协作级联阶段执行，每个阶段执行模板通过搜索上下文丰富、统一特征学习与搜索特征精化的结构化过程，以确保精确目标定位。在七个基准上的实验表明，本文方法显著优于当前最先进的跟踪器。
- 方法：提出压缩后交互的跟踪框架ETCTrack，超越手工规则。；使用自适应令牌压缩器动态构建紧凑且高度区分的模板令牌。
- 结果：在七个基准测试上显著优于当前最先进的跟踪器。；通过自适应令牌压缩器有效过滤冗余视觉令牌。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36338) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_An_Efficient_Token_Compression_Framework_for_Visual_Object_Tracking_CVPR_2026_paper.html)

#### ApET: Approximation-Error Guided Token Compression for Efficient VLMs

![ApET: Approximation-Error Guided Token Compression for Efficient VLMs](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-39704.webp)

- 导读：近期视觉语言模型（VLMs）展现显著的多模态理解能力， yet 冗余视觉token产生 prohibitive 计算开销并降低推理效率。先前研究通常依赖[CLS]注意力或文本-视觉交叉注意力识别并丢弃冗余视觉token。尽管有前景的结果，此类解决方案易引入位置偏差，更关键的是与FlashAttention等高效注意力核不兼容，限制其VLM加速的实际部署。本文中，本文脱离注意力依赖并从信息论视角重新审视视觉token压缩，旨在无需任何注意力参与最大化保留视觉信息。本文呈现ApET，一种近似误差token压缩框架。ApET首先通过线性近似用少量基token重建原始视觉token，然后利用近似误差识别并丢弃信息量最少的token。跨多个VLMs与基准的大量实验证明，ApET在图像理解任务上保留95.2%原始性能，在视频理解任务上甚至达到100.4%，同时将token预算分别压缩88.9%与87.5%。得益于其无注意力设计，ApET与FlashAttention无缝集成，实现进一步推理加速并使VLM部署更实用。
- 方法：从信息论角度重构视觉标记压缩问题，避免依赖注意力机制；通过线性近似用少量基标记重构原始视觉标记
- 结果：图像理解任务保留95.2%原始性能（标记预算减少88.9%）；视频理解任务达到100.4%原始性能（标记预算减少87.5%）
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39704) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_ApET_Approximation-Error_Guided_Token_Compression_for_Efficient_VLMs_CVPR_2026_paper.html)

#### BiGain: Unified Token Compression for Joint Generation and Classification

![BiGain: Unified Token Compression for Joint Generation and Classification](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-37406.webp)

- 导读：扩散模型的加速方法（如token合并或下采样）通常优化减少计算下的合成质量，但它们常忽视模型的潜在判别能力。本文重新审视具有联合目标的token压缩并呈现BiGain，一种无需训练、即插即用的框架，在加速扩散模型中保留生成质量同时显著改善分类。本文关键洞见是频率分离：将特征空间信号映射到频率感知表示将精细细节与全局语义解耦，实现同时尊重生成保真度与判别效用的压缩。BiGain通过两个频率感知算子反映这一原则：（1）拉普拉斯门控token合并，鼓励频谱平滑token之间的合并同时阻止高对比度token的合并，从而保留边缘与纹理；与（2）插值-外推KV下采样，通过最近与平均池化之间的可控内插-外插下采样键/值同时保持查询完整，从而无需重新训练即可保持注意力精度。跨基于DiT与U-Net的骨干以及ImageNet-1K、ImageNet-100、Oxford-IIIT Pets与COCO-2017的多个数据集，本文所提算子一致改善基于扩散分类的速度-准确率权衡，同时在可比加速下保持有时甚至增强生成质量。例如，在ImageNet-1K上，Stable Diffusion 2.0上70%的to…
- 方法：拉普拉斯门控令牌合并：鼓励频谱平滑令牌的合并，抑制高对比度令牌的合并，保留边缘和纹理。；插值-外推KV下采样：通过可控插值外推对键/值进行下采样，保持查询不变，确保注意力精度。
- 结果：在ImageNet-1K上，令牌合并比率为70%时，分类准确性提高7.15%。；生成的FID降低0.34（1.85%）。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37406) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_BiGain_Unified_Token_Compression_for_Joint_Generation_and_Classification_CVPR_2026_paper.html)

#### Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing

![Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-37693.webp)

- 导读：始终在线感知对下一代边缘/可穿戴AI系统至关重要，但连续高保真RGB视频捕获对资源受限的移动与边缘平台仍 prohibitively expensive。本文呈现高效流式视频理解的新范式：灰度始终在线，按需彩色。通过初步研究，本文发现彩色并非始终必要。当时间结构通过连续灰度流保留时，稀疏RGB帧足以获得可比性能。基于这一洞察，本文提出ColorTrigger，一种基于窗口化灰度亲和性分析选择性地激活彩色捕获的在线无需训练触发器。为实时边缘部署设计，ColorTrigger使用轻量二次规划因果检测色度冗余，辅以信用预算控制与动态token路由以联合降低感知与推理成本。在流式视频理解基准上，ColorTrigger以仅使用8.1% RGB帧实现全彩色基线91.6%性能，证明自然视频中的实质性彩色冗余，并在资源受限设备上实现实用的始终在线视频感知。
- 方法：提出'灰度持续+按需彩色'混合采集范式；基于窗口灰度相似性分析开发免训练在线触发器ColorTrigger
- 结果：仅使用8.1%的RGB帧即达到全彩色基线91.6%的性能；验证自然视频中存在显著色度冗余
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37693) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Cai_Color_When_It_Counts_Grayscale-Guided_Online_Triggering_for_Always-On_Streaming_CVPR_2026_paper.html)

#### EarlyTom: Early Token Compression Completes Fast Video Understanding

![EarlyTom: Early Token Compression Completes Fast Video Understanding](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-37950.webp)

- 导读：视频大语言模型（Video-LLMs）在视频理解任务中展示了强大能力。然而，它们的实际部署仍受处理大量视觉token带来的效率问题阻碍。虽然近期方法在保持与全token基线相当准确性的同时实现了极低的token保留率，但它们大多数仅在prefilling的后期阶段执行压缩，使视觉编码器的效率未得到优化。本文中，我们首先证明视觉编码对首token时间（TTFT）贡献很大一部分。因此，仅在视觉编码器之后压缩视觉token，在编码器内部执行压缩仍留下大量探索空间。基于这一洞察，我们提出EarlyTom，一种免训练的token压缩框架，在视觉编码器内部执行早期阶段视觉token压缩，实现显著更好的TTFT降低和更高吞吐量。此外，我们引入解耦的空间token选择策略以提高整体压缩效果。EarlyTom在单个NVIDIA A100 GPU上为LLaVA-OneVision-7B模型将TTFT降低高达2.65倍，FLOPs降低高达61%，同时保持与全token基线可比较的准确性。这些改进大幅增强了在现实生产场景中部署Video-LLMs的实用性。
- 方法：提出EarlyTom，一种无需训练的token压缩框架。；在视觉编码器内部进行早期视觉token压缩。
- 结果：TTFT最多减少2.65倍。；FLOPs最多减少61%。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37950) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_EarlyTom_Early_Token_Compression_Completes_Fast_Video_Understanding_CVPR_2026_paper.html)

#### FlashVSR: Towards Real-time Diffusion-Based Streaming Video Super Resolution

![FlashVSR: Towards Real-time Diffusion-Based Streaming Video Super Resolution](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-38690.webp)

- 导读：扩散模型近期推进视频恢复，但将其应用于真实世界与AIGC生成视频超分辨率（VSR）仍具挑战性，因高延迟、 prohibitive 计算与对超高分辨率的差泛化。本文目标是通过实现效率、可扩展性与近实时性能，使基于扩散的VSR实用化。为此，本文提出FlashVSR，首个基于扩散的一步流式框架，用于高效视频超分辨率。FlashVSR通过在单张A100 GPU上以约17 FPS运行768×1408视频，结合三项互补创新实现此目标：（i）训练友好的三阶段蒸馏流水线，实现流式超分辨率；（ii）局部约束稀疏注意力，减少冗余计算同时桥接训练-测试分辨率差距；与（iii）微小条件解码器，加速重建而不牺牲质量。为支持大规模训练，本文还构建VSR-120K，一种包含12万视频与18万图像的新数据集。大量实验证明，FlashVSR可靠扩展至超高分辨率，实现较先前基于一步扩散的VSR模型高达约12倍加速的SOTA性能。本文将发布代码、预训练模型与数据集以促进基于高效扩散的视频超分辨率未来研究。
- 方法：提出适合训练的三阶段蒸馏管道，支持流式超分辨率。；引入局部约束的稀疏注意力机制，减少冗余计算并弥合训练与测试分辨率之间的差距。
- 结果：FlashVSR在单个A100 GPU上以约17 FPS的速度处理768×1408视频。；构建了VSR-120K数据集，包含12万视频和18万图像。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38690) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhuang_FlashVSR_Towards_Real-time_Diffusion-Based_Streaming_Video_Super_Resolution_CVPR_2026_paper.html)

#### FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding

![FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-38716.webp)

- 导读：本文呈现FluxMem，一种免训练且自适应的记忆框架，用于高效流式视频理解。FluxMem通过层次冗余缩减过程渐进压缩视觉记忆。具体而言，Temporal Adjacency Selection（TAS）移除相邻帧间冗余token以缓解时间冗余，而Spatial Domain Consolidation（SDC）进一步将每帧内空间重复区域合并为紧凑表征。为确保跨多样化场景动态的鲁棒性，两个模块均采用源自内在场景统计的自适应阈值，自动调整压缩率而无需手动调参。大量实验证明FluxMem在在线基准上建立新SOTA，在StreamingBench上实时实现76.4，在OVO-Bench上实现66.3。此外，它展示强离线性能，在MLVU上实现73.1同时减少65%视觉token。
- 方法：时间邻接选择(TAS)：消除相邻帧间的冗余标记；空间域整合(SDC)：合并帧内空间重复区域为紧凑表示
- 结果：StreamingBench实时得分76.4；OVO-Bench实时得分66.3
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38716) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_FluxMem_Adaptive_Hierarchical_Memory_for_Streaming_Video_Understanding_CVPR_2026_paper.html)

#### Interactive Episodic Memory with User Feedback

![Interactive Episodic Memory with User Feedback](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-37436.webp)

- 导读：人类记忆常不可靠。我们忘记物品放置位置、忽视小细节并难以准确回忆过去事件。带有自然语言查询的情景记忆（EM-NLQ）旨在通过允许用户使用自然语言问题搜索其过去视觉体验（通过第一人称视频捕获）来克服这些局限。虽然近期模型聚焦于解决EM-NLQ中的挑战如嘈杂输入视频与效率，但它们忽视此任务的一个关键方面：交互性。在真实场景中，用户有能力精化其查询并在模型响应偏离目标时提供反馈，但当前EM-NLQ方法无法纳入或受益于此类反馈。为解决这一差距，本文引入首个交互式EM-NLQ框架，具有即插即用反馈对齐模块（FALM），使现有模型能够高效纳入用户反馈并精化其预测。此外，本文引入带有问题与反馈的情景记忆任务（EM-QnF）以及为基于反馈的交互量身定制的新数据集与轻量训练方案，消除昂贵顺序优化的需求。本文方法称为ReFocus，将FALM与最先进水平EM-NLQ方法结合以在三个挑战性基准上达到最先进水平结果，并在基于人类的反馈评估中展示显著改进，使EM-NLQ更接近真正交互式与自适应视觉记忆系统。
- 方法：提出反馈对齐模块(FALM)实现即插即用的用户反馈整合；设计带问题与反馈的新任务EM-QnF及专用数据集
- 结果：在三个基准测试达到最先进水平；人类反馈评估显示显著改进
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37436) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Subedi_Interactive_Episodic_Memory_with_User_Feedback_CVPR_2026_paper.html)

#### Joint Learning of General and Diverse Patterns with Mixture of Memory Experts for Weakly-Supervised Video Anomaly Detection

![Joint Learning of General and Diverse Patterns with Mixture of Memory Experts for Weakly-Supervised Video Anomaly Detection](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-36266.webp)

- 导读：弱监督视频异常检测（wVAD）旨在仅使用二元标签检测异常事件，使其难以同时捕获异常的多样性及其共享的语义线索。现有方法要么聚焦于通用异常模式，实现强泛化但弱判别性；要么依赖类别级多样性建模，忽视共享语义且泛化有限。为克服这些局限，本文提出混合记忆专家（MoME），一种联合学习通用与多样化模式的统一框架。MoME中的每个专家拥有用于细粒度专业化的内部记忆，并共享用于通用知识聚合的外部记忆。为增强语义多样性并提升超出粗粒度类别级监督的泛化能力，本文引入异常原型路由器，利用大语言模型构建广义异常原型，以进行语义引导的专家路由。此外，APR的正则化损失确保均衡路由，专家的区分性损失鼓励多样性，重建与记忆任务增强模式判别性。在UCF-Crime与XD-Violence上的大量实验表明，本文方法达到最先进性能，验证了在弱监督下联合建模通用性与多样性以实现鲁棒异常检测的有效性。
- 方法：提出混合记忆专家（MoME）框架，联合学习通用和多样化模式。；每个专家拥有内部记忆用于细粒度专业化，共享外部记忆用于通用知识聚合。
- 结果：在UCF-Crime和XD-Violence数据集上实现了最先进的性能。；验证了联合建模通用性和多样性对于鲁棒异常检测的有效性。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36266) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Joint_Learning_of_General_and_Diverse_Patterns_with_Mixture_of_CVPR_2026_paper.html)

#### OASIS: On-Demand Hierarchical Event Memory for Streaming Video Reasoning

![OASIS: On-Demand Hierarchical Event Memory for Streaming Video Reasoning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-36584.webp)

- 导读：流式视频推理要求模型在无限增长的历史中运行，而有意义的证据仍然稀缺。在这种景观中，相关信号如同绿洲——微小、关键且容易在冗余沙漠中丢失。扩大记忆只会拓宽沙漠；激进压缩会干涸绿洲。真正的困难在于发现去哪里寻找，而非记住多少。因此本文引入OASIS，一种通过结构化按需检索应对这一挑战的流式视频推理新颖框架。它将流式历史组织为层次化事件，并将推理执行为受控精化——先短上下文推理，随后仅在不确定性出现时进行语义锚定的检索。由于检索由高层意图而非嵌入相似性驱动，检索记忆 substantially更准确且噪声更小。此外，该机制即插即用、免训练且兼容任何流式MLLM。多基准实验表明，OASIS以更少的内存预算在长程精度与组合推理方面实现强劲增益。
- 方法：将流式历史组织为分层事件结构；采用可控细化推理流程（短上下文优先，不确定性触发语义检索）
- 结果：在多个基准测试上实现长时程准确性显著提升；组合推理能力大幅增强
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36584) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_OASIS_On-Demand_Hierarchical_Event_Memory_for_Streaming_Video_Reasoning_CVPR_2026_paper.html)

#### Revisiting Token Compression for Accelerating ViT-based Sparse Multi-View 3D Object Detectors

![Revisiting Token Compression for Accelerating ViT-based Sparse Multi-View 3D Object Detectors](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-39920.webp)

- 导读：基于视觉Transformer（ViT）的稀疏多视图3D目标检测器已取得显著精度，但仍因繁重token处理而 suffer 高推理延迟。为加速这些模型，token压缩已被广泛探索。然而，本文对现有策略（如token剪枝、合并与patch尺寸放大）的重新审视揭示它们常丢弃有信息量的背景线索、破坏上下文一致性并丢失细粒度语义，对3D检测产生负面影响。为克服这些限制，本文提出SEPatch3D，一种在粗patch内保留关键语义信息同时动态调整patch尺寸的新颖框架。具体而言，本文设计时空感知patch尺寸选择（SPSS），其向包含附近对象的场景分配小patch以保留精细细节，向背景主导场景分配大patch以降低计算成本。为进一步缓解潜在细节损失，信息性patch选择（IPS）选择信息性patch进行特征精化，跨粒度特征增强（CGFE）将细粒度细节注入选定粗patch中，丰富语义特征。nuScenes与Argoverse 2验证集上的实验表明SEPatch3D实现较StreamPETR基线快达57%的推理，较SOTA ToC3D-faster效率高20%，同时保持可比较的检测精度。
- 方法：提出SEPatch3D框架，动态调整patch尺寸，保留关键语义信息。；设计时空感知的Patch尺寸选择（SPSS），根据场景分配不同大小的patch。
- 结果：SEPatch3D比StreamPETR基线推理速度提高了57%。；SEPatch3D比最先进的ToC3D-faster效率提高了20%。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39920) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_Revisiting_Token_Compression_for_Accelerating_ViT-based_Sparse_Multi-View_3D_Object_CVPR_2026_paper.html)

#### Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation

![Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-37619.webp)

- 导读：高效流式视频生成对模拟交互式动态世界至关重要。现有方法用滑动窗口注意力蒸馏少步视频扩散模型，使用初始帧作为汇聚token以保持注意力性能并减少误差累积。然而，视频帧对这些静态token过度依赖，导致复制初始帧与运动动力学减弱。为解决这一问题，本文引入Reward Forcing，一种具有两个关键设计的新颖框架。首先，本文提出EMA-Sink，维护从初始帧初始化的固定大小token并通过指数移动平均持续融合退出滑动窗口的驱逐token来更新。无需额外计算成本，EMA-Sink token捕获长期上下文与近期动力学，防止初始帧复制同时保持长程一致性。其次，为更好地从教师模型蒸馏运动动力学，本文提出新颖奖励分布匹配蒸馏（Re-DMD）。vanilla分布匹配平等对待每个训练样本，限制模型优先处理动态内容的能力。相反，Re-DMD通过优先处理由视觉-语言模型评分的具有更大动力学的样本来使模型输出分布偏向高奖励区域。Re-DMD显著增强运动质量同时保留数据保真度。本文包含定量与定性实验以证明Reward Forcing在标准基准上达到最先进水平性能，同时在单张H100 GPU上实现23.1 F…
- 方法：提出EMA-Sink机制：通过指数移动平均更新固定大小token，平衡长期上下文与近期动态；设计Re-DMD蒸馏法：基于视觉语言模型评分优先学习高动态样本，优化运动生成质量
- 结果：在标准基准测试中达到SOTA性能；单卡H100 GPU实现23.1 FPS的高质量流式生成
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37619) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Reward_Forcing_Efficient_Streaming_Video_Generation_with_Rewarded_Distribution_Matching_CVPR_2026_paper.html)

#### Stabilizing Streaming Video Geometry via Dynamic Feature Normalization

![Stabilizing Streaming Video Geometry via Dynamic Feature Normalization](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-38162.webp)

- 导读：根据流式 RGB 输入进行一致的 3D 几何估计对于自动驾驶、嵌入式 AI 和大规模重建等现实应用至关重要。虽然现代单目几何基础模型实现了很强的单图像精度，但它们在连续输入上表现出严重的时间不一致，特别是尺度移位漂移。通过有针对性的实证分析，我们将这种不稳定性追溯到其根本原因：潜在特征统计数据的波动，其均值和方差直接决定了预测深度的尺度和偏移。基于这一见解，我们引入了动态特征归一化（DyFN），这是一个轻量级的因果循环模块，可以动态、稳健地调节特征统计数据，以随着时间的推移保持稳定的几何形状。我们通过仅微调 DyFN（仅 2％ 的附加参数）来适应强大的预训练单目几何模型，同时保持骨干网冻结，从而在不影响单图像精度的情况下实现时间一致性。跨越四个基准的大量实验表明，DyFN 有效消除了时间伪像，例如脱节的分层和位置抖动，并实现了最先进的时间稳定性，比之前的流媒体方法提高了 14%，甚至超越了更重的非因果视频基线。
- 方法：引入动态特征归一化（DyFN），动态调节特征统计量以保持几何稳定性。；仅微调DyFN，保持主干网络冻结，增加2%的参数。
- 结果：DyFN有效消除了时间伪影，如不连贯的分层和位置抖动。；在四个基准测试上实现了最先进的时间稳定性。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38162) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Lyu_Stabilizing_Streaming_Video_Geometry_via_Dynamic_Feature_Normalization_CVPR_2026_paper.html)

#### StreamReady: Learning What to Answer and When in Long Streaming Videos

![StreamReady: Learning What to Answer and When in Long Streaming Videos](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-37104.webp)

- 导读：流式视频理解常涉及时间敏感场景，其中模型需要在支持性视觉证据出现时准确回答：在证据前回答反映推测，在证据通过后回答降低实时效用。为捕获这种行为，本文引入流式视频理解的 readiness-aware 公式化，包含答案准备度分数（ARS），一种具有不对称早罚与晚罚的时间感知目标。与正确性结合时，ARS定义有效准确率，不仅衡量模型是否正确，还衡量其是否在适当时刻回答。基于该公式化，本文引入StreamReady，一种通过轻量准备度机制统一时间推理与准时回答的框架，该机制在响应前决定是否已观察到充分证据。为评估该能力，本文进一步引入ProReady-QA，一种具有注释答案证据窗口与跨局部与全局上下文的主动多轮问题的基准。StreamReady在ProReady-QA上实现优越性能，并持续优于跨八个额外流式与离线长视频基准的先前方法，展示鲁棒且广泛可泛化的视频理解能力。
- 方法：引入回答准备度评分（ARS），一种具有不对称早期和晚期惩罚的时间感知目标。；提出StreamReady框架，通过轻量级准备机制统一时间推理与及时回答。
- 结果：StreamReady在ProReady-QA基准上表现出色。；StreamReady在八个额外的流媒体和离线长视频基准上始终优于先前的方法。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37104) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Azad_StreamReady_Learning_What_to_Answer_and_When_in_Long_Streaming_CVPR_2026_paper.html)

#### Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning

![Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-39800.webp)

- 导读：犯罪预判使主动公共安全干预成为可能， yet 现有视频监控系统基本保持被动，无法检测犯罪前兆。虽当前基于视觉语言模型（VLM）的视频理解方法在高级推理中展现前景，其未设计为显式建模对预判犯罪至关重要的时空因果关系。本文通过两个因果驱动组件解决此限制。首先，本文开发时空因果推理犯罪（STCRC）数据集，一种包含73K样本跨越五个渐进因果推理任务的层级数据集，促进犯罪前兆学习。其次，本文提出时空因果超图（STCH），一种将隐式实体动态转换为显式因果结构以增强VLM犯罪因果推理的流式模块。通过结合这两个组件，本文框架推进实时犯罪预判，在预判任务中实现改善：犯罪分类70.7%相对改善、犯罪检测10.1%、时间预测误差减少3.7%。
- 方法：开发了包含73K样本的时空因果推理犯罪（STCRC）数据集。；提出了时空因果超图（STCH）流式模块，将隐式实体动态转化为显式因果结构。
- 结果：犯罪分类相对提高了70.7%。；犯罪检测提高了10.1%。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39800) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html)

#### Streaming Video Instruction Tuning

![Streaming Video Instruction Tuning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-38386.webp)

- 导读：我们提出Streamo，一种实时流式视频LLM，作为通用交互式助手。与现有聚焦于问答或描述任务的在线视频模型不同，Streamo执行广泛的流式视频任务，包括实时叙述、动作理解、事件描述、时间事件定位和时间敏感问答。为开发这种通用性，我们构建Streamo-Instruct-465K，一个面向流式视频理解的大规模指令遵循数据集。该数据集涵盖多样化时间上下文和多任务监督，实现跨异构流式任务的统一训练。通过精简流程在指令遵循数据集上端到端训练后，Streamo在各类流式基准上展现出强劲的时间推理、响应式交互和广泛泛化能力。大量实验表明，Streamo弥合了离线视频感知模型与实时多模态助手之间的差距，向连续视频流中的统一智能视频理解迈进了一步。
- 方法：构建Streamo-Instruct-465K指令数据集（46.5万条）覆盖多时序上下文；设计统一的多任务监督训练框架
- 结果：在多样化流式视频任务（5类）中实现统一性能；实验证明优于现有单任务视频模型
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38386) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Xia_Streaming_Video_Instruction_Tuning_CVPR_2026_paper.html)

#### StreamingTOM: Streaming Token Compression for Efficient Video Understanding

![StreamingTOM: Streaming Token Compression for Efficient Video Understanding](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-39087.webp)

- 导读：与离线处理不同，流式视频视觉语言模型面临两个基本约束：因果性与累积。因果性阻止访问离线方法利用的未来帧，而累积导致token无界增长，造成效率瓶颈。然而，现有方法仅调节LLM后kv缓存，使昂贵的LLM前预填充保持不变。本文引入StreamingTOM，一种免训练即插即用的两阶段框架，以可预测延迟解决LLM前与LLM后瓶颈。Causal Temporal Reduction施加固定每帧预算并基于相邻帧变化与token显著性选择token，通过每帧仅处理紧凑视觉token子集而非所有视觉token大幅降低每帧预填充成本。Online Quantized Memory以4位格式存储token、按需检索相关组并反量化，无论流长度如何保持活跃kv缓存有界。实验证明本文方法实现15.7倍kv缓存压缩、1.2倍更低峰值内存与2倍更快TTFT compared to 先前SOTA。StreamingTOM在免训练方法中保持SOTA精度，离线基准平均63.8%与RVS上55.8%/3.7。这些结果突出本文两阶段方法对高效流式视频理解的实际益处。
- 方法：因果时序缩减：固定每帧token预算，基于帧间变化和token显著性动态选择关键token；在线量化内存：4比特存储token，按需反量化检索，控制kv缓存总量
- 结果：15.7倍kv缓存压缩率（对比SOTA）；峰值内存降低1.2倍，TTFT速度提升2倍
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39087) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_StreamingTOM_Streaming_Token_Compression_for_Efficient_Video_Understanding_CVPR_2026_paper.html)

#### UniCompress: Token Compression for Unified Vision–Language Understanding and Generation

![UniCompress: Token Compression for Unified Vision–Language Understanding and Generation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/cvpr2026-38630.webp)

- 导读：统一模型旨在通过将图像编码为离散token并在单一自回归框架内与文本一起处理，同时支持理解与生成。此统一设计提供架构简洁性与跨模态协同，促进共享参数化、一致训练目标与模态间无缝迁移。然而，此类模型所需的大量视觉token引入 substantial 计算与内存开销，此低效率直接阻碍其在资源受限场景（如具身AI系统）中的部署。本文中，本文提出统一token压缩算法UniCompress，在减少视觉token数量的同时在图像理解与生成任务上保持性能。本文方法引入由可学习全局元token引导的即插即用压缩与解压机制。框架轻量且模块化，无需完全重训练即可高效整合到现有模型中。实验结果表明，本文方法将图像token减少高达4倍，在推理延迟与训练成本上实现显著增益，且仅产生最小性能退化，证明token高效统一建模对真实世界多模态应用的前景。
- 方法：提出插件式压缩/解压缩机制，由可学习全局元标记指导；轻量模块化设计，支持现有模型快速集成
- 结果：图像标记数量减少高达4倍；推理延迟和训练成本显著降低
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38630) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_UniCompress_Token_Compression_for_Unified_Vision-Language_Understanding_and_Generation_CVPR_2026_paper.html)

## AAAI 2026

本轮收录 21 篇。

### P0

#### RGMP: Recurrent Geometric-prior Multimodal Policy for Generalizable Humanoid Robot Manipulation

![RGMP: Recurrent Geometric-prior Multimodal Policy for Generalizable Humanoid Robot Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38539.webp)

- 导读：人形机器人在执行多样化人类技能方面展现出巨大潜力。然而，当前研究主要依赖数据驱动的方法，这些方法需要大量训练数据集才能实现稳健的多模态决策能力和可泛化的视觉运动控制。这些方法因忽视未见场景中的几何推理以及对训练数据中机器人-目标关系建模效率低下，导致训练资源的大量浪费。为解决这些局限性，我们提出了循环几何先验多模态策略（RGMP），这是一个端到端框架，将几何语义技能推理与数据高效的视觉运动控制统一起来。在感知能力方面，我们提出了几何先验技能选择器，它将几何归纳偏差注入视觉语言模型，为未见场景生成自适应技能序列，仅需极少的空间常识调整。为实现数据高效的机器人运动合成，我们引入了自适应递归高斯网络，它将机器人-物体交互参数化为高斯过程的紧凑层次结构，递归编码多尺度空间关系，即使从稀疏演示中也能生成灵巧且数据高效的运动合成。在人形机器人和桌面机器人上的评估表明，RGMP框架在泛化测试中实现了87%的任务成功率，并且比最先进的模型具有5倍的数据效率。这一性能凸显了其卓越的跨领域泛化能力，为更通用且数据高效的机器人系统铺平了道路。
- 方法：提出几何先验技能选择器，将几何归纳偏差注入视觉语言模型。；引入自适应递归高斯网络，递归编码多尺度空间关系。
- 结果：在泛化测试中实现了87%的任务成功率。；比最先进模型具有5倍的数据效率。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38539) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38539)

#### Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective

![Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-37337.webp)

- 导读：随着具身智能体在日益复杂的环境中运行，感知、追踪并推理单个物体实例随时间变化的能力变得至关重要，特别是在需要与视觉相似物体进行序列化交互的任务中。在非马尔可夫场景下，关键决策线索存在于物体历史而非当前场景中。若缺乏对先前交互（如使用过什么、放置在何处或如何变化）的持久记忆，视觉运动策略可能会失败、重复过去动作或忽略已完成操作。为凸显这一挑战，我们提出了LIBERO-Mem——一个用于在物体级部分可观测性下压力测试机器人操作的非马尔可夫任务套件。它结合了短/长时程物体追踪与时间序列子目标，要求推理超越当前帧。然而，视觉-语言-动作（VLA）模型在此类场景中常表现不佳，即使仅跨越几百帧的任务，其标记扩展也迅速变得难以处理。我们提出Embodied-SlotSSM，一个为时序可扩展性设计的以槽位为中心的VLA框架。它通过两种机制保持时空一致的槽位身份并加以利用：（1）用于重建短期历史的槽位状态空间建模；（2）关系编码器对齐输入标记与动作解码。这些组件共同实现了基于时序的上下文感知动作预测。实验表明Embodied-SlotSSM在LIBERO-Mem和通用任务上的基线性能，为以物体为中心策…
- 方法：设计LIBERO-Mem任务套件：结合短/长时程物体追踪与时间序列子目标；提出Embodied-SlotSSM框架：通过槽位状态空间建模重建短期历史
- 结果：在LIBERO-Mem任务套件上建立首个基准性能；实现数百帧级任务的可行标记扩展（对比传统VLA模型）
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-37337) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37337)

### P1

#### Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward

![Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38937.webp)

- 导读：现有强化学习方法在长时程机器人操作任务（尤其是稀疏奖励场景）中表现不佳。虽然动作分块是机器人操作的有前景范式，但如何用强化学习稳定且高效地直接学习连续动作块仍是一个关键挑战。本文提出AC3（连续动作块的行动者-评论家框架），这种新型强化学习框架能学习生成高维连续动作序列。为实现稳定高效的学习，AC3为行动者和评论家设计了针对性稳定机制：首先，行动者采用非对称更新规则，仅从成功轨迹中学习以保证策略可靠改进；其次，评论家通过动作块内n步回报稳定更新，并通过自监督模块在动作块锚点提供内在奖励来增强学习。在BiGym和RLBench基准的25项任务中，AC3仅需少量演示和简单模型架构即可在多数任务上取得更高成功率，验证了其有效性。
- 方法：行动者采用非对称更新规则，仅从成功轨迹学习以保证策略改进；评论家使用动作块内n步回报稳定价值函数更新
- 结果：在BiGym和RLBench的25项任务中验证有效性；仅需少量演示即实现多数任务的更高成功率
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38937) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38937)

#### EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI

![EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-39028.webp)

- 导读：生成建模近期在视觉运动策略学习领域展现出巨大潜力，能够为多样化具身AI任务提供灵活且富有表现力的控制。然而现有生成策略常面临数据效率低下（需大规模演示数据）和采样效率低下（推理时动作生成缓慢）的双重挑战。本文提出EfficientFlow——一个基于流模型的策略学习统一框架。为提升数据效率，我们将等变性引入流匹配过程，理论证明当采用各向同性高斯先验和等变速度预测网络时，生成的动作分布保持等变性，从而显著提升泛化能力并大幅降低数据需求。为加速采样，我们提出新型加速正则化策略：由于边缘流轨迹的加速度直接计算不可行，我们推导出仅需条件轨迹即可实现稳定可扩展训练的替代损失函数。在广泛机器人操作基准测试中，该算法在有限数据条件下取得竞争优势，同时提供极快推理速度，标志着EfficientFlow成为高性能具身AI的高效新范式。
- 方法：将等变性引入流匹配过程以提升数据效率；理论证明各向同性高斯先验+等变网络可保持动作分布等变性
- 结果：在有限数据条件下取得竞争性或更优性能；推理速度显著提升（摘要未提供具体倍数）
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-39028) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/39028)

#### Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic Reinforcement Learning Benchmarks

![Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic Reinforcement Learning Benchmarks](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-39658.webp)

- 导读：学习动态模型（也称为世界模型）可以提高强化学习的样本效率。最近的研究表明，这类动态模型的因果图是稀疏连接的，每个未来状态变量仅依赖于当前状态变量的一小部分，因此学习可能受益于稀疏性先验。同样，时间稀疏性，即局部动态的稀疏和突变变化，也被认为是一种有用的归纳偏差。在本研究中，我们通过分析MuJoCo Playground基准套件中一组机器人强化学习环境的真实动态，批判性地检验了这些假设，旨在确定所提出的状态和时间稀疏性概念是否在典型的强化学习任务中确实成立。我们研究了（i）环境动态的因果图是否稀疏，（ii）这种稀疏性是否依赖于状态，以及（iii）局部系统动态是否稀疏变化。我们的结果表明，全局稀疏性很少见，但任务在其动态中表现出局部、依赖于状态的稀疏性，并且这种稀疏性表现出独特的结构，出现在时间局部化的集群中（例如，在接触事件期间）并影响特定的状态维度子集。这些发现挑战了动态学习中常见的稀疏性先验假设，强调了需要反映现实世界动态中依赖于状态的稀疏性结构的归纳偏差。
- 方法：分析MuJoCo Playground基准套件中机器人强化学习环境的真实动态。；研究环境动态的因果图是否稀疏。
- 结果：全局稀疏性很少见，但任务在其动态中表现出局部、依赖于状态的稀疏性。；稀疏性表现出独特的结构，出现在时间局部化的集群中（例如，在接触事件期间）。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-39658) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/39658)

#### ForeDiffusion: Foresight-Conditioned Diffusion Policy via Future View Construction for Robot Manipulation

![ForeDiffusion: Foresight-Conditioned Diffusion Policy via Future View Construction for Robot Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38934.webp)

- 导读：扩散策略通过逐步去噪高维动作序列推进了视觉运动控制，为机器人操作提供了有前景的方法。然而随着任务复杂度提升，现有基线模型的成功率显著下降。分析表明当前扩散策略面临两大局限：一是仅依赖短期观测作为条件；二是训练目标仍局限于单一去噪损失，导致误差累积和抓取偏差。为此，本文提出前瞻条件扩散策略(ForeDiffusion)，将预测的未来视图表征注入扩散过程，使策略具备前瞻性以修正轨迹偏差。该设计采用双重损失机制，结合传统去噪损失与未来观测一致性损失实现统一优化。在Adroit套件和MetaWorld基准上的大量实验表明，ForeDiffusion总体任务平均成功率达80%，在高难度任务中显著领先主流扩散方法约20%，且在所有任务中保持更稳定表现。
- 方法：将预测的未来视图表征注入扩散过程；采用双重损失机制（去噪损失+未来观测一致性损失）
- 结果：总体任务平均成功率80%；高难度任务性能领先主流方法约20%
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38934) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38934)

#### Gentle Manipulation Policy Learning via Demonstrations from VLM Planned Atomic Skills

![Gentle Manipulation Policy Learning via Demonstrations from VLM Planned Atomic Skills](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38955.webp)

- 导读：传统上，自主执行长时间、接触密集的操作任务需要大量真实世界数据和专家工程，带来高昂成本和可扩展性挑战。本文提出一种新颖框架，整合分层语义分解、强化学习(RL)、视觉语言模型(VLM)和知识蒸馏来克服这些限制。复杂任务被分解为原子技能，每个基本动作的RL策略仅在仿真中训练。关键的是，我们的RL公式包含显式力约束以防止精细交互中的物体损坏。VLM执行高层任务分解和技能规划，生成多样化专家示范。这些示范通过视觉-触觉扩散策略蒸馏为统一策略以实现端到端执行。我们进行了全面消融研究，探索不同基于VLM的任务规划器以确定最佳示范生成流程，并系统比较了技能蒸馏的模仿学习算法。大量仿真实验和物理部署验证了我们的方法无需昂贵人类示范即可实现长时间操作策略学习，同时VLM引导的原子技能框架实现了对多样化任务的可扩展泛化。
- 方法：分层语义分解：用VLM将复杂任务拆解为原子技能；仿真RL训练：每个原子技能单独训练且加入力约束保护物体
- 结果：无需真实人类示范实现长时程操作策略学习；VLM原子技能框架支持多样化任务泛化
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38955) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38955)

#### LatentVLA: Taming Latent Space for Generalizable and Long-Horizon Bimanual Manipulation

![LatentVLA: Taming Latent Space for Generalizable and Long-Horizon Bimanual Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38926.webp)

- 导读：当前的机器人模仿学习范式面临一个明显的权衡：扩散模型的运动保真度与逆动力学模型的数据可扩展性。后者虽然可扩展，但往往学习的潜在动作空间与物理现实脱节。这一缺陷导致了关键失败：时间纠缠，即模型无法区分视觉上相似但需要不同动作的状态，例如夹爪接近或远离物体。这种模糊性，加上离散化伪影和对任务无关动态的敏感性，使得稳健规划变得不可行。我们提出了LatentVLA，一个视觉-语言-动作框架，旨在通过学习连续且时空接地的潜在动作表示来克服这些限制。其渐进的三阶段架构首先采用时间注意潜在动作模型（TA-LAM），通过语言引导的注意力和显式时间编码来解决模糊性。随后，潜在动作扩散变换器（LADT）直接在这个连续潜在空间内通过扩散进行规划，保持运动保真度而无需标记化。最后，专家策略头将这些潜在计划转化为精确的机器人动作。实验表明，LatentVLA在一系列现实世界的双手任务中设定了新的最先进水平，优于先前的方法，并展示了卓越的零样本泛化和少样本效率。
- 方法：采用时间注意潜在动作模型（TA-LAM）通过语言引导的注意力和显式时间编码来解决模糊性。；使用潜在动作扩散变换器（LADT）在连续潜在空间内通过扩散进行规划，保持运动保真度。
- 结果：LatentVLA在一系列现实世界的双手任务中设定了新的最先进水平。；LatentVLA展示了卓越的零样本泛化能力。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38926) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38926)

#### Learning Diffusion Policy from Primitive Skills for Robot Manipulation

![Learning Diffusion Policy from Primitive Skills for Robot Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38889.webp)

- 导读：扩散策略在机器人动作生成领域展现出巨大潜力，但现有方法依赖全局指令生成短期控制信号，易导致动作失准。本文提出原始技能（如'上移'、'张开夹爪'等细粒度短时操作）能为机器人学习提供更直观有效的接口。为此我们提出SDP——一种融合可解释技能学习与条件动作规划的技能条件扩散策略。SDP抽象出八种跨任务可复用原始技能，采用视觉语言模型从观测和语言指令中提取离散表征，并设计轻量路由网络为每个状态分配目标技能，进而构建单技能策略生成技能对齐的动作。通过将复杂任务分解为原始技能序列并选择单技能策略，SDP确保跨任务的技能一致性行为。在两大仿真基准和真实机器人部署中的大量实验表明，SDP持续超越现有最优方法，为基于技能的扩散策略学习提供了新范式。
- 方法：抽象出8种跨任务可复用原始技能（如'上移'、'张开夹爪'）；用视觉语言模型提取视觉观测与语言指令的离散表征
- 结果：在两大仿真基准上持续超越现有最优方法；真实机器人部署验证有效性
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38889) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38889)

#### MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation

![MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38919.webp)

- 导读：在机器人操作领域，机器人学习已成为主流方法。然而，该领域中的生成模型面临一个基本权衡：扩散模型的缓慢迭代采样与基于Flow的快速方法之间的架构限制，后者通常依赖于显式的一致性损失。为了解决这些限制，我们提出了MP1，它将3D点云输入与MeanFlow范式结合，通过一次网络函数评估（1-NFE）生成动作轨迹。通过“MeanFlow Identity”直接学习区间平均速度，我们的策略避免了任何额外的一致性约束。这种公式消除了推理过程中数值ODE求解器的误差，从而生成更精确的轨迹。MP1进一步结合了CFG以提高轨迹可控性，同时保持1-NFE推理而不重新引入结构约束。由于细微的场景变化对机器人学习至关重要，尤其是在少样本学习中，我们引入了一种轻量级的分散损失，在训练过程中排斥状态嵌入，从而在不减慢推理的情况下提升泛化能力。我们在Adroit和Meta-World基准测试以及实际场景中验证了我们的方法。实验结果表明，MP1在平均任务成功率上表现优异，分别比DP3高出10.2%，比FlowPolicy高出7.3%。其平均推理时间仅为6.8毫秒——比DP3快19倍，比FlowPolicy快近2倍。
- 方法：将3D点云输入与MeanFlow范式结合，通过一次网络函数评估生成动作轨迹。；通过“MeanFlow Identity”直接学习区间平均速度，避免额外的一致性约束。
- 结果：MP1在平均任务成功率上比DP3高出10.2%，比FlowPolicy高出7.3%。；MP1的平均推理时间仅为6.8毫秒，比DP3快19倍，比FlowPolicy快近2倍。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38919) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38919)

#### ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation

![ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38881.webp)

- 导读：单次模仿学习(OSIL)为无需大规模数据采集的机器人技能学习提供了有前景的途径。然而，现有OSIL方法主要局限于短时程任务，难以应用于复杂的长时程操作。为此，我们提出ManiLong-Shot框架，通过围绕物理交互事件构建长时程任务，将问题重构为交互感知基元序列而非直接模仿连续轨迹。该框架可由视觉语言模型(VLM)的高层推理驱动，或基于机器人状态变化的启发式规则实现。针对每个基元，ManiLong-Shot预测交互关键的不变区域，建立演示与当前观察的对应关系，并计算机械臂目标位姿以实现有效任务执行。大量仿真实验表明，仅需10个短时程任务训练的ManiLong-Shot，通过单次模仿能泛化至20个未见长时程任务（覆盖三个难度等级），相对现有技术提升22.8%。真实机器人实验进一步验证了其通过OSIL稳健执行三个长时程操作任务的能力。
- 方法：将长时程任务解构为交互感知基元序列而非连续轨迹；通过视觉语言模型或状态启发式规则驱动基元选择
- 结果：仅需10个短时程任务训练即可泛化至20个未见长时程任务；在三个难度等级上相对SOTA提升22.8%
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38881) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38881)

#### ManipDreamer3D: Synthesizing Plausible Robotic Manipulation Video with Occupancy-aware 3D Trajectory

![ManipDreamer3D: Synthesizing Plausible Robotic Manipulation Video with Occupancy-aware 3D Trajectory](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-37595.webp)

- 导读：数据稀缺仍然是机器人操作领域的关键瓶颈，限制了训练鲁棒且可泛化模型的能力。虽然扩散模型为合成逼真的机器人操作视频提供了有前景的途径，但其效果依赖于精确合理的控制指令。当前方法主要依赖2D轨迹作为指令提示，这本质上存在3D空间模糊性问题。本文提出了一种名为ManipDreamer3D的新框架，可从输入图像和文本指令生成具有3D感知的合理机器人操作视频。我们的方法将3D轨迹规划与基于第三人称视角重建的3D占据地图相结合，并采用了一种新颖的轨迹到视频扩散模型。具体而言，ManipDreamer3D首先从输入图像重建3D占据表示，然后计算优化的3D末端执行器轨迹，最小化路径长度、避免碰撞并重新计时。接着，我们采用潜在编辑技术，从初始图像潜在、文本指令和优化的3D轨迹创建视频序列。这一过程通过我们专门训练的轨迹到视频扩散模型生成机器人抓放视频。我们的方法通过自主规划合理的3D轨迹，显著减少了人工干预需求。实验结果表明其具有卓越的视觉质量和精度。
- 方法：结合3D轨迹规划与第三人称视角重建的3D占据地图；提出轨迹到视频扩散模型生成操作视频
- 结果：显著减少人工干预需求；实验证明具有卓越的视觉质量和精度
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-37595) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37595)

#### ODYSSEY: Open-World Quadrupeds Exploration and Manipulation for Long-Horizon Tasks

![ODYSSEY: Open-World Quadrupeds Exploration and Manipulation for Long-Horizon Tasks](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38927.webp)

- 导读：语言引导的长时程移动操作长期以来是具身语义推理、泛化操作和自适应运动领域的重大挑战。三个根本性局限阻碍了进展：首先，尽管大语言模型通过习得的语义先验在增强空间推理和任务规划方面展现出潜力，现有实现仍局限于桌面场景，未能解决移动平台特有的感知受限和执行范围有限的问题。其次，当前操作策略在开放世界中遇到多样化物体配置时表现出泛化能力不足。第三，尽管对实际部署至关重要，但在非结构化环境中同时保持高平台机动性和精确末端执行器控制的双重要求在文献中仍研究不足。本文提出ODYSSEY——一种为配备机械臂的敏捷四足机器人设计的统一移动操作框架，将高层任务规划与低层全身控制无缝集成。针对语言条件任务中自我中心感知的挑战，我们引入由视觉语言模型驱动的分层规划器，实现长时程指令分解和精确动作执行。在控制层面，新颖的全身策略实现了跨挑战性地形的稳健运动与操作协调。我们进一步提出首个长时程移动操作综合基准，评估多样室内外场景。通过成功的仿真到现实迁移，我们验证了系统在真实部署中的泛化性和鲁棒性，突显了腿式操作器在非结构化环境中的实用性。本工作推进了能够执行复杂动态任务的通用机器人助手的可行性。
- 方法：开发统一框架整合高层语言任务规划与低层全身控制；采用视觉语言模型驱动的分层规划器解决自我中心感知问题
- 结果：实现仿真到现实的成功迁移验证系统实用性；在多样化室内外场景中展示泛化能力
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38927) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38927)

#### Steering Visuomotor Policy in Open Worlds via Cross-View Goal Alignment

![Steering Visuomotor Policy in Open Worlds via Cross-View Goal Alignment](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38876.webp)

- 导读：我们的目标是开发一种语义清晰、空间敏感、领域无关且对人类用户直观的目标指定方法，用于指导3D环境中的智能体交互。具体而言，我们提出了一种新颖的跨视角目标对齐框架，允许用户通过其相机视角的分割掩码而非智能体的观察来指定目标物体。我们指出，当人类与智能体的相机视角差异显著时，仅靠行为克隆无法使智能体行为与人类意图对齐。为此，我们引入了两个辅助目标：跨视角一致性损失和目标可见性损失，显式增强了智能体的空间推理能力。基于此，我们开发了ROCKET-2，一个在Minecraft中训练的最先进智能体，推理效率提升了3至6倍。我们证明ROCKET-2可以直接从人类相机视角解读目标，实现更好的人机交互。值得注意的是，ROCKET-2展示了零样本泛化能力：尽管仅在Minecraft数据集上训练，它可以通过简单的动作空间映射适应并泛化到其他3D环境如Doom、DMLab和Unreal。
- 方法：提出跨视角目标对齐框架，用人类视角的分割掩码指定目标；引入跨视角一致性损失增强空间推理
- 结果：推理效率提升3-6倍（Minecraft环境）；实现从人类视角直接解读目标的零样本泛化
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38876) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38876)

#### TCoT: Trajectory Chain-of-Thoughts for Robotic Manipulation with Failure Recovery in Vision-Language-Action Model

![TCoT: Trajectory Chain-of-Thoughts for Robotic Manipulation with Failure Recovery in Vision-Language-Action Model](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-37577.webp)

- 导读：近期视觉-语言-动作（VLA）模型在机器人操作任务中展现出卓越的泛化能力。然而，这些模型通常直接将视觉和语言输入映射为后续动作，缺乏中间任务规划、失败检测与恢复能力，导致其难以有效分解复杂任务、识别问题并纠正错误动作，最终造成任务完全失败。这严重限制了其执行长期任务和泛化的能力。为此，我们提出TCoT：轨迹思维链，一个统一的VLA框架，通过轨迹规划及失败检测与恢复机制增强直接映射。TCoT利用层次化轨迹作为操作任务中思维链推理的精确简洁表示：全局规划提供面向目标的高层轨迹以引导机器人完成任务，而局部规划专注于实时调整以应对动态变化。此外，我们设计了全局-局部切换恢复算法来检测并有效恢复失败。实验结果表明，TCoT在真实和模拟场景中均超越现有最优方法，并展现出更优的泛化能力。
- 方法：提出层次化轨迹表示（全局规划+局部调整）作为思维链推理载体；设计全局-局部切换恢复算法实现失败检测与自主修正
- 结果：在真实与模拟场景中全面超越现有最优方法（摘要未提具体数值）；展现出更优异的任务泛化能力
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-37577) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37577)

### P2

#### Backtrace Mamba: Reviving Critical Temporal Contexts via Hierarchical Memory Compression for Online Action Detection

![Backtrace Mamba: Reviving Critical Temporal Contexts via Hierarchical Memory Compression for Online Action Detection](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38139.webp)

- 导读：在线动作检测(OAD)需要实时预测正在进行的动作而无法访问未来帧，这给计算效率和长期依赖建模之间的平衡带来了挑战。现有方法要么训练速度慢且时间感受野有限，要么面临高计算成本和延迟推理的问题，缺乏处理超长视频输入的能力。为此，我们提出了一种基于Mamba的新型OAD框架(MOAD)，能够高效且有效地执行OAD。引入了分层记忆机制，基于运动感知相似性度量智能存储高价值的场景和动作帧，以在线方式保存关键历史知识。为了进一步减少存储空间，我们设计了一种记忆量化方法来压缩存储的历史特征。此外，还提出了基于记忆库的时间软剪枝策略，动态去除冗余特征，在保持时间连贯性的同时减少时间冗余。在四个具有挑战性的基准测试上的充分实验证明，我们的方法显著优于现有方法。
- 方法：引入分层记忆机制智能存储高价值帧；设计记忆量化方法压缩历史特征
- 结果：在四个基准测试上显著优于现有方法；有效平衡计算效率和长期依赖建模
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38139) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38139)

#### CogStream: Context-guided Streaming Video Question Answering

![CogStream: Context-guided Streaming Video Question Answering](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38336.webp)

- 导读：尽管视频大型语言模型(Vid-LLMs)在多模态理解方面取得了进展，但由于其对上下文信息的依赖，流式视频推理仍面临挑战。现有范式将所有可用的历史上下文信息输入Vid-LLMs，导致视觉数据处理的计算负担显著增加。此外，无关上下文的包含会分散模型对关键细节的注意力。本文引入了一个名为'上下文引导的流式视频推理'(CogStream)的挑战性任务，该任务模拟现实世界的流式视频场景，要求模型识别最相关的历史上下文信息以推断关于当前流的问题答案。为支持CogStream，我们提出了一个密集注释的数据集，包含由半自动管道生成的大量分层问答对。此外，我们还提出了CogReasoner作为基线模型，它通过利用视觉流压缩和历史对话检索有效解决了这一任务。大量实验证明了该方法的有效性。
- 方法：提出CogStream任务框架模拟真实流式视频场景；开发半自动管道生成密集注释的分层问答数据集
- 结果：创建了包含大量分层问答对的标注数据集；CogReasoner模型有效解决了流式视频问答任务
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38336) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38336)

#### CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking

![CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-38385.webp)

- 导读：LiDAR点云中的3D单目标跟踪(SOT)是计算机视觉和自动驾驶领域的关键任务。尽管已取得重大进展，但点云固有的稀疏性带来了双重冗余挑战：(1)背景噪声导致的空间冗余影响精度，(2)前景内部的信息冗余阻碍效率。为此，我们提出CompTrack，一种端到端框架，系统性地消除点云中的这两种冗余。首先，CompTrack通过基于信息熵的空间前景预测器(SFP)模块过滤无关背景噪声，解决空间冗余问题；其核心是信息瓶颈引导的动态令牌压缩(IB-DTC)模块，通过低秩近似理论，利用在线SVD分析自适应地将冗余前景压缩为一组紧凑且信息丰富的代理令牌。在KITTI、nuScenes和Waymo数据集上的大量实验表明，CompTrack以90 FPS的实时性能在单块RTX 3090 GPU上实现了顶尖的跟踪效果。
- 方法：空间前景预测器(SFP)模块基于信息熵过滤背景噪声；信息瓶颈引导的动态令牌压缩(IB-DTC)模块通过在线SVD分析压缩前景冗余
- 结果：在KITTI、nuScenes和Waymo数据集上实现顶尖跟踪性能；单块RTX 3090 GPU上达到90 FPS实时速度
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38385) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38385)

#### Contribution-aware Token Compression for Efficient Video Understanding via Reinforcement Learning

![Contribution-aware Token Compression for Efficient Video Understanding via Reinforcement Learning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-39612.webp)

- 导读：视频大语言模型在视频理解任务中展现出卓越能力，但视频令牌的冗余性导致推理时计算开销巨大，限制了实际部署。现有压缩算法通常优先保留注意力分数最高的特征以最小化注意力计算扰动，然而注意力分数与对正确答案的实际贡献之间的关联尚不明确。为此，我们提出了一种新颖的贡献感知令牌压缩算法CaCoVID，其核心在于根据令牌对正确预测的贡献度显式优化选择策略。首先，我们引入基于强化学习的框架，通过策略网络主动筛选对正确预测贡献最大的视频令牌组合，实现从被动保留到主动探索最优压缩组合的范式转变。其次，我们提出带在线组合空间采样的组合策略优化算法，大幅缩减视频令牌组合的探索空间并加速策略收敛。跨多个视频理解基准的实验验证了CaCoVID的有效性。代码将开源。
- 方法：强化学习框架优化策略网络，主动选择对正确预测贡献最大的令牌组合；在线组合空间采样算法大幅降低探索空间并加速策略收敛
- 结果：在多个视频理解基准测试中验证有效性（摘要未提具体指标）；代码将开源（实现可复现性）
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-39612) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/39612)

#### Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation

![Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-37386.webp)

- 导读：图像反射分离旨在从混合图像中分离出透射层和反射层。现有方法依赖于单幅图像的有限信息，当两层对比度相似时容易混淆，这一挑战在夜间更为严峻。为解决此问题，我们提出深度记忆解耦网络(DMDNet)。它采用深度感知扫描(DAScan)引导Mamba关注显著结构，通过语义连贯性促进信息流动以构建稳定状态。与DAScan协同工作的深度协同状态空间模型(DS-SSM)通过深度调节状态激活的敏感性，抑制干扰层分离的模糊特征扩散。此外，我们引入记忆专家补偿模块(MECM)，利用跨图像历史知识指导专家提供层特异性补偿。针对夜间反射分离数据集的缺乏，我们构建了夜间图像反射分离(NightIRS)数据集。大量实验表明DMDNet在白天和夜间均优于最先进方法。
- 方法：深度感知扫描(DAScan)引导Mamba关注语义连贯的显著结构；深度协同状态空间模型(DS-SSM)通过深度调节抑制模糊特征扩散
- 结果：构建首个夜间反射分离数据集NightIRS；白天/夜间均超越现有最优方法(具体数值见原文)
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-37386) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37386)

#### Vista: Scene-Aware Optimization for Streaming Video Question Answering Under Post-Hoc Queries

![Vista: Scene-Aware Optimization for Streaming Video Question Answering Under Post-Hoc Queries](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/aaai2026-article-37694.webp)

- 导读：流式视频问答（Streaming Video QA）对多模态大语言模型（MLLMs）提出了独特的挑战，因为视频帧是顺序到达的，而用户查询可以在任意时间点发出。现有的解决方案依赖于固定大小的内存或简单的压缩，往往会导致上下文丢失或内存溢出，限制了它们在长时间实时场景中的有效性。我们提出了Vista，一种用于场景感知流式视频问答的新框架，能够在连续视频流上进行高效且可扩展的推理。Vista的创新可以总结为三个方面：（1）场景感知分割。Vista动态地将传入的帧聚类为时间和视觉上连贯的场景单元。（2）场景感知压缩。每个场景被压缩成一个紧凑的令牌表示，并存储在GPU内存中以进行高效的基于索引的检索，而全分辨率帧则卸载到CPU内存中。（3）场景感知召回。在接收到问题时，相关场景被选择性地召回并重新整合到模型的输入空间中，从而实现效率和完整性的平衡。Vista是模型无关的，可以与各种视觉语言骨干无缝集成，在不影响延迟或内存效率的情况下实现长上下文推理。在StreamingBench上的大量实验表明，Vista实现了最先进的性能，为现实世界中的流式视频理解建立了强大的基线。
- 方法：动态聚类传入帧为时间和视觉上连贯的场景单元。；将每个场景压缩为紧凑的令牌表示并存储在GPU内存中。
- 结果：Vista在StreamingBench上实现了最先进的性能。；Vista为现实世界中的流式视频理解建立了强大的基线。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-37694) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37694)

## WACV 2026

本轮收录 2 篇。

### P0

#### Memory-Augmented Representation for Efficient Event-based Visuomotor Policy Learning with Adaptive Perception and Control

![Memory-Augmented Representation for Efficient Event-based Visuomotor Policy Learning with Adaptive Perception and Control](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/wacv2026-Kamal_Memory-Augmented_Representation_for_Efficient_Event-ba.webp)

- 导读：事件相机因其超快的微秒级时间分辨率，非常适合快速敏捷的自主导航。然而，充分发挥这一潜力需要高效的异步处理算法，能够逐事件更新表示和策略。现有方法采用同步密集表示或在固定速率时间窗口内处理事件，导致冗余计算效率低下。我们通过提出一种端到端的事件到控制策略学习框架来解决这一问题，该框架专为反应式导航任务设计。我们的方法包括一个记忆增强的感知模块，该模块异步更新表示并自适应选择处理的事件数量。利用记忆表示，轻量级策略模块与感知模块联合优化，并在基于事件的强化学习设置中动态调整预测控制命令的速率以适应场景复杂性。在模拟无人机导航任务中的评估表明，与基于密集帧的方法相比，我们的方法具有更高的样本效率和鲁棒性。此外，我们的方法通过最小化处理步骤和事件数量，显著降低了计算复杂度，同时保持了与最先进的事件相机方法相当的性能。
- 方法：提出了一种端到端的事件到控制策略学习框架。；设计了一个记忆增强的感知模块，异步更新表示并自适应选择处理的事件数量。
- 结果：在模拟无人机导航任务中表现出更高的样本效率和鲁棒性。；显著降低了计算复杂度，同时保持了与最先进的事件相机方法相当的性能。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/wacv2026-Kamal_Memory-Augmented_Representation_for_Efficient_Event-ba) · [原文](https://openaccess.thecvf.com/content/WACV2026/html/Kamal_Memory-Augmented_Representation_for_Efficient_Event-based_Visuomotor_Policy_Learning_with_Adaptive_WACV_2026_paper.html)

### P2

#### Online Episodic Memory Visual Query Localization with Egocentric Streaming Object Memory

![Online Episodic Memory Visual Query Localization with Egocentric Streaming Object Memory](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/wacv2026-Manigrasso_Online_Episodic_Memory_Visual_Query_Localization_.webp)

- 导读：情景记忆检索使可穿戴相机能够回忆视频中先前观察到的物体或事件。然而，现有的方法假设在查询时可以完全访问视频的“离线”设置，这限制了它们在现实世界场景中的应用，尤其是在功耗和存储受限的可穿戴设备上。为了开发更适用于实际应用的情景记忆系统，我们引入了在线视觉查询2D（OVQ2D）任务，其中模型在线处理视频流，每帧只观察一次，并使用紧凑的内存而不是完整的视频历史记录来检索物体定位。我们通过ESOM（Egocentric Streaming Object Memory）框架来解决OVQ2D问题，该框架集成了物体发现模块、物体跟踪模块和内存模块，用于查找、跟踪和存储时空物体信息以实现高效查询。在Ego4D上的实验表明，ESOM优于其他在线方法，尽管OVQ2D仍然具有挑战性，最高成功率仅为4%。ESOM的准确性在完美物体跟踪（31.91%）、发现（40.55%）或两者兼备（81.92%）时显著提高，强调了这些组件应用研究的必要性。
- 方法：引入在线视觉查询2D（OVQ2D）任务，模型在线处理视频流，每帧只观察一次。；提出ESOM框架，集成物体发现、跟踪和内存模块。
- 结果：在Ego4D数据集上，ESOM优于其他在线方法，尽管OVQ2D任务仍然具有挑战性，最高成功率仅为4%。；ESOM的准确性在完美物体跟踪时显著提高至31.91%。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/wacv2026-Manigrasso_Online_Episodic_Memory_Visual_Query_Localization_) · [原文](https://openaccess.thecvf.com/content/WACV2026/html/Manigrasso_Online_Episodic_Memory_Visual_Query_Localization_with_Egocentric_Streaming_Object_WACV_2026_paper.html)

## AAMAS 2026

本轮收录 0 篇。

本轮未发现明确相关论文。

## ACL 2026

本轮收录 3 篇。

### P0

#### Conflict-Aware Memory for Embodied Agents: Enhancing Vector Data Quality via Detection Rules

![Conflict-Aware Memory for Embodied Agents: Enhancing Vector Data Quality via Detection Rules](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/acl2026-7cc20c97f79e.webp)

- 导读：该方法利用冲突检测规则改善具身智能体向量记忆的数据质量与任务规划可靠性。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2026-7cc20c97f79e)

### P2

#### Beyond Markovian Forgetfulness: Episodic Memory for Reasoning-Intensive Retrieval

![Beyond Markovian Forgetfulness: Episodic Memory for Reasoning-Intensive Retrieval](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/acl2026-c3258eb5eaab.webp)

- 导读：该工作为推理密集型检索引入情景记忆，以缓解仅依赖当前查询造成的信息遗忘。
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2026-c3258eb5eaab)

#### GMSA: Enhancing Context Compression via Group Merging and Layer Semantic Alignment

![GMSA: Enhancing Context Compression via Group Merging and Layer Semantic Alignment](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2026_index/assets/covers/acl2026-192bf264ea8c.webp)

- 导读：GMSA通过分组合并与层级语义对齐压缩长上下文，属于可迁移的上下文压缩方法。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2026-192bf264ea8c)

## EACL 2026

本轮收录 2 篇。

### P2

#### H-Mem: Hybrid Multi-Dimensional Memory Management for Long-Context Conversational Agents.

- 导读：H-Mem以混合多维记忆管理机制组织长上下文对话智能体的不同类型历史信息。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/dblp:conf/eacl/YeHCZ26) · [原文](https://doi.org/10.18653/v1/2026.eacl-long.363)

#### Tailoring Memory Granularity for Multi-Hop Reasoning over Long Contexts.

- 导读：该工作研究多跳长上下文推理中的记忆粒度选择，以及压缩粒度对证据保留的影响。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/dblp:conf/eacl/QingDMHV26) · [原文](https://doi.org/10.18653/v1/2026.findings-eacl.189)

## WSDM 2026

本轮收录 0 篇。

本轮未发现明确相关论文。
