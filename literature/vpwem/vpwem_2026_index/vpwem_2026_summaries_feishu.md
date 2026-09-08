# 2026 顶会与 VPWEM 相关论文：快速阅读版

更新时间：2026-07-21

共 122 篇。摘要优先采用 Connected Researchers 中文导读；本版用于快速定位，关键结论仍需回到原文核验。

阅读顺序：先看 P0，再看 P1；P2 仅在设计压缩或流式记忆实验时查阅。

## ICLR 2026

本轮收录 27 篇。

### P0

**Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative**

- 一句话：扩散策略的惊人性能源于动作记忆，而一个简单的查找表就能达到类似效果。
- 对 VPWEM：关键反例：用于检验策略是否只是复现动作或轨迹。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-PL0tJOfm7I-90e4971575) · [原文](https://openreview.net/forum?id=PL0tJOfm7I)

**Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization**

- 一句话：通过分层知识图内存模块，显著提升具身代理的个性化任务表现。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-E5L43l5EIu-76e1d62d28) · [原文](https://openreview.net/forum?id=E5L43l5EIu)

**HAMLET: Switch Your Vision-Language-Action Model into a History-Aware Policy**

- 一句话：HAMLET将视觉-语言-动作模型升级为历史感知策略，显著提升依赖上下文的机器人任务性能。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-KcJ9U0x6kO-5e81815031) · [原文](https://openreview.net/forum?id=KcJ9U0x6kO)

**Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning**

- 一句话：MIKASA：首个记忆强化学习的综合基准，推动智能体在复杂任务中的记忆能力评估。
- 对 VPWEM：直接评测依据：可用于补充记忆任务与指标。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-9cLPurIZMj-9f2377caa1) · [原文](https://openreview.net/forum?id=9cLPurIZMj)

**MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation**

- 一句话：MemoryVLA以工作记忆和长期感知-认知记忆库保存时序上下文，并用记忆条件扩散专家生成动作。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-54U3XHf7qq-981996a4fc) · [原文](https://openreview.net/forum?id=54U3XHf7qq)

**Scaling up Memory for Robotic Control via Experience Retrieval**

- 一句话：MemER从长时机器人经验中检索任务相关关键帧，再由层级策略利用这些证据完成分钟级任务。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-1dH4ARGdwD-0aa94ae7c2) · [原文](https://openreview.net/forum?id=1dH4ARGdwD)

### P1

**Ctrl-World: A Controllable Generative World Model for Robot Manipulation**

- 一句话：提出可控多视角世界模型 Ctrl-World，无需真实机器人 rollout 即可评估和提升通用策略的指令跟随能力，成功率提升 44.7%。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-748bHL2BAv-84770bc914) · [原文](https://openreview.net/forum?id=748bHL2BAv)

**DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving**

- 一句话：DriveVLA-W0通过世界建模放大自动驾驶的数据缩放效应，让模型性能随数据增长加速提升
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-plrGn3RdzN-602fb368ee) · [原文](https://openreview.net/forum?id=plrGn3RdzN)

**Empowering Multi-Robot Cooperation via Sequential World Models**

- 一句话：SeqWM用顺序世界模型刻画多机器人之间的协作过程，为跨主体时序建模提供参考。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-IvUM6UwYCJ-8bad520056) · [原文](https://openreview.net/forum?id=IvUM6UwYCJ) · [代码](https://github.com/zhaozijie2022/seqwm)

**H$^3$DP: Triply‑Hierarchical Diffusion Policy for Visuomotor Learning**

- 一句话：H$^3$DP通过三重分层结构显著提升了视觉运动策略中感知与动作的耦合性能。
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-Q1CP0iAmOb-1adba80f6f) · [原文](https://openreview.net/forum?id=Q1CP0iAmOb)

**Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control**

- 一句话：该方法通过协同轨迹控制生成机器人操作视频，重点处理多主体与多物体之间的运动一致性。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-OeDwYtp8n1-64271e301d) · [原文](https://openreview.net/forum?id=OeDwYtp8n1)

**Real-Time Robot Execution with Masked Action Chunking**

- 一句话：REMAC通过掩码动作块校正和前缀保留采样，解决了异步推理中块内不一致性问题，实现更可靠的实时机器人操作。
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-r0RGJ1j9on-6789bd231b) · [原文](https://openreview.net/forum?id=r0RGJ1j9on)

**SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation**

- 一句话：SARM以阶段感知奖励模型处理长时程操作中的演示质量差异，并改善多阶段任务学习。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-aemqAxScl9-d56aaf448a) · [原文](https://openreview.net/forum?id=aemqAxScl9)

**Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments**

- 一句话：该方法在测试时动态混合多个世界模型，使具身智能体适应持续变化的环境动力学。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-LQD1MrnbxH-1f89c32997) · [原文](https://openreview.net/forum?id=LQD1MrnbxH)

**WMPO: World Model-based Policy Optimization for Vision-Language-Action Models**

- 一句话：WMPO：无需真实环境交互，基于像素预测的视觉-语言-动作模型强化学习框架，显著提升样本效率和性能。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-qE2FyvRvuF-12550585ae) · [原文](https://openreview.net/forum?id=qE2FyvRvuF)

**villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models**

- 一句话：villa-X联合视觉、语言与潜在动作表示，增强VLA模型的动作规划和生成能力。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-y5CaJb17Fn-6ac90d96f8) · [原文](https://openreview.net/forum?id=y5CaJb17Fn)

### P2

**COMI: Coarse-to-fine Context Compression via Marginal Information Gain**

- 一句话：COMI：通过边际信息增益实现从粗到细的上下文压缩，大幅提升长上下文处理效率。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-OGDIXDfaN4-7c6c53e9fc) · [原文](https://openreview.net/forum?id=OGDIXDfaN4)

**Composition of Memory Experts for Diffusion World Models**

- 一句话：用'记忆专家委员会'破解世界模型的记忆悖论：扩散框架整合三专家，既保细节又省算力。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-sUEdpZCHdp-e92e8e2b7e) · [原文](https://openreview.net/forum?id=sUEdpZCHdp)

**FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding**

- 一句话：FLoC框架通过设施位置函数高效压缩视觉标记，显著提升长视频理解的处理效率。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-tPhcrP75OP-df41e33f50) · [原文](https://openreview.net/forum?id=tPhcrP75OP)

**MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding**

- 一句话：MARC通过记忆增强强化学习实现视频token智能压缩，在保持精度的同时减少95%计算量。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-5gLgjn3MVT-3bae051ec4) · [原文](https://openreview.net/forum?id=5gLgjn3MVT)

**Memento: Toward an All-Day Proactive Assistant for Ultra-Long Streaming Video**

- 一句话：Memento：首个实现全天候超长视频流主动理解的视觉-语言框架。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-FtdbdoGbk3-73ac99ea3c) · [原文](https://openreview.net/forum?id=FtdbdoGbk3)

**Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity**

- 一句话：用协同双阶段剪枝法，在保留核心语义的同时将视觉令牌压缩至原数5.6%，精度损失仅3.7%
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-i36E5Ezm0H-0c4d7af46a) · [原文](https://openreview.net/forum?id=i36E5Ezm0H) · [代码](https://github.com/ZhengyaoFang/PruneSID)

**QueryStream: Advancing Streaming Video Understanding with Query-Aware Pruning and Proactive Response**

- 一句话：QueryStream通过查询感知剪枝和主动响应机制，在视频流理解中实现高效计算与精准时机的完美平衡。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-738HjJEbml-c010eb64ca) · [原文](https://openreview.net/forum?id=738HjJEbml)

**REMem: Reasoning with Episodic Memory in Language Agent**

- 一句话：REMem通过两阶段框架显著提升了语言代理的情景记忆和推理能力。
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-fugnQxbvMm-e62f715ff0) · [原文](https://openreview.net/forum?id=fugnQxbvMm)

**RMAAT: Astrocyte-Inspired Memory Compression and Replay for Efficient Long-Context Transformers**

- 一句话：受星形胶质细胞启发的记忆压缩与回放机制，让Transformer高效处理长上下文不再头疼。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-sTkJdbVxsI-e73aff2703) · [原文](https://openreview.net/forum?id=sTkJdbVxsI)

**UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning**

- 一句话：UltraMemV2通过五项革新将内存层架构性能提升至8专家MoE水平，同时大幅降低内存访问成本。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-QWuXU0qNX0-521375583d) · [原文](https://openreview.net/forum?id=QWuXU0qNX0)

**VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling**

- 一句话：提出分层压缩技术HiCo，让AI用1/50计算量无损处理超长视频，并打造首个10,000帧99.1%准确率的开源视频模型。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2026-MUjdNcfNPv-658cf6598e) · [原文](https://openreview.net/forum?id=MUjdNcfNPv) · [代码](https://github.com/OpenGVLab/VideoChat-Flash)

## ICML 2026

本轮收录 32 篇。

### P0

**HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control**

- 一句话：分层记忆框架让机器人同时拥有闪电般的反应和深思熟虑的规划能力
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-60897) · [原文](https://icml.cc/virtual/2026/poster/60897)

**RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies**

- 一句话：首个标准化机器人记忆能力评测基准RoboMME，揭示VLA模型记忆机制的任务依赖性
- 对 VPWEM：直接评测依据：可用于补充记忆任务与指标。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65933) · [原文](https://icml.cc/virtual/2026/poster/65933)

**Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action**

- 一句话：该方法为VLA维护视野外物体的空间记忆，使机器人在目标暂时不可见时仍能继续操作。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-66214) · [原文](https://icml.cc/virtual/2026/poster/66214)

### P1

**Cross-Embodiment Robot Foundation World Models with Latent Actions**

- 一句话：提出统一潜在动作空间的世界模型LAC-WM，跨机器人形态性能提升46.7%
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-63978) · [原文](https://icml.cc/virtual/2026/poster/63978)

**Sample from What You See: Visuomotor Policy Learning via Diffusion Bridge with Observation-Embedded Stochastic Differential Equation**

- 一句话：通过扩散桥将视觉观测直接整合到动作生成动力学中，实现更精准的机器人控制策略
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61138) · [原文](https://icml.cc/virtual/2026/poster/61138)

**WestWorld: A Knowledge-Encoded Scalable Trajectory World Model for Diverse Robotic Systems**

- 一句话：WestWorld通过知识编码和系统感知混合专家模型，为多样化机器人系统打造可扩展的轨迹世界模型，实现零样本泛化与精准控制。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61703) · [原文](https://icml.cc/virtual/2026/poster/61703)

**DreamDojo: A Real-Time Robot World Model from Large-Scale Human Videos**

- 一句话：从4.4万小时人类视频中学习的世界模型DreamDojo，实现实时物理模拟与灵巧控制
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65193) · [原文](https://icml.cc/virtual/2026/poster/65193)

**Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning**

- 一句话：CAPS框架通过功率分布和信噪比控制，有效解决了长期任务中的指令漂移问题。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-64055) · [原文](https://icml.cc/virtual/2026/poster/64055)

**DriveWorld-VLA: Unified Latent-Space World Modeling with Vision–Language–Action for Autonomous Driving**

- 一句话：DriveWorld-VLA通过将视觉-语言-动作与世界模型统一在潜在空间中，提升了自动驾驶的决策和前瞻性想象力。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61191) · [原文](https://icml.cc/virtual/2026/poster/61191)

**FocalPolicy: Frequency-Optimized Chunking and Locally Anchored Flow Matching for Coherent Visuomotor Policy**

- 一句话：用频域分块+局部锚定流匹配，让机械臂动作既丝滑又精准
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-66520) · [原文](https://icml.cc/virtual/2026/poster/66520)

**From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation**

- 一句话：MoLA通过潜在动作混合将视频想象的未来转化为可执行控制，提升机器人操作的稳定性和成功率。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-66091) · [原文](https://icml.cc/virtual/2026/poster/66091)

**From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models**

- 一句话：潜在动作监督为视觉-语言-动作模型提供了一种统一且高效的训练框架。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-63621) · [原文](https://icml.cc/virtual/2026/poster/63621)

**LARA: Latent Action Representation Alignment for Vision-Language-Action Models**

- 一句话：LARA通过联合优化潜在动作模型和视觉语言动作模型，显著提升了机器人动作预测的准确性和泛化能力。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61232) · [原文](https://icml.cc/virtual/2026/poster/61232)

**LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries**

- 一句话：LangForce通过贝叶斯分解对抗'信息坍缩'，让机器人真正听懂人话
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65457) · [原文](https://icml.cc/virtual/2026/poster/65457)

**MVISTA-4D: View-Consistent 4D World Model with Test-Time Action Inference for Robotic Manipulation**

- 一句话：首个支持几何一致多视角RGBD生成的4D世界模型，通过测试时动作优化实现机器人精准操控
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65571) · [原文](https://icml.cc/virtual/2026/poster/65571)

**OXE-AugE: A Large-Scale Robot Augmentation of OXE for Scaling Cross-Embodiment Policy Learning**

- 一句话：通过9种新机器人构型增强OXE数据集,创造三倍规模的新基准OXE-AugE,显著提升跨构型策略泛化能力
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-64619) · [原文](https://icml.cc/virtual/2026/poster/64619)

**Predicting What Matters: Robust Generalist Robot Policy Learning via Future Semantic Mask**

- 一句话：用语义掩码预测替代RGB视频预测，打造抗干扰的机器人策略学习模型
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65548) · [原文](https://icml.cc/virtual/2026/poster/65548)

**Reflex: Real-Time Vision-Language-Action Control through Streaming Inference**

- 一句话：Reflex通过时间步不变性特性和增量缓存更新，首次实现视觉-语言-动作模型的实时流式推理。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-63338) · [原文](https://icml.cc/virtual/2026/poster/63338)

**RoboFlow4D: A Lightweight Flow World Model Toward Real-Time Flow-Guided Robotic Manipulation**

- 一句话：轻量级流世界模型RoboFlow4D用3D运动预测统一感知与规划，实现实时流引导机器人操控
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-62543) · [原文](https://icml.cc/virtual/2026/poster/62543)

**Scaling Real-World Robot Policy Evaluation via Discrete Diffusion World Model**

- 一句话：提出dWorldEval，一种以动作为中心的离散扩散世界模型，显著提升机器人策略评估的可靠性和可扩展性。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-65898) · [原文](https://icml.cc/virtual/2026/poster/65898)

**Structured 4D Latent World Model for Robot Planning**

- 一句话：用结构化4D潜在空间建模物理世界，让机器人像搭积木一样规划未来动作
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-63054) · [原文](https://icml.cc/virtual/2026/poster/63054)

**Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Models**

- 一句话：VLA-MBPO：用多模态世界模型解决机器人强化学习的三大挑战，实现高效安全的策略微调。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-60629) · [原文](https://icml.cc/virtual/2026/poster/60629)

**VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model**

- 一句话：用世界模型和VLA策略的迭代互哺，实现机器人任务成功率39.2%的飞跃
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-66169) · [原文](https://icml.cc/virtual/2026/poster/66169)

### P2

**Episodic Memory-Guided Controllable Experience Synthesis for Reinforcement Learning**

- 一句话：首个用情景记忆引导扩散模型生成高质量强化学习训练数据的方法
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61811) · [原文](https://icml.cc/virtual/2026/poster/61811)

**Learning to Share: Selective Memory for Efficient Parallel Agentic Systems**

- 一句话：LTS通过选择性共享内存机制，显著提升并行智能体系统的效率。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-62890) · [原文](https://icml.cc/virtual/2026/poster/62890)

**Little By Little: Continual Learning via Incremental Mixture of Rank-1 Associative Memory Experts**

- 一句话：MoRAM通过细粒度的rank-1记忆专家混合，实现了持续学习中的高效任务适应和遗忘缓解。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-64295) · [原文](https://icml.cc/virtual/2026/poster/64295)

**NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating**

- 一句话：NAACA通过神经启发的振荡工作记忆实现免训练的听觉显著性过滤，显著提升长音频事件检测效率。
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-60523) · [原文](https://icml.cc/virtual/2026/poster/60523)

**PRIM：Cooperative Dynamic Token Compression for Efficient Large Multimodal Models**

- 一句话：PRIM通过协作压缩策略，显著提升多模态模型的推理效率，同时保持高准确性。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-60866) · [原文](https://icml.cc/virtual/2026/poster/60866)

**Progressive Cramming: Reliable Token Compression and What It Reveals**

- 一句话：渐进式压缩揭示：完美重构可能是注意力劫持的假象，而非真正的语义压缩。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-66285) · [原文](https://icml.cc/virtual/2026/poster/66285)

**ProtoKV: Streaming Video Understanding under Delayed Evidence with Summary-State Memory**

- 一句话：ProtoKV通过原型记忆库解决流式视频理解中的延迟证据问题,准确率提升最高达12.5%
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-63663) · [原文](https://icml.cc/virtual/2026/poster/63663)

**Context Forcing: Consistent Autoregressive Video Generation with Long Context**

- 一句话：提出上下文强制框架，通过师生模型同步长上下文训练，实现超20秒连贯视频生成
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61892) · [原文](https://icml.cc/virtual/2026/poster/61892)

**When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning**

- 一句话：用带双门的智能记忆系统破解长文本推理难题：该更新时更新，该停止时绝不废话
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2026-61843) · [原文](https://icml.cc/virtual/2026/poster/61843)

## CVPR 2026

本轮收录 35 篇。

### P0

**Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulation**

- 一句话：通过3D空间可供性场干预，帮助视觉-语言-动作模型逃离‘记忆陷阱’，提升机器人操作的适应性。
- 对 VPWEM：关键反例：用于检验策略是否只是复现动作或轨迹。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-40221) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Affordance_Field_Intervention_Enabling_VLAs_to_Escape_Memory_Traps_in_CVPR_2026_paper.html)

**CycleManip: Enabling Cycle-based Manipulation via Effective History Perception and Understanding**

- 一句话：CycleManip框架通过历史感知与理解，让机器人高效完成周期性操作任务。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36298) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_CycleManip_Enabling_Cycle-based_Manipulation_via_Effective_History_Perception_and_Understanding_CVPR_2026_paper.html)

**Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation**

- 一句话：OptimusVLA通过双记忆框架提升机器人操作的效率和鲁棒性，显著优于现有方法。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39665) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Global_Prior_Meets_Local_Consistency_Dual-Memory_Augmented_Vision-Language-Action_Model_for_CVPR_2026_paper.html)

**SemanticVLA: Towards Semantic Reasoning over Action Memorization via Synergistic Explicit Trace and Latent Action Planning**

- 一句话：首创双路径协同架构，让机器人既会说坐标语言又会做视觉动作
- 对 VPWEM：关键反例：用于检验策略是否只是复现动作或轨迹。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39352) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Ni_SemanticVLA_Towards_Semantic_Reasoning_over_Action_Memorization_via_Synergistic_Explicit_CVPR_2026_paper.html)

**TRM-VLA: Temporal-Aware Chain-of-Thought Reasoning and Memorization for Vision-Language-Action Models**

- 一句话：TRM-VLA通过时间感知的推理和记忆框架，显著提升了VLA模型在机器人操作中的推理效率和性能。
- 对 VPWEM：关键反例：用于检验策略是否只是复现动作或轨迹。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36166) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TRM-VLA_Temporal-Aware_Chain-of-Thought_Reasoning_and_Memorization_for_Vision-Language-Action_Models_CVPR_2026_paper.html)

### P1

**PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation**

- 一句话：PALM框架通过可用性推理和进度感知机制，显著提升机器人长周期操作任务的执行成功率
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38810) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_PALM_Progress-Aware_Policy_Learning_via_Affordance_Reasoning_for_Long-Horizon_Robotic_CVPR_2026_paper.html)

**AGiLe: Learning Robust Long-Horizon Manipulation via Affordance-Grounded Bidirectional Latent Planning**

- 一句话：AGiLe通过双向潜在规划和功能可供性落地，实现长时程操作的时空鲁棒性
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39123) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_AGiLe_Learning_Robust_Long-Horizon_Manipulation_via_Affordance-Grounded_Bidirectional_Latent_Planning_CVPR_2026_paper.html)

**Adaptive Action Chunking at Inference-time for Vision-Language-Action Models**

- 一句话：提出自适应动作分块策略，用动作熵动态调整分块尺寸，让机器人操作既灵活又稳定。
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39692) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.html)

**ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving**

- 一句话：ColaVLA通过统一的视觉-语言-动作框架，实现了高效、准确且安全的自动驾驶轨迹生成。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36544) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Peng_ColaVLA_Leveraging_Cognitive_Latent_Reasoning_for_Hierarchical_Parallel_Trajectory_Planning_CVPR_2026_paper.html)

**Dexterous World Models**

- 一句话：灵巧世界模型（DWM）通过视频扩散技术，首次实现了静态3D场景中的逼真具身交互。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37216) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Dexterous_World_Models_CVPR_2026_paper.html)

**Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics for Robot Manipulation**

- 一句话：E3Flow首次实现高效修正流与多模态等变学习的融合，推理速度提升7倍的同时成功率提高3.12%
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37268) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Efficient_Hybrid_SE3-Equivariant_Visuomotor_Flow_Policy_via_Spherical_Harmonics_for_CVPR_2026_paper.html)

**From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings**

- 一句话：首个全自动化端到端系统，从工业视频中提取结构化VLA预训练数据。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37209) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_From_Observation_to_Action_Latent_Action-based_Primitive_Segmentation_for_VLA_CVPR_2026_paper.html)

**Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild**

- 一句话：JALA框架通过联合对齐潜在动作，利用混合人类视频数据实现了机器人动作预训练的可扩展突破。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38737) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Joint-Aligned_Latent_Action_Towards_Scalable_VLA_Pretraining_in_the_Wild_CVPR_2026_paper.html)

**PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation**

- 一句话：PointWorld通过3D点流统一建模动作与状态，实现跨形态机器人对复杂物理交互的零样本预测
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38054) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_PointWorld_Scaling_3D_World_Models_for_In-The-Wild_Robotic_Manipulation_CVPR_2026_paper.html)

**RehearseVLA: Simulated Post-Training for VLAs with Physically-Consistent World Model**

- 一句话：用虚拟世界模拟器给VLA模型开‘安全训练小灶’，仅需5次演示就能突破数据稀缺与高危环境限制
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37811) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_RehearseVLA_Simulated_Post-Training_for_VLAs_with_Physically-Consistent_World_Model_CVPR_2026_paper.html)

**Scalable Trajectory Generation for Whole-Body Mobile Manipulation**

- 一句话：AutoMoMa通过GPU加速规划，每小时生成5000个任务片段，解决了移动操作数据稀缺问题。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-40034) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Niu_Scalable_Trajectory_Generation_for_Whole-Body_Mobile_Manipulation_CVPR_2026_paper.html)

**Scaling Spatial and Temporal Context for Robotic Imitation Learning Policies With Scene Graphs**

- 一句话：通过动态场景图，机器人模仿学习在部分观测环境中实现了长期推理和鲁棒泛化。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38076)

### P2

**An Efficient Token Compression Framework for Visual Object Tracking**

- 一句话：ETCTrack通过自适应令牌压缩和分层交互编码器，显著提升了视觉跟踪的性能和效率。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36338) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_An_Efficient_Token_Compression_Framework_for_Visual_Object_Tracking_CVPR_2026_paper.html)

**ApET: Approximation-Error Guided Token Compression for Efficient VLMs**

- 一句话：ApET框架通过近似误差引导的视觉标记压缩，在保持95%以上模型性能的同时减少近90%计算开销
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39704) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_ApET_Approximation-Error_Guided_Token_Compression_for_Efficient_VLMs_CVPR_2026_paper.html)

**BiGain: Unified Token Compression for Joint Generation and Classification**

- 一句话：BiGain通过频率感知令牌压缩，在加速扩散模型中同时提升生成质量和分类准确性。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37406) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_BiGain_Unified_Token_Compression_for_Joint_Generation_and_Classification_CVPR_2026_paper.html)

**Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing**

- 一句话：用'灰度持续+智能触发彩色'方案，仅8.1%的RGB帧就保留91.6%性能，破解持续视频感知的能耗困局
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37693) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Cai_Color_When_It_Counts_Grayscale-Guided_Online_Triggering_for_Always-On_Streaming_CVPR_2026_paper.html)

**EarlyTom: Early Token Compression Completes Fast Video Understanding**

- 一句话：EarlyTom通过在视觉编码器内部进行早期token压缩，显著提升了视频大语言模型的部署效率。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37950) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_EarlyTom_Early_Token_Compression_Completes_Fast_Video_Understanding_CVPR_2026_paper.html)

**FlashVSR: Towards Real-time Diffusion-Based Streaming Video Super Resolution**

- 一句话：FlashVSR：首个基于扩散模型的一步流式视频超分辨率框架，实现高效、可扩展且接近实时的性能。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38690) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhuang_FlashVSR_Towards_Real-time_Diffusion-Based_Streaming_Video_Super_Resolution_CVPR_2026_paper.html)

**FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding**

- 一句话：FluxMem通过自适应分层记忆压缩技术，在流式视频理解中实现实时高效处理，同时减少65%冗余标记。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38716) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_FluxMem_Adaptive_Hierarchical_Memory_for_Streaming_Video_Understanding_CVPR_2026_paper.html)

**Interactive Episodic Memory with User Feedback**

- 一句话：首个支持用户反馈的交互式情景记忆框架，通过反馈对齐模块显著提升搜索准确性
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37436) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Subedi_Interactive_Episodic_Memory_with_User_Feedback_CVPR_2026_paper.html)

**Joint Learning of General and Diverse Patterns with Mixture of Memory Experts for Weakly-Supervised Video Anomaly Detection**

- 一句话：通过混合记忆专家框架，联合学习通用和多样化模式，实现了弱监督视频异常检测的突破。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36266) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Joint_Learning_of_General_and_Diverse_Patterns_with_Mixture_of_CVPR_2026_paper.html)

**OASIS: On-Demand Hierarchical Event Memory for Streaming Video Reasoning**

- 一句话：OASIS框架通过分层事件记忆和意图驱动的按需检索，在流式视频推理中实现高效精准的长时程分析。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-36584) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_OASIS_On-Demand_Hierarchical_Event_Memory_for_Streaming_Video_Reasoning_CVPR_2026_paper.html)

**Revisiting Token Compression for Accelerating ViT-based Sparse Multi-View 3D Object Detectors**

- 一句话：SEPatch3D通过动态调整patch尺寸和特征增强，显著加速了ViT-based稀疏多视角3D目标检测器，同时保持高精度。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39920) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_Revisiting_Token_Compression_for_Accelerating_ViT-based_Sparse_Multi-View_3D_Object_CVPR_2026_paper.html)

**Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation**

- 一句话：通过EMA-Sink和奖励驱动的分布匹配蒸馏，实现高效动态的流式视频生成。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37619) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Reward_Forcing_Efficient_Streaming_Video_Generation_with_Rewarded_Distribution_Matching_CVPR_2026_paper.html)

**Stabilizing Streaming Video Geometry via Dynamic Feature Normalization**

- 一句话：动态特征归一化（DyFN）让连续视频中的3D几何估计稳如泰山，仅增加2%的参数就解决了时间不一致性问题。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38162) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Lyu_Stabilizing_Streaming_Video_Geometry_via_Dynamic_Feature_Normalization_CVPR_2026_paper.html)

**StreamReady: Learning What to Answer and When in Long Streaming Videos**

- 一句话：StreamReady通过引入回答准备度评分（ARS），实现了在流媒体视频中准确且及时地回答问题的能力。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-37104) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Azad_StreamReady_Learning_What_to_Answer_and_When_in_Long_Streaming_CVPR_2026_paper.html)

**Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning**

- 一句话：通过时空因果推理，我们显著提升了实时犯罪预测的准确性。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39800) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html)

**Streaming Video Instruction Tuning**

- 一句话：Streamo：首个能实时处理多种流式视频任务的通用交互助手，填补视频感知与实时交互的空白
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38386) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Xia_Streaming_Video_Instruction_Tuning_CVPR_2026_paper.html)

**StreamingTOM: Streaming Token Compression for Efficient Video Understanding**

- 一句话：首个通过因果时序缩减+在线量化内存实现流式视频理解效率革命的无训练框架
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-39087) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_StreamingTOM_Streaming_Token_Compression_for_Efficient_Video_Understanding_CVPR_2026_paper.html)

**UniCompress: Token Compression for Unified Vision–Language Understanding and Generation**

- 一句话：UniCompress通过可学习的元标记压缩视觉标记，在统一视觉-语言模型中实现4倍压缩率且性能几乎无损。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2026-38630) · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_UniCompress_Token_Compression_for_Unified_Vision-Language_Understanding_and_Generation_CVPR_2026_paper.html)

## AAAI 2026

本轮收录 21 篇。

### P0

**RGMP: Recurrent Geometric-prior Multimodal Policy for Generalizable Humanoid Robot Manipulation**

- 一句话：RGMP框架通过几何先验和高效数据利用，显著提升了人形机器人的泛化能力和任务成功率。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38539) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38539)

**Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective**

- 一句话：提出首个面向物体级非马尔可夫推理的槽位化VLA框架，解决机器人操作中历史感知的时序扩展难题。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-37337) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37337)

### P1

**Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward**

- 一句话：AC3框架通过稳定化行动者-评论家学习，用连续动作块攻克稀疏奖励的长时程机器人操作难题
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38937) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38937)

**EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI**

- 一句话：通过等变流匹配与加速正则化，实现数据高效、推理极快的具身AI策略学习
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-39028) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/39028)

**Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic Reinforcement Learning Benchmarks**

- 一句话：研究发现，机器人强化学习中的动态模型并非全局稀疏，而是局部、状态依赖的稀疏性，挑战了传统的稀疏性假设。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-39658) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/39658)

**ForeDiffusion: Foresight-Conditioned Diffusion Policy via Future View Construction for Robot Manipulation**

- 一句话：给扩散策略装上'未来眼'，机器人操作成功率飙升20%
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38934) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38934)

**Gentle Manipulation Policy Learning via Demonstrations from VLM Planned Atomic Skills**

- 一句话：用VLM规划原子技能+仿真RL训练，实现零人类示范的长时程精细操作策略学习
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38955) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38955)

**LatentVLA: Taming Latent Space for Generalizable and Long-Horizon Bimanual Manipulation**

- 一句话：LatentVLA通过连续且时空接地的潜在动作表示，解决了机器人模仿学习中的运动保真度与数据可扩展性的权衡问题。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38926) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38926)

**Learning Diffusion Policy from Primitive Skills for Robot Manipulation**

- 一句话：用乐高式原始技能组装扩散策略，让机器人动作既精准又好懂
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38889) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38889)

**MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation**

- 一句话：MP1通过MeanFlow范式在单次网络评估中生成精确的机器人动作轨迹，显著提升了任务成功率和推理速度。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38919) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38919)

**ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation**

- 一句话：首创交互感知基元序列框架，让机器人通过单次模仿学习掌握复杂长时程操作
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38881) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38881)

**ManipDreamer3D: Synthesizing Plausible Robotic Manipulation Video with Occupancy-aware 3D Trajectory**

- 一句话：ManipDreamer3D：通过3D占据感知轨迹合成逼真机器人操作视频的框架
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-37595) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37595)

**ODYSSEY: Open-World Quadrupeds Exploration and Manipulation for Long-Horizon Tasks**

- 一句话：ODYSSEY让四足机器人像奥德修斯般完成开放世界的长时程移动操作冒险
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38927) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38927)

**Steering Visuomotor Policy in Open Worlds via Cross-View Goal Alignment**

- 一句话：ROCKET-2：通过跨视角目标对齐，让智能体在3D环境中‘看懂’人类意图的视觉导航新框架。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38876) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38876)

**TCoT: Trajectory Chain-of-Thoughts for Robotic Manipulation with Failure Recovery in Vision-Language-Action Model**

- 一句话：TCoT通过轨迹思维链和失败恢复机制，提高VLA在长时程机器人操作中的纠错能力。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-37577) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37577)

### P2

**Backtrace Mamba: Reviving Critical Temporal Contexts via Hierarchical Memory Compression for Online Action Detection**

- 一句话：提出基于Mamba的分层记忆压缩框架，实现高效在线动作检测
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38139) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38139)

**CogStream: Context-guided Streaming Video Question Answering**

- 一句话：提出CogStream任务和CogReasoner模型，通过上下文筛选解决流式视频问答的计算负担问题
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38336) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38336)

**CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking**

- 一句话：CompTrack通过信息瓶颈动态压缩点云冗余，实现90 FPS的高效3D跟踪
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-38385) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/38385)

**Contribution-aware Token Compression for Efficient Video Understanding via Reinforcement Learning**

- 一句话：用强化学习教AI主动筛选『有用』视频片段，告别无效计算！
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-39612) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/39612)

**Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation**

- 一句话：首个通过深度记忆协同实现全天候图像反射分离的框架，夜间性能提升显著
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-37386) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37386)

**Vista: Scene-Aware Optimization for Streaming Video Question Answering Under Post-Hoc Queries**

- 一句话：Vista通过场景感知优化，实现了流式视频问答的高效推理，解决了长视频实时问答的难题。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2026-article-37694) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37694)

## WACV 2026

本轮收录 2 篇。

### P0

**Memory-Augmented Representation for Efficient Event-based Visuomotor Policy Learning with Adaptive Perception and Control**

- 一句话：提出了一种记忆增强的事件相机策略学习框架，显著提高了无人机导航的效率和鲁棒性。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/wacv2026-Kamal_Memory-Augmented_Representation_for_Efficient_Event-ba) · [原文](https://openaccess.thecvf.com/content/WACV2026/html/Kamal_Memory-Augmented_Representation_for_Efficient_Event-based_Visuomotor_Policy_Learning_with_Adaptive_WACV_2026_paper.html)

### P2

**Online Episodic Memory Visual Query Localization with Egocentric Streaming Object Memory**

- 一句话：ESOM框架通过在线处理视频流和紧凑内存，显著提升了可穿戴相机的情景记忆检索能力。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/wacv2026-Manigrasso_Online_Episodic_Memory_Visual_Query_Localization_) · [原文](https://openaccess.thecvf.com/content/WACV2026/html/Manigrasso_Online_Episodic_Memory_Visual_Query_Localization_with_Egocentric_Streaming_Object_WACV_2026_paper.html)

## AAMAS 2026

本轮收录 0 篇。

本轮未发现明确相关论文。

## ACL 2026

本轮收录 3 篇。

### P0

**Conflict-Aware Memory for Embodied Agents: Enhancing Vector Data Quality via Detection Rules**

- 一句话：该方法利用冲突检测规则改善具身智能体向量记忆的数据质量与任务规划可靠性。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2026-7cc20c97f79e)

### P2

**Beyond Markovian Forgetfulness: Episodic Memory for Reasoning-Intensive Retrieval**

- 一句话：该工作为推理密集型检索引入情景记忆，以缓解仅依赖当前查询造成的信息遗忘。
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2026-c3258eb5eaab)

**GMSA: Enhancing Context Compression via Group Merging and Layer Semantic Alignment**

- 一句话：GMSA通过分组合并与层级语义对齐压缩长上下文，属于可迁移的上下文压缩方法。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2026-192bf264ea8c)

## EACL 2026

本轮收录 2 篇。

### P2

**H-Mem: Hybrid Multi-Dimensional Memory Management for Long-Context Conversational Agents.**

- 一句话：H-Mem以混合多维记忆管理机制组织长上下文对话智能体的不同类型历史信息。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/dblp:conf/eacl/YeHCZ26) · [原文](https://doi.org/10.18653/v1/2026.eacl-long.363)

**Tailoring Memory Granularity for Multi-Hop Reasoning over Long Contexts.**

- 一句话：该工作研究多跳长上下文推理中的记忆粒度选择，以及压缩粒度对证据保留的影响。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/dblp:conf/eacl/QingDMHV26) · [原文](https://doi.org/10.18653/v1/2026.findings-eacl.189)

## WSDM 2026

本轮收录 0 篇。

本轮未发现明确相关论文。
