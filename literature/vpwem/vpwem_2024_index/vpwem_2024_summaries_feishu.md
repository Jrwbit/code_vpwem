# 2024 顶会与 VPWEM 相关论文：快速阅读版

更新时间：2026-07-22

共 20 篇。站内未提供中文导读，本版采用论文英文摘要；本版用于快速定位，关键结论仍需回到原文核验。

阅读顺序：先看 P0，再看 P1；P2 仅在设计压缩或流式记忆实验时查阅。

## ICLR 2024

本轮收录 5 篇。

### P0

**Embodied Active Defense: Leveraging Recurrent Feedback to Counter Adversarial Patches**

- 一句话：The vulnerability of deep neural networks to adversarial patches has motivated numerous defense strategies for boosting model robustness.
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2024-uXjfOmTiDt-b7f2948f56) · [原文](https://iclr.cc/virtual/2024/poster/17570)

### P1

**Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks**

- 一句话：Large Language Models (LLMs) are highly capable of performing planning for long-horizon robotics tasks, yet existing methods require access to a pre-defined skill library (*e.g.* picking, placing, pulling, pushing, navigating).
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2024-hQVCCxQrYN-43282ad489) · [原文](https://iclr.cc/virtual/2024/poster/18096)

**RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches**

- 一句话：Generalization remains one of the most important desiderata for robust robot learning systems.
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2024-F1TKzG8LJO-cbae61a4a6) · [原文](https://iclr.cc/virtual/2024/poster/19084)

### P2

**Efficient Episodic Memory Utilization of Cooperative Multi-Agent Reinforcement Learning**

- 一句话：In cooperative multi-agent reinforcement learning (MARL), agents aim to achieve a common goal, such as defeating enemies or scoring a goal.
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2024-LjivA1SLZ6-68d3b3eefc) · [原文](https://iclr.cc/virtual/2024/poster/18851)

**RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation**

- 一句话：Retrieval-augmented language models improve language models (LMs) by retrieving documents and prepending them in-context.
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2024-mlJLVigNHp-c9889d4865) · [原文](https://iclr.cc/virtual/2024/poster/17885)

## ICML 2024

本轮收录 8 篇。

### P1

**3D-VLA: A 3D Vision-Language-Action Generative World Model**

- 一句话：Recent vision-language-action (VLA) models rely on 2D inputs, lacking integration with the broader realm of the 3D physical world.
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2024-EZcFK8HupF-27608de818) · [原文](https://icml.cc/virtual/2024/poster/34575)

**KISA: A Unified Keyframe Identifier and Skill Annotator for Long-Horizon Robotics Demonstrations**

- 一句话：Robotic manipulation tasks often span over long horizons and encapsulate multiple subtasks with different skills.
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2024-oCI9gHocws-420840fb44) · [原文](https://icml.cc/virtual/2024/poster/33091)

**RoboDreamer: Learning Compositional World Models for Robot Imagination**

- 一句话：Text-to-video models have demonstrated substantial potential in robotic decision-making, enabling the imagination of realistic plans of future actions as well as accurate environment simulation.
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2024-kHjOmAUfVe-e243e92a9f) · [原文](https://icml.cc/virtual/2024/poster/33266)

### P2

**Hidden Traveling Waves bind Working Memory Variables in Recurrent Neural Networks**

- 一句话：Traveling waves are a fundamental phenomenon in the brain, playing a crucial role in short-term information storage.
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2024-NCjlFw1Ab0-0e30843af4) · [原文](https://icml.cc/virtual/2024/poster/34210)

**LoCoCo: Dropping In Convolutions for Long Context Compression**

- 一句话：This paper tackles the memory hurdle of of processing long context sequences in Large Language Models (LLMs), by presenting a novel approach, Dropping In Convolutions for **Lo**ng **Co**ntext **Co**mpression (**LoCoCo**).
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2024-NUlyqMyhO9-25d4b66075) · [原文](https://icml.cc/virtual/2024/poster/34202)

**Think Before You Act: Decision Transformers with Working Memory**

- 一句话：Decision Transformer-based decision-making agents have shown the ability to generalize across multiple tasks.
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2024-PSQ5Z920M8-2c07c0aac4) · [原文](https://icml.cc/virtual/2024/poster/34121)

**A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts**

- 一句话：Current Large Language Models (LLMs) are not only limited to some maximum context length, but also are not able to robustly consume long inputs.
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2024-OTmcsyEO5G-cbc99f8760) · [原文](https://icml.cc/virtual/2024/poster/34161)

**Memory Consolidation Enables Long-Context Video Understanding**

- 一句话：Most transformer-based video encoders are limited to short temporal contexts due to their quadratic complexity.
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2024-qeFgvVVAJ2-f3fd6bd443) · [原文](https://icml.cc/virtual/2024/poster/32983)

## CVPR 2024

本轮收录 2 篇。

### P1

**Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation**

- 一句话：This paper introduces Hierarchical Diffusion Policy (HDP) a hierarchical agent for multi-task robotic manipulation.
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2024-29288-c651af94b6) · [原文](https://cvpr.thecvf.com/virtual/2024/poster/29288)

### P2

**MemoNav: Working Memory Model for Visual Navigation**

- 一句话：Image-goal navigation is a challenging task that requires an agent to navigate to a goal indicated by an image in unfamiliar environments.
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2024-29828-dbdfa6ef7a) · [原文](https://cvpr.thecvf.com/virtual/2024/poster/29828)

## AAAI 2024

本轮收录 2 篇。

### P0

**READ-PVLA: Recurrent Adapter with Partial Video-Language Alignment for Parameter-Efficient Transfer Learning in Low-Resource Video-Language Modeling**

- 一句话：Fully fine-tuning pretrained large-scale transformer models has become a popular paradigm for video-language modeling tasks, such as temporal language grounding and video-language summarization.
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2024-article-29847-1d4438db2f) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/29847)

### P2

**Working Memory Capacity of ChatGPT: An Empirical Study**

- 一句话：Working memory is a critical aspect of both human intelligence and artificial intelligence, serving as a workspace for the temporary storage and manipulation of information.
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2024-article-28868-5ecb7f10d6) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/28868)

## WACV 2024

本轮收录 1 篇。

### P1

**Optimizing Long-Term Robot Tracking With Multi-Platform Sensor Fusion**

- 一句话：Monitoring a fleet of robots requires stable long-term tracking with re-identification, which is yet an unsolved challenge in many scenarios.
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/wacv2024-519418ebd7-632a816f42) · [原文](https://openaccess.thecvf.com/content/WACV2024/html/Albanese_Optimizing_Long-Term_Robot_Tracking_With_Multi-Platform_Sensor_Fusion_WACV_2024_paper.html)

## AAMAS 2024

本轮收录 1 篇。

### P0

**Offline Risk-sensitive RL with Partial Observability to Enhance Performance in Human-Robot Teaming.**

- 一句话：暂无站内摘要，后续按需精读原文。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/dblp%3Aconf%2Fatal%2FAngelottiCPLCD24) · [原文](https://dblp.org/rec/conf/atal/AngelottiCPLCD24)

## ACL 2024

本轮收录 0 篇。

本轮未发现明确相关论文。

## EACL 2024

本轮收录 1 篇。

### P0

**Multi-party Multimodal Conversations Between Patients, Their Companions, and a Social Robot in a Hospital Memory Clinic.**

- 一句话：暂无站内摘要，后续按需精读原文。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/dblp%3Aconf%2Feacl%2FAddleseeCNGGSDL24) · [原文](https://doi.org/10.18653/v1/2024.eacl-demo.8)

## WSDM 2024

本轮收录 0 篇。

本轮未发现明确相关论文。
