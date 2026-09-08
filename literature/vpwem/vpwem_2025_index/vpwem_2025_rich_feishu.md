# 2025 顶会与 VPWEM 相关论文：图文导读版

更新时间：2026-07-22

共 44 篇。图片与摘要素材来自 Connected Researchers；没有站内素材的论文仅保留文字和原文链接。

说明：本版用于快速阅读与筛选，方法和结果数字在正式引用前仍需回到官方论文核验。

## ICLR 2025

本轮收录 16 篇。

### P1

#### STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning

![STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-4VHiptx7xe-1db98bf74e.webp)

- 导读：机器人学习正在见证预收集数据集在规模、多样性和复杂性上的显著增长，这反映了自然语言处理和计算机视觉等领域的趋势。许多机器人学习方法将这些数据集视为多任务专家数据，并通过广泛训练来学习多任务通用策略。值得注意的是，虽然这些通用策略可以提高多个任务的平均性能，但由于数据分区之间的负迁移，通用策略在单个任务上的性能往往不如任务特定的专用策略。在这项工作中，我们主张在部署时根据遇到的情景训练策略：不是以零样本方式将预训练策略部署到未见问题，而是在测试时非参数地检索相关数据并直接训练模型。此外，我们表明许多机器人任务共享大量低级行为，并且“子轨迹”粒度的检索能够显著提高数据利用、泛化性和对新颖问题的策略适应性。相比之下，现有的全轨迹检索方法往往数据利用不足，并错失了跨任务的共享内容。这项工作提出了STRAP，一种利用预训练视觉基础模型和动态时间规整以鲁棒方式从大型训练语料库中检索轨迹子序列的技术。STRAP在模拟和真实实验中均优于先前的检索算法和多任务学习方法，展示了扩展到真实世界中更大离线数据集的能力，以及仅用少量真实世界演示学习鲁棒控制策略的能力。
- 方法：利用预训练视觉基础模型和动态时间规整从大型训练数据中鲁棒检索子轨迹序列；在测试时非参数地检索相关子轨迹并直接训练策略
- 结果：在模拟和真实实验中优于先前检索算法和多任务学习方法；展示了扩展到更大离线数据集的能力
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-4VHiptx7xe-1db98bf74e) · [原文](https://iclr.cc/virtual/2025/poster/31008)

#### AdaManip: Adaptive Articulated Object Manipulation Environments and Policy Learning

![AdaManip: Adaptive Articulated Object Manipulation Environments and Policy Learning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-Luss2sa0vc-7bd6e95ce4.webp)

- 导读：关节物体操作是机器人在现实场景中执行各种任务的关键能力。关节物体由多个通过关节连接的部件组成，通过复杂的相对运动赋予多样的功能机制。例如，保险柜由门、把手和锁组成，只有当锁栓解锁时门才能打开。内部结构，如锁的状态或关节角度约束，无法直接从视觉观察中获取。因此，成功操作这些物体需要基于试错的适应性调整，而非一次性的视觉推断。然而，先前的关节物体数据集和仿真环境主要关注简单的操作机制，即整个操作过程可以从物体外观推断。为了增强适应性操作机制的多样性和复杂性，我们构建了一个新颖的关节物体操作环境，并配备了9个类别的物体。基于该环境和物体，我们进一步提出了一种适应性演示收集和基于3D视觉扩散的模仿学习流水线，以学习适应性操作策略。通过仿真和真实世界实验验证了我们的设计和所提方法的有效性。
- 方法：构建包含9类物体的关节物体操作环境，设计适应性机制。；提出适应性演示收集方法。
- 结果：在仿真和真实世界实验中验证了方法的有效性。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-Luss2sa0vc-7bd6e95ce4) · [原文](https://iclr.cc/virtual/2025/poster/29969)

#### COMBO: Compositional World Models for Embodied Multi-Agent Cooperation

![COMBO: Compositional World Models for Embodied Multi-Agent Cooperation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-YXRyYkb1im-7ccf0863ed.webp)

- 导读：本文研究了具身多智能体合作问题，其中分散的智能体必须仅依靠自我中心视角的观测进行合作。为了在这种设置下有效规划，与学习单智能体场景中的世界动态不同，我们需要在仅提供部分自我中心视觉观测的情况下，模拟任意数量智能体动作下的世界动态。为了解决部分可观测性问题，我们首先训练生成模型来根据部分自我中心观测估计整体世界状态。为了在此世界状态下精确模拟多组动作，我们提出学习多智能体合作的组合世界模型，通过分解自然可组合的多个智能体的联合动作，并根据世界状态组合生成视频。利用该组合世界模型，结合视觉语言模型推断其他智能体的动作，我们可以使用树搜索过程整合这些模块以促进在线合作规划。我们在三个包含2-4个智能体的挑战性基准上评估了该方法。结果表明，我们的组合世界模型有效，该框架使具身智能体能够在不同任务和任意数量的智能体下高效协作，展示了所提方法的未来潜力。更多视频见https://umass-embodied-agi.github.io/COMBO/。
- 方法：训练生成模型从部分自我中心观测估计整体世界状态。；学习组合世界模型，分解联合动作并组合生成视频，模拟多智能体动作。
- 结果：在三个包含2-4个智能体的挑战性基准上验证了组合世界模型的有效性。；框架使智能体在不同任务和任意数量智能体下高效协作。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-YXRyYkb1im-7ccf0863ed) · [原文](https://iclr.cc/virtual/2025/poster/29260)

#### Dream to Manipulate: Compositional World Models Empowering Robot Imitation Learning with Imagination

- 导读：世界模型为智能体提供其环境的表征，使其能够预测自身行为的因果结果。当前的世界模型通常无法直接明确地模仿机器人面前的真实环境，常导致不真实的行为和幻觉，使其不适合实际机器人应用。为克服这些挑战，我们提出重新思考机器人世界模型作为可学习的数字孪生。我们介绍DreMa，一种使用现实世界及其动态的显式学习表征自动构建数字孪生的新方法，弥合传统数字孪生与世界模型之间的差距。DreMa通过整合高斯泼溅和物理模拟器来复制观察到的世界及其结构，利用其组合性使机器人能够想象物体的新配置并预测机器人行为的未来后果。我们利用这一能力，通过将等变变换应用于少量演示来生成用于模仿学习的新数据。我们在各种设置下的评估表明，通过增加动作和物体分布，准确性和鲁棒性有显著提升，减少了学习策略所需的数据，并提高了智能体的泛化能力。亮点是，我们展示了一个真实的Franka Emika Panda机器人，借助DreMa的想象力，能够从每个任务变体仅一个示例成功学习新的物理任务（一次性策略学习）。
- 方法：将世界模型重构为可学习的数字孪生；结合高斯泼溅和物理模拟器复制世界及其结构
- 结果：在多种设置下准确性和鲁棒性显著提升；减少了学习策略所需的数据
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-3RSLW9YSgk-68d9b74573) · [原文](https://iclr.cc/virtual/2025/poster/31075)

#### FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model

![FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-B2N0nCVC91-d20d5e6f36.webp)

- 导读：我们旨在开发一个基于模型的规划框架，使其世界模型能够随着模型和数据预算的增加而扩展，用于仅需语言和视觉输入的通用操作任务。为此，我们提出了以流为中心的生成式规划（FLIP），这是一种在视觉空间上的基于模型的规划算法，包含三个关键模块：1. 一个多模态流生成模型作为通用动作提议模块；2. 一个流条件视频生成模型作为动力学模块；3. 一个视觉-语言表示学习模型作为价值模块。给定初始图像和作为目标的语言指令，FLIP能够逐步搜索长期流和视频计划，最大化折现回报以完成任务。FLIP能够跨物体、机器人和任务合成长期计划，以图像流作为通用动作表示，密集的流信息也为长期视频生成提供了丰富的指导。此外，合成的流和视频计划可以指导低级控制策略的训练以执行机器人操作。在多个基准上的实验表明，FLIP能够提高长期视频计划合成的成功率和质量，并具有交互式世界模型的特性，为未来的工作开辟了更广泛的应用。视频演示见我们网站：https://nus-lins-lab.github.io/flipweb/。
- 方法：提出多模态流生成模型作为通用动作提议模块；使用流条件视频生成模型作为动力学模块
- 结果：在多个基准上提高了长期视频计划合成的成功率和质量；展现了交互式世界模型的特性
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-B2N0nCVC91-d20d5e6f36) · [原文](https://iclr.cc/virtual/2025/poster/30597)

#### Generalized Consistency Trajectory Models for Image Manipulation

![Generalized Consistency Trajectory Models for Image Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-Zjv38dg1Hb-b0f93fc152.webp)

- 导读：扩散模型（DM）在无条件生成以及图像编辑和修复等应用中表现出色。扩散的成功在于其迭代性质：扩散将噪声映射到数据的复杂过程分解为一系列简单的去噪任务。此外，我们可以通过向每个去噪步骤注入引导项来对生成过程进行细粒度控制。然而，迭代过程计算密集，通常需要数十到数千次函数评估。尽管一致性轨迹模型（CTM）能够沿概率流ODE（PFODE）在任意时间点之间遍历，并通过单次函数评估进行分数推断，但CTM只允许从高斯噪声到数据的转换。这项工作旨在通过提出广义一致性轨迹模型（GCTM）来释放CTM的全部潜力，该模型通过ODE在任意分布之间进行转换。我们讨论了GCTM的设计空间，并展示了其在图像到图像转换、修复和编辑等多种图像处理任务中的有效性。
- 方法：提出广义一致性轨迹模型（GCTM），扩展CTM以支持任意分布间的转换。；利用概率流ODE实现分布间的高效遍历。
- 结果：GCTM能够以单次函数评估实现任意分布之间的转换。；在图像到图像转换、修复和编辑等任务中展示了有效性。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-Zjv38dg1Hb-b0f93fc152) · [原文](https://iclr.cc/virtual/2025/poster/29202)

#### Learning View-invariant World Models for Visual Robotic Manipulation

![Learning View-invariant World Models for Visual Robotic Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-vJwjWyt4Ed-b80733cedf.webp)

- 导读：机器人操作任务通常依赖摄像头视觉输入来感知环境。然而，当操作过程中摄像头视角发生变化时，现有方法仍会出现性能下降。本文提出ReViWo（视角不变世界模型的表示学习），利用多视图数据学习在视角干扰下进行控制的鲁棒表示。ReViWo采用自编码器框架，通过结合视角不变表示（VIR）和视角相关表示的架构来重构目标图像。为了训练ReViWo，我们在模拟器中收集带有已知视图标签的多视图数据，同时也在无视图标签的Open X-Embodiment数据集上训练。然后，VIR用于在预收集的操作数据上训练世界模型，并通过与世界模型交互来训练策略。我们在Meta-world和PandaGym环境中评估了ReViWo在各种视角干扰场景下的有效性，包括新颖相机位置下的控制和频繁相机抖动。此外，我们还在真实世界的ALOHA机器人上进行了实验。结果表明，ReViWo在视角干扰下保持鲁棒性能，而基线方法遭受显著性能下降。此外，我们展示了VIR捕捉了任务相关的状态信息，并对新颖视角的观测保持稳定，验证了ReViWo方法的有效性。
- 方法：提出ReViWo框架，用自编码器结合视角不变表示和视角相关表示重构目标图像。；利用模拟器中带视图标签的多视图数据和无标签的Open X-Embodiment数据集联合训练。
- 结果：在Meta-world和PandaGym中，ReViWo在视角变化和相机抖动下保持鲁棒性能，基线方法显著下降。；在真实ALOHA机器人上验证了ReViWo的实际有效性。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-vJwjWyt4Ed-b80733cedf) · [原文](https://iclr.cc/virtual/2025/poster/27921)

#### TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies

![TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-b1CVu9l5GO-9e3065e500.webp)

- 导读：尽管在大量机器人数据集上预训练的大型视觉-语言-行动（VLA）模型为机器人学习提供了有前景的通用策略，但它们在交互式机器人中的时空动态方面仍存在困难，使其在处理复杂任务（如操作）时效果不佳。在这项工作中，我们引入了视觉轨迹提示，这是一种简单而有效的方法，通过视觉编码状态-行动轨迹来促进VLA模型的时空感知以预测行动。我们通过在自己的150K机器人操作轨迹数据集上微调OpenVLA并使用视觉轨迹提示，开发了一个新的TraceVLA模型。在SimplerEnv的137个配置和物理WidowX机器人的4个任务上的评估显示，TraceVLA达到了最先进的性能，在SimplerEnv上比OpenVLA高出10%，在真实机器人任务上高出3.5倍，并在不同实体和场景中展现出强大的泛化能力。为了进一步验证我们方法的有效性和通用性，我们基于4B Phi-3-Vision提出了一个紧凑的VLA模型，该模型在Open-X-Embodiment上预训练并在我们的数据集上微调，与7B OpenVLA基线相媲美，同时显著提高了推理效率。
- 方法：通过视觉方式编码状态-行动轨迹，作为视觉轨迹提示；在150K机器人操作轨迹数据集上微调OpenVLA
- 结果：在SimplerEnv的137个配置中达到最先进性能，比OpenVLA高10%；在真实WidowX机器人4个任务上比OpenVLA高3.5倍
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-b1CVu9l5GO-9e3065e500) · [原文](https://iclr.cc/virtual/2025/poster/29130)

#### VICtoR: Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation

![VICtoR: Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-UpQLu9bzAR-f68492d0dc.webp)

- 导读：我们研究通过从无动作视频和语言指令中学习来为长程操作任务建模奖励，这被称为视觉-指令相关性（VIC）问题。跨模态建模的最新进展突出了通过视觉和语言相关性进行奖励建模的潜力。然而，现有的VIC方法由于缺乏子阶段意识、难以建模任务复杂性以及对象状态估计不足，在学习长程任务的奖励方面面临挑战。为了解决这些问题，我们引入了VICtoR，一种新颖的层次化VIC奖励模型，能够为长程操作任务提供有效的奖励信号。VICtoR通过新颖的阶段检测器和运动进度评估器精确评估不同层次的任务进度，为智能体有效学习任务提供有见地的指导。为了验证VICtoR的有效性，我们在模拟和真实环境中进行了大量实验。结果表明，VICtoR优于现有的最佳VIC方法，在长程任务中成功率提高了43%。
- 方法：提出层次化VIC奖励模型VICtoR，包含阶段检测器和运动进度评估器；阶段检测器评估任务子阶段进度
- 结果：在模拟和真实环境中，VICtoR优于现有最佳VIC方法；长程任务成功率提升43%
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-UpQLu9bzAR-f68492d0dc) · [原文](https://iclr.cc/virtual/2025/poster/29454)

#### VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning

![VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-QOfswj7hij-723d69f4cb.webp)

- 导读：广泛智能的智能体应形成任务特定的抽象，选择性地暴露任务的关键要素，同时抽象掉原始传感器运动空间的复杂性。在这项工作中，我们提出了神经符号谓词，一种一阶抽象语言，结合了符号和神经知识表示的优势。我们概述了一种在线算法，用于发明此类谓词并学习抽象世界模型。我们将我们的方法与层次强化学习、视觉-语言模型规划和符号谓词发明方法进行比较，在五个模拟机器人领域的分布内和分布外任务上。结果表明，我们的方法提供了更好的样本复杂度、更强的分布外泛化能力和改进的可解释性。
- 方法：提出神经符号谓词作为一阶抽象语言，融合符号和神经知识表示。；设计在线算法来自动发明谓词并学习抽象世界模型。
- 结果：在五个模拟机器人领域的分布内外任务上，相比基线方法表现更优。；获得了更好的样本复杂度、更强的分布外泛化能力和改进的可解释性。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-QOfswj7hij-723d69f4cb) · [原文](https://iclr.cc/virtual/2025/poster/29691)

### P2

#### Long Context Compression with Activation Beacon

![Long Context Compression with Activation Beacon](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-1eQT9OzfNQ-8458f67998.webp)

- 导读：长上下文压缩是一个关键的研究问题，因为它对于降低LLMs的高计算和内存成本具有重要意义。在本文中，我们提出了Activation Beacon，一个用于基于Transformer的LLMs的插件模块，旨在实现有效、高效和灵活的长上下文压缩。为此，我们的方法引入了以下技术设计。1）我们直接压缩激活（即每层的键和值），而不是利用软提示来传递信息（这是封装长上下文中复杂信息的主要瓶颈）。2）我们定制了压缩工作流，其中每个细粒度的输入单元被逐步压缩，从而在训练和推理期间实现高质量的压缩和高效计算。3）我们通过基于压缩的自回归训练模型，充分利用纯文本和指令数据来优化模型的压缩性能。4）在训练期间，我们在每一步随机采样压缩比，教导模型支持广泛的压缩配置。我们在各种长上下文任务上进行了广泛评估，这些任务的长度（例如128K）可能远远超过最大训练长度（20K），例如文档理解、少样本学习和大海捞针。尽管现有方法难以处理这些具有挑战性的任务，Activation Beacon在各种场景下保持了与未压缩基线相当的性能，实现了2倍的推理加速和8倍的KV缓存内存成本降低。我们的数据、模型和代码已在https:…
- 方法：直接压缩每层的键和值激活，避免软提示的信息瓶颈；采用渐进式压缩工作流，逐步压缩细粒度输入单元
- 结果：与未压缩基线性能相当；实现2倍推理加速
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-1eQT9OzfNQ-8458f67998) · [原文](https://iclr.cc/virtual/2025/poster/31191)

#### Looking Backward: Streaming Video-to-Video Translation with Feature Banks

![Looking Backward: Streaming Video-to-Video Translation with Feature Banks](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-AMkf7h7HER-d7f09f0da2.webp)

- 导读：本文介绍了StreamV2V，一种基于扩散模型的实时流式视频到视频（V2V）翻译方法，支持用户提示。与以往使用批处理有限帧的V2V方法不同，我们选择以流式方式处理帧，以支持无限帧。StreamV2V的核心是后向观察原则，将当前帧与过去帧关联。这通过维护一个特征库实现，该库存档了过去帧的信息。对于新输入的帧，StreamV2V将自注意力机制扩展到包含库中的键和值，并直接将相似的过去特征融合到输出中。特征库通过合并存储和新特征不断更新，使其紧凑且信息丰富。StreamV2V以其适应性和效率脱颖而出，无需微调即可无缝集成到图像扩散模型中。它在一块A100 GPU上可达到20 FPS，分别比FlowVid、CoDeF、Rerender和TokenFlow快15倍、46倍、108倍和158倍。定量指标和用户研究证实了StreamV2V在保持时间一致性方面的卓越能力。
- 方法：采用流式处理框架，支持无限帧视频翻译。；引入后向观察原则，通过特征库关联当前与过去帧。
- 结果：在A100 GPU上实现20 FPS实时处理。；比FlowVid、CoDeF、Rerender和TokenFlow分别快15倍、46倍、108倍和158倍。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-AMkf7h7HER-d7f09f0da2) · [原文](https://iclr.cc/virtual/2025/poster/32105)

#### SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding

![SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-Hz4BYVY8YM-62463e2e7e.webp)

- 导读：尽管大型视觉语言模型（LVLMs）在已有基准上取得了显著进展，但在新兴的长上下文流式视频理解领域，仍缺乏合适的评估方法。当前视频理解基准通常强调孤立的单实例文本输入，未能评估模型在整个视频流中维持时间推理的能力。为解决这些局限，我们引入了SVBench，一个创新的基准，采用时间多轮问答链，专门用于全面评估当前LVLMs的流式视频理解能力。我们设计了一个半自动标注流程，获得了1,353个流式视频的49,979个问答对，包括生成代表视频片段上连续多轮对话的问答链，并构建相邻问答链之间的时间连接。我们的实验结果来自14个模型在对话和流式评估中，显示闭源GPT-4o表现最佳，但大多数开源LVLMs在长上下文流式视频理解上表现不佳。我们还构建了StreamingChat模型，该模型在SVBench上显著优于开源LVLMs，并在多种视觉语言基准上达到可比性能。我们希望SVBench通过提供对当前LVLMs的全面深入分析，推动流式视频理解的研究。我们的基准和模型可在 https://github.com/sotayang/SVBench 获取。
- 方法：设计半自动标注流程，生成1,353个流式视频的49,979个问答对。；构建时间多轮问答链，代表视频片段上的连续多轮对话，并建立时间连接。
- 结果：闭源GPT-4o表现最佳，而大多数开源LVLMs在长上下文流式视频理解上表现不佳。；StreamingChat模型在SVBench上显著优于开源LVLMs，并在多种视觉语言基准上达到可比性能。
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-Hz4BYVY8YM-62463e2e7e) · [原文](https://iclr.cc/virtual/2025/poster/30198)

#### Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge

![Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-JbPb6RieNC-64f842ff30.webp)

- 导读：近期大语言模型(LLM)的进展催生了视频-LLM模型，通过连接视频数据与语言任务推动了多模态学习。然而当前视频理解模型存在三大局限：长视频序列处理困难、多轮对话支持不足、难以适应真实动态场景。为此我们提出StreamChat——一个免训练的流式视频推理与会话交互框架。该框架采用创新的分层记忆系统高效处理并压缩长视频序列特征，实现实时多轮对话；通过并行系统调度策略提升处理速度并降低延迟；同时推出StreamBench多维度评估基准，涵盖多种媒体类型和交互场景的流式视频理解能力。在StreamBench及其他公开基准上的大量实验表明，StreamChat在准确率和响应时间上显著超越现有最优模型。
- 方法：采用分层记忆系统高效压缩长视频特征；并行系统调度策略提升处理速度
- 结果：在StreamBench基准上显著超越现有最优模型；实现实时多轮视频对话交互
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-JbPb6RieNC-64f842ff30) · [原文](https://iclr.cc/virtual/2025/poster/30097)

#### MELODI: Exploring Memory Compression for Long Contexts

![MELODI: Exploring Memory Compression for Long Contexts](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-TvGPP8i18S-07a983dd5b.webp)

- 导读：我们提出MELODI，一种新颖的记忆架构，旨在使用短上下文窗口高效处理长文档。MELODI背后的关键原理是将短期记忆和长期记忆表示为跨网络层和上下文窗口的分层压缩方案。具体而言，短期记忆通过多层上下文窗口的循环压缩实现，确保窗口间的平滑过渡；而长期记忆在单个中间层内进行进一步压缩，并跨上下文窗口聚合信息，有效整合整个历史中的关键信息。与使用大容量长期记忆（64K键值对）密集注意力的强基线——Memorizing Transformer相比，我们的方法在各种长上下文数据集上展现出更优的性能，同时将内存占用显著降低8倍。
- 方法：将短期记忆和长期记忆表示为跨网络层和上下文窗口的分层压缩方案。；短期记忆通过多层的循环压缩实现上下文窗口间的平滑过渡。
- 结果：在多种长上下文数据集上优于使用密集注意力的大容量长期记忆基线（Memorizing Transformer）。；内存占用显著降低至基线模型的1/8。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-TvGPP8i18S-07a983dd5b) · [原文](https://iclr.cc/virtual/2025/poster/29502)

#### VideoWebArena: Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks

![VideoWebArena: Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/iclr2025-unDQOUah0F-2a2bad6456.webp)

- 导读：视频常用于学习或提取完成任务所需的信息，其作用远超文本和静态图像。然而，现有许多智能体基准忽视了长期上下文视频理解，而侧重于文本或静态图像输入。为填补这一空白，我们提出了VideoWebArena（VideoWA），这是一个用于评估长上下文多模态智能体视频理解能力的基准。VideoWA包含2021个基于人工制作视频教程的网络智能体任务，总时长近4小时。我们定义了长上下文视频任务分类法，重点关注技能保留和事实保留两大类。技能保留任务评估智能体是否能够利用给定的人类演示高效完成任务，而事实保留任务则评估智能体是否能够从视频中检索与指令相关的信息以完成任务。我们发现，最佳模型在事实保留任务上仅达到13.3%的成功率，在事实保留QA对上为45.8%，远低于人类表现（分别为73.9%和79.3%）。在技能保留任务上，使用教程后长上下文模型的表现反而更差，在WebArena任务中性能下降5%，在VisualWebArena任务中下降10.3%。我们的工作凸显了提升长上下文多模态模型智能体能力的必要性，并为未来基于长上下文视频智能体的开发提供了测试平台。
- 方法：构建包含2021个基于视频教程的网页任务，总时长近4小时。；定义长上下文视频任务分类：技能保留（模仿演示完成任务）和事实保留（从视频中检索信息）。
- 结果：最佳模型在事实保留任务上仅13.3%成功，QA对45.8%，远低于人类（73.9%和79.3%）。；在技能保留任务上，长上下文模型使用教程后性能反而下降5%（WebArena）和10.3%（VisualWebArena）。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/iclr2025-unDQOUah0F-2a2bad6456) · [原文](https://iclr.cc/virtual/2025/poster/27960)

## ICML 2025

本轮收录 10 篇。

### P0

#### A Large Recurrent Action Model: xLSTM enables Fast Inference for Robotics Tasks

![A Large Recurrent Action Model: xLSTM enables Fast Inference for Robotics Tasks](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/icml2025-J2JxJ0P1LI-4ee08d24ad.webp)

- 导读：近年来，强化学习领域出现了一种趋势，即通过序列建模在大型数据集上离线训练大型动作模型。现有模型主要基于Transformer架构，虽然性能强大，但由于推理速度慢，不适用于机器人等实时应用。最近，现代循环架构（如xLSTM和Mamba）被提出，它们在训练时具有与Transformer类似的并行化优势，同时支持快速推理。本文研究了这些现代循环架构在大型动作模型中的适用性，并提出了一种以xLSTM为核心的大型循环动作模型（LRAM），该模型具有线性时间推理复杂度和自然的序列长度外推能力。在来自6个领域的432个任务上的实验表明，LRAM在性能和速度方面均优于Transformer。
- 方法：以xLSTM为核心构建大型循环动作模型（LRAM）；利用现代循环架构实现线性时间推理
- 结果：在6个领域的432个任务上，LRAM优于Transformer；LRAM在性能和速度上均表现更好
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2025-J2JxJ0P1LI-4ee08d24ad) · [原文](https://icml.cc/virtual/2025/poster/45715)

#### SAM2Act: Integrating Visual Foundation Model with A Memory Architecture for Robotic Manipulation

- 导读：针对在多样、动态环境中运行的机器人操作系统需具备多任务交互、泛化到未见场景和空间记忆等关键能力，现有方法在处理复杂环境变化和记忆依赖任务时存在不足。为此，我们提出SAM2Act，一种基于多视图机器人变换器的策略，利用多分辨率上采样和大规模基础模型的视觉表征。SAM2Act在RLBench基准的18个任务上达到86.8%的平均成功率，并在Colosseum基准上仅出现4.3%的性能差距，展现出强大的泛化能力。在此基础上，我们提出SAM2Act+，一种受SAM2启发的记忆架构，包含记忆库、编码器和注意力机制以增强空间记忆。为评估记忆依赖任务，我们引入新基准MemoryBench。SAM2Act+在MemoryBench的记忆任务上平均成功率达94.3%，显著超越现有方法，推动了记忆型机器人系统的发展。项目页面：sam2act.github.io。
- 方法：基于多视图机器人变换器，结合多分辨率上采样与大规模视觉基础模型。；提出SAM2Act+，包含记忆库、编码器和注意力机制的空间记忆架构。
- 结果：在RLBench的18个任务上平均成功率86.8%。；在Colosseum基准上仅4.3%性能差距。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2025-anSWDvJm8v-a844d23462) · [原文](https://icml.cc/virtual/2025/poster/44779)

### P1

#### Closed-Loop Long-Horizon Robotic Planning via Equilibrium Sequence Modeling

![Closed-Loop Long-Horizon Robotic Planning via Equilibrium Sequence Modeling](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/icml2025-MCqlamfhAy-555183f320.webp)

- 导读：In the endeavor to make autonomous robots take actions, task planning is a major challenge that requires translating high-level task descriptions to long-horizon action sequences.
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2025-MCqlamfhAy-555183f320) · [原文](https://icml.cc/virtual/2025/poster/45554) · [代码](https://github.com/Singularity0104/equilibrium-planner)

#### Efficient Robotic Policy Learning via Latent Space Backward Planning

![Efficient Robotic Policy Learning via Latent Space Backward Planning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/icml2025-DJiouYdH19-4c159e5720.webp)

- 导读：当前的机器人规划方法通常依赖于预测具有完整像素细节的多帧图像。虽然这种细粒度的方法可以作为通用的世界模型，但它给下游策略学习带来了两个重大挑战：巨大的计算成本阻碍了实时部署，以及累积的不准确性可能导致动作提取的误导。使用粗粒度子目标进行规划部分缓解了效率问题。然而，它们的前向规划方案仍然可能由于累积误差导致偏离任务的预测，从而与长期目标不一致。这提出了一个关键问题：机器人规划能否既高效又足够准确，以支持长时域、多阶段任务中的实时控制？为了解决这个问题，我们提出了一种潜在空间后向规划方案（LBP），它首先将任务接地到最终的潜在目标，然后递归地预测更接近当前状态的中间子目标。接地后的最终目标使后向子目标规划始终意识到任务完成，从而在整个规划视野中促进任务导向的预测。子目标条件策略包含一个可学习的标记，用于总结子目标序列，并决定每个子目标如何指导动作提取。通过大量仿真和真实机器人长期实验，我们表明LBP优于现有的细粒度和前向规划方法，达到了最先进的性能。项目页面：https://lbp-authors.github.io
- 方法：将任务接地到最终潜在目标，并递归预测更接近当前状态的中间子目标（后向规划）。；子目标条件策略使用可学习标记来总结子目标序列，并指导动作提取。
- 结果：在仿真和真实机器人长期实验中，LBP优于现有的细粒度和前向规划方法。；达到最先进的性能（SOTA）。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2025-DJiouYdH19-4c159e5720) · [原文](https://icml.cc/virtual/2025/poster/46016)

#### Empowering World Models with Reflection for Embodied Video Prediction

![Empowering World Models with Reflection for Embodied Video Prediction](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/icml2025-onumui0nHi-6446ba1a35.webp)

- 导读：视频生成模型在模拟未来状态方面取得了显著进展，展示了其作为场景世界模拟器的潜力。然而，现有模型通常缺乏稳健的理解能力，限制了其执行多步预测或处理分布外（OOD）场景的能力。为应对这一挑战，我们提出了生成反思（RoG），这是一组中间推理策略，旨在增强视频预测。它利用了预训练视觉语言和视频生成模型的互补优势，使其能够在场景中作为世界模型运行。为支持RoG，我们引入了场景视频预测基准（EVA-Bench），这是一个全面的基准，用于评估场景世界模型在各种任务和场景中的表现，包括域内和OOD数据集。在此基础上，我们设计了一个世界模型——场景视频预测器（EVA），它采用多阶段训练范式生成高保真视频帧，并应用自回归策略实现对更长视频序列的自适应泛化。大量实验证明了EVA在视频生成和机器人等各种下游任务中的有效性，从而为大规模预训练模型在真实世界视频预测应用中的应用铺平了道路。视频演示可在https://sites.google.com/view/icml-eva获取。
- 方法：提出生成反思（RoG）策略，结合视觉语言与视频生成模型。；构建场景视频预测基准（EVA-Bench），包含域内和OOD数据集。
- 结果：EVA在视频生成和机器人等下游任务中展现有效性。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2025-onumui0nHi-6446ba1a35) · [原文](https://icml.cc/virtual/2025/poster/44044)

#### FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making

![FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/icml2025-UTT5OTyIWm-33e127ef28.webp)

- 导读：基础模型和世界模型在不同层面上具有互补的任务泛化优势。本文提出FOUNDER框架，通过将基础模型的通用知识与世界模型的动态建模能力相结合，在无需奖励的情况下实现具身环境中的开放式任务求解。我们学习一个映射函数，将基础模型表示映射到世界模型状态空间，从而从外部观测推断智能体在世界模拟器中的物理状态。该映射使得行为学习阶段能够通过想象学习目标条件策略，其中映射任务作为目标状态。我们利用到目标状态的预测时间距离作为奖励信号。FOUNDER在多种多任务离线视觉控制基准上表现出优越性能，尤其在处理复杂观测或领域差距的场景中优于先前方法，能够深入理解文本或视频指定任务的语义。实验验证了所学奖励函数与真实奖励的一致性。
- 方法：学习一个映射函数，将基础模型表示（如文本或视频）映射到世界模型状态空间。；通过想象在学到的世界模型中进行行为学习，以映射后的任务状态作为目标。
- 结果：在多种多任务离线视觉控制基准上表现优越，尤其是复杂观测和领域差距场景。；验证了所学奖励函数与真实奖励的一致性。
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2025-UTT5OTyIWm-33e127ef28) · [原文](https://icml.cc/virtual/2025/poster/45117)

#### LARM: Large Auto-Regressive Model for Long-Horizon Embodied Intelligence

![LARM: Large Auto-Regressive Model for Long-Horizon Embodied Intelligence](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/icml2025-zcx7jqUZg5-ab6c32f0f5.webp)

- 导读：近期具身智能体主要基于强化学习（RL）或大语言模型（LLM）构建。其中，RL智能体部署高效但只能执行极少数任务。相反，巨型LLM智能体（参数常超过1000B）展现出强泛化能力，但消耗巨大计算资源。本研究通过在我们开发的大型自回归模型（LARM）上执行所提出的裁判强化学习，结合两者优势并避免缺点。具体而言，LARM基于轻量级LLM（参数少于5B）构建，直接输出要执行的下一个动作而非文本。我们从数学上揭示经典RL反馈在长视野具身探索中消失，并引入一个基于巨型LLM的裁判来处理LARM训练中的奖励消失问题。通过这种方式，LARM学会完成多样化的开放世界任务而无需人工干预。尤其，LARM成功在Minecraft中收获附魔钻石装备，这需要比先前最佳方法的最高成就显著更长的决策链。
- 方法：构建轻量级自回归模型LARM（<5B参数），直接输出动作。；提出裁判强化学习，利用巨型LLM裁判解决长视野奖励消失问题。
- 结果：LARM无需人工干预完成多样化开放世界任务。；成功在Minecraft中收获附魔钻石装备，决策链显著长于先前最佳方法。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2025-zcx7jqUZg5-ab6c32f0f5) · [原文](https://icml.cc/virtual/2025/poster/43466)

#### Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning

![Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/icml2025-Bv7LUUYOiq-354ebcc039.webp)

- 导读：机器人操作中的长时域任务在强化学习（RL）中带来了重大挑战，因为设计稠密奖励函数和有效探索庞大的状态-动作空间非常困难。然而，尽管缺乏稠密奖励，这些任务通常具有多阶段结构，可以利用该结构将整体目标分解为可管理的子目标。在这项工作中，我们提出了DEMO3，一个利用这种结构从视觉输入进行高效学习的框架。具体来说，我们的方法将多阶段稠密奖励学习、双阶段训练方案以及世界模型学习整合到一个精心设计的演示增强RL框架中，该框架极大地缓解了长时域任务中的探索挑战。我们的评估表明，与最先进的方法相比，我们的方法平均提高了40%的数据效率，在特别困难的任务上提高了70%。我们在涵盖四个领域的16个稀疏奖励任务上验证了这一点，包括使用仅五次演示的具有挑战性的人形视觉控制任务。
- 方法：多阶段稠密奖励学习；双阶段训练方案
- 结果：平均数据效率提升40%，困难任务提升70%；在16个稀疏奖励任务上验证，涵盖四个领域
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2025-Bv7LUUYOiq-354ebcc039) · [原文](https://icml.cc/virtual/2025/poster/46087)

#### One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation

![One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/icml2025-E2VsqgKNlr-9c3a0b4b22.webp)

- 导读：扩散模型因在生成任务中的成功而受到赞誉，并被越来越多地应用于机器人领域，在行为克隆方面表现出卓越性能。然而，其由于迭代去噪步骤导致的缓慢生成过程，给资源受限的机器人设置和动态变化环境中的实时应用带来了挑战。本文介绍了一步扩散策略（OneDP），这是一种新颖的方法，将预训练扩散策略的知识提炼为单步动作生成器，显著加速了机器人控制任务的响应时间。我们确保提炼的生成器与原始策略分布紧密对齐，通过最小化扩散链上的KL散度，仅需额外2％-10％的预训练成本即可收敛。我们在6个具有挑战性的模拟任务以及4个使用Franka机器人自行设计的真实任务上评估了OneDP。结果表明，OneDP不仅实现了最先进的成功率，而且推理速度提高了数量级，将动作预测频率从1.5 Hz提升到62 Hz，展示了其在动态和计算受限机器人应用中的潜力。项目页面：https://research.nvidia.com/labs/dir/onedp/。
- 方法：提出一步扩散策略（OneDP），将预训练扩散策略蒸馏为单步动作生成器。；通过最小化扩散链上的KL散度确保生成分布与原始策略分布对齐。
- 结果：在6个模拟任务和4个真实世界任务上取得最先进成功率。；推理速度提高数量级，动作预测频率从1.5 Hz提升至62 Hz。
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2025-E2VsqgKNlr-9c3a0b4b22) · [原文](https://icml.cc/virtual/2025/poster/45971)

#### World Model Implanting for Test-time Adaptation of Embodied Agents

![World Model Implanting for Test-time Adaptation of Embodied Agents](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/icml2025-tpbtodnI1p-ba4a34e8d5.webp)

- 导读：In embodied AI, a persistent challenge is enabling agents to robustly adapt to novel domains without requiring extensive data collection or retraining.
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/icml2025-tpbtodnI1p-ba4a34e8d5) · [原文](https://icml.cc/virtual/2025/poster/43758)

## CVPR 2025

本轮收录 6 篇。

### P0

#### 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning

![3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/cvpr2025-34125-4128eb533a.webp)

- 导读：构建紧凑且信息丰富的三维场景表示对于有效的具身探索和推理至关重要，尤其是在复杂环境和长时间任务中。现有表示（如以对象为中心的三维场景图）通过将场景建模为孤立对象并附加限制性的文本关系来过度简化空间关系，难以处理需要细微空间理解的查询。此外，这些表示缺乏主动探索和记忆管理的自然机制，阻碍了其在终身自主性中的应用。本文提出3D-Mem，一种用于具身智能体的三维场景记忆框架。3D-Mem使用信息丰富的多视图图像（称为记忆快照）来表示场景并捕获已探索区域的丰富视觉信息。它进一步通过引入前沿快照（对未探索区域的瞥见）集成了基于前沿的探索，使智能体能够同时考虑已知和潜在的新信息来做出明智决策。为了支持主动探索设置中的终身记忆，我们提出了3D-Mem的增量构建管道以及用于记忆管理的记忆检索技术。在三个基准上的实验结果表明，3D-Mem显著增强了智能体在三维环境中的探索和推理能力，突显了其在具身AI应用中的潜力。
- 方法：利用多视图图像（记忆快照）表示场景，捕获丰富视觉信息。；集成基于前沿的探索，通过前沿快照提供对未探索区域的瞥见。
- 结果：在三个基准上显著增强智能体的探索和推理能力。
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2025-34125-4128eb533a) · [原文](https://cvpr.thecvf.com/virtual/2025/poster/34125)

### P1

#### Spatial-Temporal Graph Diffusion Policy with Kinematic Modeling for Bimanual Robotic Manipulation

- 导读：尽管模仿学习在机器人操作中取得了显著成功，但其在双臂任务中的应用仍极具挑战性。现有方法主要学习策略以预测远距离的下一个最佳末端执行器姿态（NBP），然后使用逆运动学计算相应的关节旋转角度进行运动。然而，它们存在两个重要问题：（1）很少考虑物理机器人结构，可能导致自身碰撞或干扰；（2）忽视运动学约束，可能导致预测姿态不符合机器人关节的实际限制。在本文中，我们提出了运动学增强时空图扩散器（KStar Diffuser）。具体而言，（1）为了将物理机器人结构信息纳入动作预测，KStar Diffuser根据连续时间步上的物理双臂关节运动维护一个动态时空图。该动态图作为去噪动作的机器人结构条件；（2）为了使NBP学习目标与运动学一致，我们引入可微运动学为优化KStar Diffuser提供参考。该模块正则化策略，使其预测更可靠且具有运动学意识的下一个末端执行器姿态。实验结果表明，我们的方法在仿真和真实世界中均有效利用了物理结构信息并生成了运动学感知的动作。
- 方法：根据物理双臂关节运动构建动态时空图，作为去噪动作的机器人结构条件。；引入可微运动学提供参考，正则化策略预测运动学感知的下一个最佳姿态。
- 结果：有效利用物理结构信息。；生成运动学感知的动作。
- 对 VPWEM：策略基线：可用于控制动作生成器和时序建模的影响。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2025-33065-4db506b3c6) · [原文](https://cvpr.thecvf.com/virtual/2025/poster/33065)

### P2

#### Layer- and Timestep-Adaptive Differentiable Token Compression Ratios for Efficient Diffusion Transformers

- 导读：扩散变换器（DiTs）在图像生成质量上达到了最先进水平，但存在高延迟和内存效率低下的问题，难以部署在资源受限设备上。一个主要效率瓶颈是现有DiTs对所有图像区域施加相同的计算量。然而，并非所有图像令牌都同等重要，某些局部区域（如物体）需要更多计算。为了解决这个问题，我们提出了DiffCR，一种具有可微分压缩率的动态DiT推理框架，它自动学习为每个图像令牌在层和时间步上动态路由计算，从而实现高效的DiTs。具体而言，DiffCR集成了三个特点：（1）一个令牌级路由方案，每个DiT层包含一个与模型权重联合微调的路由器，用于预测令牌重要性分数。这样，不重要的令牌跳过整个层的计算；（2）一种层间可微分比率机制，不同DiT层从零初始化自动学习不同的压缩率，导致冗余层具有大压缩率，而其他层保持较少压缩甚至不压缩；（3）一种时间步可微分比率机制，每个去噪时间步学习自己的压缩率。生成的模式显示，噪声较大的时间步具有更高的比率，而随着图像变清晰，比率降低。在文本到图像和修复任务上的大量实验表明，DiffCR有效捕捉了令牌、层和时间步轴上的动态性，相比先前工作实现了生成质量与效率之间的优越权衡。项目网站…
- 方法：令牌级路由：每个DiT层包含一个路由器，预测令牌重要性并跳过不重要令牌的计算。；层间可微分压缩率：不同层从零初始化自动学习不同压缩率。
- 结果：在文本到图像和修复任务上，DiffCR有效捕捉了令牌、层和时间步的动态性。；相比先前工作，实现了生成质量与效率的更优权衡。
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2025-34258-27672a9ad0) · [原文](https://cvpr.thecvf.com/virtual/2025/poster/34258)

#### Learning from Streaming Video with Orthogonal Gradients

![Learning from Streaming Video with Orthogonal Gradients](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/cvpr2025-33786-375eb00dc4.webp)

- 导读：我们解决了从连续视频流中进行自监督表示学习的挑战。这与标准的视频学习方法不同，后者在训练过程中对视频进行切分和打乱，以创建满足传统训练范式所期望的独立同分布（IID）样本假设的非冗余批次。当视频仅作为连续输入流时，IID假设显然被打破，导致性能下降。我们在三个任务上展示了从打乱学习到顺序学习时的性能下降：单视频表示学习方法DoRA、多视频数据集上的标准VideoMAE以及未来视频预测任务。为了解决性能下降问题，我们提出了一种对标准优化器的几何修改，通过在训练过程中利用正交梯度来解相关批次。该修改可以应用于任何优化器——我们使用随机梯度下降（SGD）和AdamW进行了演示。我们提出的正交优化器使得从流视频训练的模型能够缓解下游任务评估中的表示学习性能下降。在三个场景（DoRA、VideoMAE、未来预测）中，我们展示了我们的正交优化器在所有三个场景中都优于强大的AdamW。
- 方法：对标准优化器进行几何修改；利用正交梯度解相关批次
- 结果：从打乱学习转为顺序学习时性能下降；正交优化器缓解了性能下降
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2025-33786-375eb00dc4) · [原文](https://cvpr.thecvf.com/virtual/2025/poster/33786)

#### OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts

![OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/cvpr2025-32692-47f2f2fec6.webp)

- 导读：多模态语言模型（MLLMs）如GPT-4o的快速发展推动了全模态语言模型（OmniLLMs）的发展，这类模型旨在处理和主动响应连续的多模态数据流。尽管潜力巨大，但在流式视频环境中评估其真实世界交互能力仍然是一个巨大挑战。在这项工作中，我们引入了OmniMMI，一个专门为流式视频场景下OmniLLMs设计的全面多模态交互基准。OmniMMI包含超过1121个视频和2290个问题，解决了现有视频基准中两个关键但尚未充分探索的挑战：流式视频理解和主动推理，涵盖六个不同的子任务。此外，我们提出了一种新的框架——多模态复用建模（M4），旨在实现一个推理高效的流式模型，可以同时观看、聆听和生成。
- 方法：提出OmniMMI基准，包含1121个视频和2290个问题，覆盖六个子任务；提出多模态复用建模（M4）框架，实现推理高效的流式模型（看到、听到并生成）
- 结果：构建了包含1121个视频和2290个问题的OmniMMI基准；提出了多模态复用建模（M4）框架
- 对 VPWEM：流式启发：关注在线更新以及成本是否随回合长度保持有界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2025-32692-47f2f2fec6) · [原文](https://cvpr.thecvf.com/virtual/2025/poster/32692) · [代码](https://omnimmi.github.io)

#### Zero-shot 3D Question Answering via Voxel-based Dynamic Token Compression

![Zero-shot 3D Question Answering via Voxel-based Dynamic Token Compression](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/cvpr2025-33335-823b8aeca7.webp)

- 导读：近期3D大型多模态模型（3D-LMMs）的进展推动了3D问答的显著进步。然而，最近的多帧视觉语言模型（VLMs）在3D问答任务上表现出优于3D-LMMs的性能，这主要得益于可用的2D图像数据在规模和多样性上远超有限的3D数据。多帧VLMs虽然性能优越，但在限制视觉token数量的同时保留3D场景中所有详细视觉信息方面存在困难。常见的token池化等方法减少了视觉token使用，但往往导致信息丢失，损害模型保留3D问答任务所需视觉细节的能力。为解决此问题，我们提出基于体素的动态token压缩（DTC），它结合3D空间先验和视觉语义，对当前多帧VLMs实现超过90%的视觉token使用减少。我们的方法在OpenEQA和ScanQA等3D问答基准上保持与最先进模型相当的性能，证明了其有效性。
- 方法：提出基于体素的动态token压缩（DTC）；结合3D空间先验和视觉语义
- 结果：在OpenEQA和ScanQA基准上与最先进模型性能相当；保持性能的同时减少视觉token使用超过90%
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/cvpr2025-33335-823b8aeca7) · [原文](https://cvpr.thecvf.com/virtual/2025/poster/33335)

## AAAI 2025

本轮收录 2 篇。

### P1

#### Instruction-Augmented Long-Horizon Planning: Embedding Grounding Mechanisms in Embodied Mobile Manipulation

![Instruction-Augmented Long-Horizon Planning: Embedding Grounding Mechanisms in Embodied Mobile Manipulation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/aaai2025-article-33610-b44f398c44.webp)

- 导读：使类人机器人基于具身感知和理解能力在真实环境中进行长期移动操作规划一直是一个长期挑战。随着大型语言模型（LLMs）的兴起，基于LLM的规划器显著增加。这些方法要么利用人类提供的真实世界文本表示，要么严重依赖提示工程来提取此类表示，缺乏定量理解环境的能力，例如确定操作物体的可行性。为了解决这些限制，我们提出了指令增强的长期规划（IALP）系统，这是一种新颖的框架，利用LLMs基于实时传感器反馈（包括环境的接地知识）在闭环交互中生成可行且最优的行动。与先前工作不同，我们的方法通过利用LLMs的抽象推理能力和接地机制，将用户指令增强为PDDL问题。通过在真实世界中进行各种长期任务（每个任务包含七种不同的操作技能），我们的结果表明，IALP系统可以高效地解决这些任务，平均成功率超过80%。我们提出的方法可以作为高级规划器，通过利用多模态传感器输入，使机器人在非结构化环境中具有显著的自主性。
- 方法：利用LLM的推理能力和接地机制，将用户指令增强为PDDL问题；；基于实时传感器反馈进行闭环交互，生成可行最优动作；
- 结果：在包含七种不同操作技能的真实世界长期任务中，平均成功率超过80%。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2025-article-33610-b44f398c44) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/33610)

### P2

#### Planning from Imagination: Episodic Simulation and Episodic Memory for Vision-and-Language Navigation

![Planning from Imagination: Episodic Simulation and Episodic Memory for Vision-and-Language Navigation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/aaai2025-article-32679-b9a275fa4c.webp)

- 导读：人类通过情景模拟和情景记忆在陌生环境中导航，这有助于深入理解环境与物体之间的复杂关系。开发受人类机制启发的想象记忆系统可以增强具身代理在未知环境中的导航性能。然而，现有的视觉与语言导航（VLN）代理缺乏这种记忆机制。为此，我们提出了一种新颖的架构，为代理配备虚实混合记忆系统。该系统使代理能够通过想象机制和导航动作来维持和扩展记忆。此外，我们设计了定制的预训练任务来培养代理的想象能力。我们的代理能够想象未来场景的高保真RGB图像，在基于路径长度的成功率（SPL）上取得了最先进的结果。
- 方法：提出虚实混合记忆系统，通过想象机制和导航动作维持和扩展记忆。；设计定制的预训练任务来培养代理的想象能力。
- 结果：代理能够想象未来场景的高保真RGB图像。；在基于路径长度的成功率（SPL）上取得了最先进的结果。
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/aaai2025-article-32679-b9a275fa4c) · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/32679)

## WACV 2025

本轮收录 3 篇。

### P0

#### Enhancing Embodied Object Detection with Spatial Feature Memory

![Enhancing Embodied Object Detection with Spatial Feature Memory](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/wacv2025-dc357a432a-298dbbfa3c.webp)

- 导读：Deep-learning and large scale language-image training have produced image object detectors that generalise well to diverse environments and semantic classes.
- 对 VPWEM：直接相关：需比较记忆内容、更新方式、容量和单回合边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/wacv2025-dc357a432a-298dbbfa3c) · [原文](https://openaccess.thecvf.com/content/WACV2025/html/Chapman_Enhancing_Embodied_Object_Detection_with_Spatial_Feature_Memory_WACV_2025_paper.html)

### P1

#### Incorporating Task Progress Knowledge for Subgoal Generation in Robotic Manipulation through Image Edits

- 导读：理解任务进度使得人类不仅能追踪已完成的工作，还能更好地规划未来的目标。我们提出了TaKSIE，一个新颖的框架，将任务进度知识融入机器人操作任务的视觉子目标生成中。我们联合训练一个循环网络和一个潜在扩散模型，根据机器人的当前观察和输入的语言命令生成下一个视觉子目标。在执行时，机器人利用视觉进度表示来监控任务进度，并自适应地从模型中采样下一个视觉子目标以指导操作策略。我们在模拟和真实世界的机器人任务中训练和验证了我们的模型，在CALVIN操作基准上达到了最先进的性能。我们发现，任务进度知识的纳入可以增强训练策略对于不同初始机器人姿态或演示中不同移动速度的鲁棒性。项目网站：https://live-robotics-uva.github.io/TaKSIE/。
- 方法：联合训练循环网络与潜在扩散模型，基于当前观察和语言指令生成下一个视觉子目标。；利用视觉进度表示监控任务进度，自适应地采样子目标以指导操作策略。
- 结果：在CALVIN操作基准上达到最先进性能。；任务进度知识提升了不同初始位姿和移动速度下的策略鲁棒性。
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/wacv2025-6b73234651-9e3c619472) · [原文](https://openaccess.thecvf.com/content/WACV2025/html/Kang_Incorporating_Task_Progress_Knowledge_for_Subgoal_Generation_in_Robotic_Manipulation_WACV_2025_paper.html)

### P2

#### MemFusionMap: Working Memory Fusion for Online Vectorized HD Map Construction

![MemFusionMap: Working Memory Fusion for Online Vectorized HD Map Construction](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/wacv2025-517e46eead-be125739a0.webp)

- 导读：高精地图为自动驾驶系统提供环境信息，是安全规划的基础。尽管现有单帧输入方法在在线矢量高精地图构建中表现优异，但在复杂场景和遮挡下仍存在困难。我们提出MemFusionMap，一种具有增强时序推理能力的新型时序融合模型，用于在线高精地图构建。具体而言，我们贡献了一个工作记忆融合模块，提升模型跨帧推理的记忆能力；并设计了一种新颖的时序重叠热图，在鸟瞰视角下显式告知模型时序重叠信息和车辆轨迹。通过整合这两个设计，MemFusionMap显著优于现有方法，同时保持了可扩展的通用设计。我们在开源基准上进行了广泛评估，在mAP上相比最先进方法实现了最高5.4%的提升。项目页面：https://song-jingyu.github.io/MemFusionMap
- 方法：提出工作记忆融合模块，增强模型跨帧推理的记忆能力；设计时序重叠热图，显式提供鸟瞰视角下的时序重叠和车辆轨迹信息
- 结果：在开源基准上，mAP相比最先进方法最高提升5.4%
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/wacv2025-517e46eead-be125739a0) · [原文](https://openaccess.thecvf.com/content/WACV2025/html/Song_MemFusionMap_Working_Memory_Fusion_for_Online_Vectorized_HD_Map_Construction_WACV_2025_paper.html)

## AAMAS 2025

本轮收录 0 篇。

本轮未发现明确相关论文。

## ACL 2025

本轮收录 7 篇。

### P1

#### World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning

![World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/acl2025-2025.acl-long.1044-dd61fb0d1a.webp)

- 导读：近期大型视觉语言模型（LVLM）在具身任务规划方面展现出潜力，但仍面临依赖约束和效率等基本挑战。现有方法要么只优化动作选择，要么在推理时利用世界模型，忽视了将学习建模世界作为增强规划能力的一种方式的好处。我们提出了双偏好优化（D²PO），一种新的学习框架，通过偏好学习联合优化状态预测和动作选择，使LVLM能够理解环境动态以进行更好的规划。为了在没有人工标注的情况下自动收集轨迹和逐步偏好数据，我们引入了一种树搜索机制，通过试错进行广泛探索。在VoTa-Bench上的大量实验表明，我们的基于D²PO的方法在应用于Qwen2-VL（7B）、LLaVA-1.6（7B）和LLaMA-3.2（11B）时，显著优于现有方法和GPT-4o，实现了更高的任务成功率和更高效的执行路径。
- 方法：提出双偏好优化（D²PO）框架，联合优化状态预测和动作选择；通过偏好学习使LVLM理解环境动态
- 结果：在VoTa-Bench上，基于D²PO的方法显著优于现有方法和GPT-4o；应用到多个LVLM（Qwen2-VL 7B、LLaVA-1.6 7B、LLaMA-3.2 11B）均实现更高任务成功率和更高效执行路径
- 对 VPWEM：替代路线：比较预测未来与压缩过去哪种信息更有效。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2025-2025.acl-long.1044-dd61fb0d1a) · [原文](https://aclanthology.org/2025.acl-long.1044/)

### P2

#### A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression

![A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/acl2025-2025.acl-long.241-1e4daacaa6.webp)

- 导读：本文深入研究了基于gist标记的上下文压缩方法，以改进大语言模型中的长上下文处理。我们聚焦两个关键问题：（1）这些方法能在多大程度上替代全注意力模型？（2）压缩可能导致哪些潜在的失败模式？通过大量实验，我们发现基于gist的压缩在检索增强生成和长文档问答等任务中可达到近乎无损的性能，但在合成召回等任务中面临挑战。此外，我们识别出三种关键失败模式：边界丢失、意外丢失和沿途丢失。为缓解这些问题，我们提出两种有效策略：细粒度自编码（增强原始标记信息的重建）和分段标记重要性估计（根据标记依赖调整优化）。我们的工作为理解基于gist标记的上下文压缩提供了宝贵见解，并提出了改进压缩能力的实用策略。
- 方法：提出细粒度自编码，增强原始标记信息的重建；提出分段标记重要性估计，根据标记依赖调整优化
- 结果：在检索增强生成和长文档问答中达到近乎无损性能；在合成召回等任务中面临挑战
- 对 VPWEM：压缩启发：可迁移到历史 token 选择、压缩率和效率实验。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2025-2025.acl-long.241-1e4daacaa6) · [原文](https://aclanthology.org/2025.acl-long.241/)

#### Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition

![Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/acl2025-2025.acl-long.462-8d15370e8d.webp)

- 导读：大型语言模型具备普遍的語言能力，但其語言習得效率不如人類。本研究提出一種方法，將語言習得關鍵期（人類語言習得特別高效的階段）的工作記憶發展特性整合到語言模型的訓練過程中。該方法在訓練初期限制工作記憶容量，並隨著學習進展以指數方式逐漸放鬆此限制。針對性句法評估顯示，所提方法優於無記憶限制或靜態記憶限制的傳統方法。這些發現不僅為設計數據高效的語言模型提供了新方向，也間接支持了工作記憶發展特性作為語言習得關鍵期潛在機制的觀點。
- 方法：在訓練初期引入工作記憶限制，模擬人類關鍵期的初期階段。；以指數方式逐漸放鬆記憶限制，模擬工作記憶的發展特性。
- 结果：在句法評估上，所提方法優於無記憶限制或靜態記憶限制的傳統方法。；方法在數據效率上表現更好，但僅在句法測試上驗證。
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2025-2025.acl-long.462-8d15370e8d) · [原文](https://aclanthology.org/2025.acl-long.462/)

#### EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts

![EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/acl2025-2025.acl-long.574-9d42ec2036.webp)

- 导读：近年来，大型语言模型（LLM）在许多语言任务上取得了令人瞩目的成功。然而，使用LLM高效处理长上下文仍然是一个重大挑战。我们引入了EpMAN——一种方法，它在情景记忆模块中处理长上下文，同时整体地关注语义相关的上下文块。情景注意的输出用于在训练和生成期间重新加权解码器对上下文存储的KV缓存的自注意。当LLM解码器使用EpMAN进行训练时，其在多个具有挑战性的单跳长上下文回忆和问答基准上的表现，在16k到256k token范围内，比使用自注意训练的解码器和流行的检索增强生成框架更强且更鲁棒。
- 方法：构建情景记忆模块，将长上下文划分为多个块存储，并维护语义相关块的整体注意力。；情景注意输出用于重新加权解码器对上下文KV缓存的自注意力，在训练和生成阶段使用。
- 结果：在16k到256k token范围内，EpMAN在多个单跳长上下文回忆和问答基准上表现更强且更鲁棒。；优于使用自注意力训练的解码器和流行的检索增强生成框架。
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2025-2025.acl-long.574-9d42ec2036) · [原文](https://aclanthology.org/2025.acl-long.574/)

#### Improving Factuality with Explicit Working Memory

![Improving Factuality with Explicit Working Memory](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/acl2025-2025.acl-long.548-c511082b30.webp)

- 导读：大型语言模型可能会生成事实不准确的内容，即所谓幻觉。近期研究在检索增强生成基础上通过迭代提示改进事实性，但受限于传统RAG设计。为解决这些问题，我们提出EWE（显式工作记忆），一种通过集成工作记忆来增强长文本生成事实性的新方法，该记忆接收来自外部资源的实时反馈。基于在线事实核查和检索反馈刷新记忆，使EWE能在生成过程中纠正错误声明，确保更准确可靠的输出。实验表明，EWE在四个追求事实的长文本生成数据集上优于强基线，将事实性指标VeriScore绝对提高2到6个百分点，且不牺牲响应的有用性。进一步分析揭示，记忆更新规则设计、记忆单元配置以及检索数据存储质量是影响模型性能的关键因素。
- 方法：集成显式工作记忆，接收外部资源实时反馈；基于在线事实核查和检索反馈刷新记忆
- 结果：在四个事实性长文本生成数据集上，事实性指标VeriScore绝对提高2-6个百分点；不牺牲响应有用性
- 对 VPWEM：记忆表示启发：可对照情景记忆与工作记忆的职责划分。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2025-2025.acl-long.548-c511082b30) · [原文](https://aclanthology.org/2025.acl-long.548/)

#### SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation

![SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/acl2025-2025.acl-long.529-bdac307fb9.webp)

- 导读：键值（KV）缓存已成为长上下文生成中大语言模型的瓶颈。尽管该领域已有大量研究，但解码阶段的优化通常被忽视。然而，我们认为这种优化至关重要，尤其是对于长输出生成任务，基于以下两个观察：（i）预填充阶段过度压缩需要完整上下文，会损害推理任务的理解；（ii）在长输出推理任务中，重头标记会发生偏移。因此，本文提出了SCOPE，一个简单而高效的框架，分别在预填充和解码阶段进行KV缓存优化。具体而言，预填充阶段的KV缓存被保留以维持关键信息，而提出了一种基于滑动窗口的新策略来选择解码阶段的重头标记。通过自适应和非连续策略进一步优化了内存使用和内存传输。在LongGenBench上的大量实验表明，SCOPE的有效性和泛化能力，以及作为插件兼容其他仅预填充KV压缩方法的优势。
- 方法：预填充阶段保留完整KV缓存以维持关键信息；解码阶段引入滑动窗口策略选择重头标记
- 结果：在LongGenBench上验证了SCOPE的有效性和泛化能力；可作为插件兼容其他仅预填充的KV压缩方法
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2025-2025.acl-long.529-bdac307fb9) · [原文](https://aclanthology.org/2025.acl-long.529/)

#### Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference

![Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](/data/wjr/vpwem/code_vpwem/literature/vpwem/vpwem_2025_index/assets/covers/acl2025-2025.acl-long.127-eb15724e2f.webp)

- 导读：编码器-only的Transformer模型（如BERT）在检索和分类任务中相比于较大的解码器-only模型提供了更好的性能-大小权衡。尽管它们是许多生产流水线的主力，但自BERT发布以来，其帕累托改进有限。本文介绍了ModernBERT，将现代模型优化引入编码器-only模型，代表了比旧编码器更显著的帕累托改进。在2万亿token上训练，原生支持8192序列长度，ModernBERT在涵盖多种分类任务以及单/多向量检索（包括代码）的大量评估中表现出了最先进的结果。除了强大的下游性能外，ModernBERT还是速度最快、内存效率最高的编码器，并针对常见GPU上的推理进行了设计。
- 方法：将现代模型优化应用于编码器-only架构；在2万亿token上训练，原生支持8192序列长度
- 结果：在分类和检索（包括代码）任务中达到最先进水平；是速度最快、内存效率最高的编码器
- 对 VPWEM：邻近工作：用于补充基线、Related Work 或实验边界。
- 链接：[导读](https://connectedresearchers.top/guide/paper/acl2025-2025.acl-long.127-eb15724e2f) · [原文](https://aclanthology.org/2025.acl-long.127/)

## EACL 2025

本轮收录 0 篇。

本轮未发现明确相关论文。

## WSDM 2025

本轮收录 0 篇。

本轮未发现明确相关论文。
