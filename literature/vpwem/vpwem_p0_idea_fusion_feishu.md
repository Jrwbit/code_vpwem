# VPWEM P0 论文精读、代码审计与投稿方案（飞书版）

更新时间：2026-08-21

## 一、结论先行

本轮共复核 2024-2026 年 VPWEM P0 候选论文 26 篇。其中：

- 13 篇保留为 P0：会直接影响 VPWEM 的新颖性、强基线或核心实验设计；
- 7 篇降为 P1：机制邻近或可迁移，但任务、记忆对象或控制层级不同；
- 6 篇降为 P2/排除：属于关键词误检，不构成直接竞争。

原 VPWEM 与新增 PER 的逐项追溯总表见 `docs/vpwem_claim_code_experiment_audit_2026-08-09.md`。该表把论文主张、公式、代码位置、实际调用链、配置、checkpoint 和原始结果分开判定；本文件只保留可用于写作的结论。

当前正式 VPWEM 配置的实际机制是：训练时从当前时刻之前的单回合历史中分段随机采样，并截断到有限长度；Q-Former 再将这些历史特征压缩为固定数量的 memory tokens，与最近观测一起条件化 Diffusion Policy。在线推理时，历史特征按固定间隔进入有界缓存，再由 Q-Former 生成固定数量的 memory tokens。

但在 MemoryVLA、HAMLET、MemER、RoboMME 等工作出现后，下一篇论文不能继续只强调“工作记忆 + 情景记忆”“固定容量”或“长时历史压缩”。以下是待实现和待验证的目标主线，不是对当前代码效果的结论：

> 将 VPWEM 从“无差别压缩所有越窗历史”升级为“通过预测误差发现关键事件，在固定预算内选择性写入，并能根据新证据修正旧状态的任务记忆”。

建议方法暂名：

> **PE-VPWEM: Predictive Event-Gated Memory for Bounded Non-Markovian Visuomotor Control**

这个名称目前只表示一个候选实现，不等于已经成立的创新点。研究中必须固定的是下述问题和验收标准，而不是固定某个模块名称：

> **在当前观测不足以决定动作的任务中，能否在固定存储和计算预算下，形成对动作充分、对轨迹身份不敏感、并能在新证据到来后修正的任务状态？**

在这个目标下，Q-Former、分段采样、事件写入、预测误差和状态修正都只是可替换的实现假设。后续若替换某个模块，必须说明它解决了哪一个已测出的失败模式；不得随着单次实验成败更换研究问题和论文叙事。

## 一-A、三个根本问题与修正后的研究逻辑

### 1. DP 的本质是什么，分段记忆为什么可能提高成功率

这里的 DP 是 Diffusion Policy，不是 Dynamic Programming。它的本质是一个以观测条件为输入的动作序列条件生成模型：训练时给示范动作块加入不同强度的噪声，学习在条件 `c_t` 下预测噪声或还原动作；推理时从随机动作序列出发反复去噪，得到未来动作块，并以滚动时域方式执行其中前几步。

```text
action chunk A_t = (a_t, ..., a_(t+H-1))
condition c_t = recent observations W_t + compressed history E_t

train:    minimize mean((eps_theta(noisy(A_t), noise_level, c_t) - eps)^2)
test:     random action chunk -> iterative denoising conditioned on c_t -> execute first h actions
```

因此 DP 本身不是记忆模块、世界模型或显式规划器。它学习的是 `p(A_t | c_t)`。在小数据视觉操作中，它还可能表现为根据相似视觉条件召回训练动作块。给 DP 增加记忆，只是改变条件 `c_t`；不能据此声称模型已经学会任务状态或长期规划。

长期记忆有必要的前提不是 episode 很长，而是存在两段历史 `H` 和 `H'`：它们产生几乎相同的近期观测 `W_t`，但正确动作不同。等价地，旧历史在给定近期观测后仍携带动作信息：

```text
I(A_t; H_old | W_t) > 0
```

如果任务近似 Markov，上式接近 0，增加历史没有必然收益，反而可能引入轨迹泄漏和错误惯性。

当前代码中的“分段记忆”更准确地说是**分层时间采样**：把当前时刻之前的历史按 `r` 帧划段，每段随机取一帧；超过上限后保留最早的一个采样帧和最近若干采样帧。只有启用 `preserve_episode_anchor` 时，最早采样帧才被强制替换为 episode 首帧。它可能提高成功率的机制只有以下几个待检验解释：

1. **时间覆盖**：同样读取 `K` 帧时，分层采样减少样本全部聚集在某个局部时间段的风险；
2. **时间增强**：每次从段内随机取帧，使压缩器对关键事件的小幅时间偏移更稳健；
3. **有界计算**：用稀疏历史扩大可见时间范围，而不把全部帧送入注意力；
4. **首帧捷径**：若实验启用 `preserve_episode_anchor`，且任务答案恰好只出现在首帧，提升可能来自这个人工先验，而不是学习到更好的记忆；
5. **额外容量或优化差异**：更大的条件网络、不同训练噪声和配置漂移也可能造成表面提升。

所以不能直接写“分段记忆提高成功率”。严谨表述应是：

> 分层时间采样在固定输入帧数下扩大了历史覆盖；它是否因此提高成功率，以及收益是否来自关键事件保留，必须通过等预算对照和记忆干预验证。

至少需要以下对照来区分上述解释：

| 对照 | 固定项 | 检验对象 |
| --- | --- | --- |
| recent-only | backbone、训练量 | 任务是否真的需要越窗历史 |
| uniform random `K` frames | 帧数、历史范围 | 分段约束是否优于普通随机采样 |
| periodic/segment-center `K` frames | 帧数、历史范围 | 收益是否仅来自段内随机增强 |
| anchor on/off，且随机化 cue 位置 | 其他全部配置 | 是否依赖首帧捷径 |
| zero/wrong-history intervention | 当前观测不变 | 策略是否因果使用匹配的历史 |
| shuffled trajectory ID probe | memory budget 不变 | memory 保存的是任务状态还是轨迹身份 |

### 2. 创新点从哪里来，为什么不能直接复现已有论文

创新点不能从“读了很多论文后挑一个没人用过的模块”产生，而应从一个稳定问题、已有方法的共同失败边界和可证伪假设中产生。对本课题，文献矩阵应始终围绕五个问题组织：

```text
What:   应保存什么任务变量？
When:   什么时候写入？
Update: 新证据到来时怎样修正旧状态？
Read:   当前动作如何读取记忆？
Proof:  怎样证明记住的是任务状态而非轨迹捷径？
```

大量论文不能原样复制，可能因为动作层级、视觉 backbone、训练数据、记忆预算、是否依赖 VLM/对象标注或代码开放程度不同。但这些差异不是不复现的理由。能够隔离核心机制的方法，必须在统一 backbone、统一历史可见范围和统一 token 预算下直接实现为基线：

- uniform/FrameSamp：最简单的稀疏历史基线；
- HAMLET-style moment pooling：紧凑的逐时刻表征与轻量聚合；
- feature change-point/keyframe：不学习或弱学习的关键帧选择；
- recurrent state/xLSTM：固定大小递归状态；
- adjacent-similarity update：MemoryVLA 类的相似项更新；
- nearest-observation action lookup：检验 DP 是否主要在召回训练动作块。

对于依赖大型 VLM、语言推理、对象分割或外部 planner 的完整系统，不应假装可以严格同配置复现；应复现其最小机制、报告无法对齐的变量，并在原论文设置上引用其公开结果。若强简单基线已经解决问题，就没有必要再提出复杂模块。

由现有文献和代码审计能够得到的只是一个**候选缺口**：原 VPWEM 无差别压缩采样历史，FrameSamp/HAMLET 强调覆盖或聚合，关键帧方法强调选择，但“固定预算下形成可随新证据修正的动作充分任务状态”尚未被当前实现证明。对应的候选机制可以分三步验证：

```text
H1 Coverage:  稀疏覆盖历史是否足以恢复隐藏任务变量？
H2 Selection: 在同样预算下，选择关键事件是否优于均匀覆盖？
H3 Revision:  隐藏状态改变后，显式 update/reset 是否优于保留旧事件？
```

三步的最终目标和评价指标不变。只有 H1 成立才研究 H2，只有 stale-memory 测试显示旧状态污染才进入 H3。这样，即使中间方案从 delta 换成 prediction error，或者从 overwrite 换成 update/reset，改变的也是实现假设，不是研究故事。

### 3. 当前创新点是否有效

目前不能回答“有效”。现有证据只能说明 PER-delta 在 Transport/mh 的三个训练 seed 上相对 FIFO 观察到平均 `+2.89 pp`：一个 seed 下降，配对差值的 95% t 区间约为 `[-6.34, 12.12] pp`，跨过 0。并且这组实验没有正式 wrong-history 干预，没有跨任务结果，也没有验证 predictive score、显式 evidence token、长度外推或状态修正。

更关键的是，当前 PER-delta 的核心行为是“按特征变化选帧并维护有界槽位”，它本身接近关键帧/变化点记忆，尚不足以和已有工作拉开方法差异。predictor 接口存在但正式配置没有训练预测辅助项；相似槽更新只是覆盖；冲突检测与 `reset` 尚未实现。因此，现阶段不能把 PE-VPWEM 当作已经形成且有效的创新点。

以后不再以单次 success 的涨跌决定叙事，而采用固定的证据闸门：

| 闸门 | 必须回答的问题 | 失败后的动作 |
| --- | --- | --- |
| G0 任务必要性 | recent-only 是否在正对照失败、在 Markov 负对照足够？ | 若否，停止做长期记忆 |
| G1 记忆因果性 | zero/wrong history 是否显著改变动作和成功率？ | 若否，先解决策略忽略记忆，不换写入故事 |
| G2 选择有效性 | 是否超过等预算 FrameSamp、uniform、HAMLET-style 和 recurrent baseline？ | 若否，接受简单基线，不包装 event gate |
| G3 状态表征 | 是否能 probe 隐藏任务变量，同时不能轻易预测 trajectory ID？ | 若否，处理泄漏和伪相关 |
| G4 修正能力 | 状态中途改变后是否更快恢复，且不增加错误惯性？ | 若否，才引入 update/reset 机制 |
| G5 稳健结论 | 多任务、多训练 seed、冻结配置下效应区间是否超过预注册最小提升？ | 若否，只报告边界或负结果 |

正式实验前应冻结：主任务、强基线、memory/token/参数预算、训练步数、训练 seed、评估 episode、主要指标、最小有意义效应和失败条件。任何新模块只能由某个闸门的失败诊断触发，并必须回到同一套闸门重新验证。

### 当前代码审计结论

以下结论以 2026-08-09 的仓库代码和配置为准。状态中的“已实现”只表示存在调用链，不代表已有论文级实验结果。

| 主张 | 代码状态 | 论文可否声称 |
| --- | --- | --- |
| 原 VPWEM 的固定长度 Q-Former memory tokens | 已实现，正式配置正在使用 | 可以，但结果只能归于原 VPWEM |
| PER 有界事件库和固定 token 输出 | 已实现；Transport/mh 有三训练 seed 评估 | 可以写成“已实现并有单任务初步结果” |
| 特征变化分数 delta | 训练和在线推理均已实现 | 可以 |
| 单步特征预测误差 | 已实现 predictor 和辅助损失接口 | 只能写成可训练组件；正式配置尚未启用 |
| skip、相似槽更新、低分槽替换 | 在线 PER 已实现 | 可以；代码中的“merge”实际是覆盖相似槽，不是加权平均 |
| novelty 独立打分项 | 未实现 | 不可以 |
| action influence 写入分数 | 未实现 | 不可以 |
| 稀疏写入损失和状态 probe 损失 | 未实现 | 不可以 |
| 因果记忆干预 | 已实现 zero、batch shift 和 token reverse | 只能写成评估工具；当前没有正式干预结果 |
| PE-VPWEM 超过现有基线 | Transport/mh 三 seed 平均高 2.89 个百分点，但 seed 100 低于 FIFO | 只能报告该任务结果，不可概括为普遍优越 |

用于原 VPWEM 论文核心结果的 Robomimic、MIKASA 和 MoMaRT long-memory YAML 均设置为 `mem_compress_method: qformer`；RMBench 扩展配置使用的是 `random`。PER 没有独立提交的正式 YAML，但已有通过 Hydra 命令覆盖生成的配置快照、微调 checkpoint 和 Transport/mh 评估文件。原 VPWEM 结果仍不得改标为 PER；下面单列的 PER 结果可以按其真实设置报告。

### 代码证据链

| 论文对象 | 定义位置 | 实际调用位置 | 审计结论 |
| --- | --- | --- | --- |
| 历史采样 | cleandiffuser/dataset/dataset_utils.py | LongMem 数据集 | 分段随机取一帧，并截断到 maximum memory length |
| 原 VPWEM 压缩器 | cleandiffuser/nn_memory/qformer_memory.py | longmemdpptp pipelines | 正式 YAML 使用 qformer |
| PER 事件库 | cleandiffuser/nn_memory/per_memory.py | 训练、评估 pipeline 和 artifacts/per_vpwem | 已接入并产生 Transport/mh 结果 |
| 记忆训练损失 | cleandiffuser/diffusion/longmem_ddpm.py | agent update | predictor 和反事实项受系数开关控制 |
| 记忆条件化动作模型 | cleandiffuser/nn_diffusion/longmem_chitransformerptp.py | LongMemDDPM | memory tokens 与近期观测共同进入 decoder memory |
| 记忆干预 | cleandiffuser/diffusion/longmem_ddpm.py | Robomimic embedding eval | 已接入评估入口，尚无正式对比结果 |

### 投稿前必须消除的实现风险

| 风险 | 当前证据 | 处理要求 |
| --- | --- | --- |
| 训练数据不是完整历史 | sampler 默认最多保留 16 个历史样本 | 不得声称训练时压缩任意长完整历史 |
| memory_type 名称可能误导 | sampler 保存该参数，但当前采样逻辑没有按类型分支 | 若要比较 sliding window，需先实现并测试 |
| hybrid 训练与推理不一致 | 离线使用均值归一化，在线直接相加 | 主实验前统一公式和代码 |
| merge 命名过强 | 在线代码直接覆盖相似非锚点槽 | 论文改称 similar-slot update，或实现真正融合 |
| predictor 可能未受训练 | 正式 YAML 未启用 memory_aux_coeff | predictive 实验必须显式设置并记录系数 |
| 因果损失可能未启用 | 正式 YAML 未启用 counterfactual_memory_coeff | 不得把现有结果归因于反事实训练 |
| 论文配置与产物漂移 | 论文为 8-layer diffusion、2-layer compressor；发布版 pipeline 曾硬编码 4-layer，多数本地 Robomimic checkpoint 也是 4-layer | 旧结果只能标为本地变体；严格复现必须冻结配置并核验 checkpoint 结构 |
| episode 边界可能污染 | sampler 在新 episode 首样本处可能错误引用上一 episode | 修复索引并增加边界测试后重跑关键实验 |
| 单元测试不能替代实验结果 | 本机 vpwem5090 环境全量测试 21 passed | 保留测试命令与环境版本；性能主张仍需正式评估 |

### 结果引用口径

| 标签 | 含义 | 写作要求 |
| --- | --- | --- |
| 原论文报告 | 数值来自既有论文 | 必须写明论文和设置，不得称为本地结果 |
| 本地复现 | 当前仓库产生的有效结果 | 绑定配置、checkpoint、seed 和评估回合数 |
| 失败或回归 | 流程跑完但结果不可信 | 保留并明确标红，不计入方法平均值 |
| 计划实验 | 尚无结果 | 只写假设和验收条件，不填占位数值 |

任何主结果表中的一行都应能反查到配置文件、模型 checkpoint 和原始评估文件。缺少其中任一项时，只能进入附录中的待核验区。

### 当前可追溯的 PER 初步结果

任务为 Robomimic Transport/mh。每个训练 seed 使用各自的 10k checkpoint 文件进行 300 episodes 评估；FIFO 和 PER 使用相同的评估 seed `20260806`。模型为 4-layer diffusion、2-layer Q-Former、2 个输出 memory tokens。PER 使用 delta 分数、8 槽事件库、`evidence_length = 0`，因此本组实验只验证事件库加 belief tokens，不验证显式 evidence token。

| 训练 seed | FIFO success | PER success | PER - FIFO |
| --- | --- | --- | --- |
| 100 | 40.67% | 39.33% | -1.33 pp |
| 200 | 37.67% | 42.00% | +4.33 pp |
| 300 | 40.67% | 46.33% | +5.67 pp |
| 三 seed 均值 | 39.67% | 42.56% | +2.89 pp |

三 seed 间 success 的样本标准差分别为 FIFO `1.73 pp`、PER `3.53 pp`。平均 reward 从 `7.19` 增至 `8.92`，平均单次推理时间从 `0.0810 s` 增至 `0.0860 s`，约增加 `6.10%`。

以三个训练 seed 的配对差值为统计单位，PER - FIFO 的样本标准差为 `3.72 pp`，95% t 区间约为 `[-6.34, 12.12] pp`，跨过 0。因此只能写“观察到 `+2.89 pp` 的平均差”，不能写“显著提升”或“稳定优于 FIFO”。

证据文件位于 `artifacts/per_vpwem/formal_eval/train{100,200,300}_{fifo,per}/`。限制是：目前只有一个任务、三个训练 seed、同一个评估 seed，而且正式 300-episode 矩阵只运行了 `memory_interventions = [none]`。seed 200 另有正常 46.67%、zero 40.00%、shuffle_batch 10.00% 的 30-episode 诊断，但它不是三 seed 正式因果实验。因此现有证据不能支撑跨任务优势、正式因果结论或长度外推结论。

### 飞书排版约定

- 公式统一放在纯文本代码块中，不使用美元符号、LaTeX 环境或 HTML 标签。
- 公式变量只使用 ASCII 字符；下标写成 `z_t`，均值写成 `mean(...)`。
- 表格保持短列、单行单义；长公式不放入表格。
- 百分比统一写成 `94.44% +/- 1.92%`，避免复制链路替换特殊数学符号。

## 二、26 篇候选论文资格审计

| 年份 | 论文简称 | 复核级别 | 对 VPWEM 的意义 |
| --- | --- | --- | --- |
| 2024 | Embodied Active Defense | P2 | 对抗贴片防御，不研究任务历史记忆 |
| 2024 | READ-PVLA | P2 | PVLA 指视频-语言对齐，不是机器人策略 |
| 2024 | Offline Risk-sensitive RL | P1 | POMDP 邻近工作，但不是视觉运动历史压缩 |
| 2024 | Social Robot Memory Clinic | 排除 | Memory Clinic 是医院科室，属于误检 |
| 2025 | LRAM/xLSTM | P1 | 可作为常数状态 recurrent baseline |
| 2025 | SAM2Act/SAM2Act+ | P1 | 强空间记忆路线，但依赖 SAM2 memory bank 和 planner |
| 2025 | 3D-Mem | P1 | 3D 探索和导航记忆，不直接生成低层操作动作 |
| 2025 | Spatial Feature Memory | P2 | 服务具身检测，不是动作策略记忆 |
| 2026 | ALT | P0-D | 直接质疑 Diffusion Policy 是否只在做动作查表 |
| 2026 | Personalization Memory | P1 | 揭示多记忆干扰和 coordination failure |
| 2026 | HAMLET | P0-A | 最重要的轻量、紧凑历史基线 |
| 2026 | MIKASA | P0-B | 证明任务是否真的需要记忆的核心 benchmark |
| 2026 | MemoryVLA | P0-A | 双记忆、固定容量和 diffusion expert 的正面竞争者 |
| 2026 | MemER | P0-A | 强关键帧检索路线，挑战无差别 latent compression |
| 2026 | HiMe | P0-A | 说明记忆需要 Add/Update/Delete 和状态修正 |
| 2026 | RoboMME | P0-B | 统一比较多种记忆表示，FrameSamp 是强简单基线 |
| 2026 | SOMA | P1 | 视野外对象的持久空间地图路线 |
| 2026 | AFI/Memory Traps | P0-D | 说明记忆可能固化错误轨迹、妨碍纠错 |
| 2026 | CycleManip | P0-T | 周期操作中的历史采样和进度监督 |
| 2026 | OptimusVLA | P0-A | 跨轨迹动作先验和近期动作块的双记忆路线 |
| 2026 | SemanticVLA | P0-D | 区分语义状态推理与动作记忆 shortcut |
| 2026 | TRM-VLA | P0-A | 缓存关键帧视觉推理特征用于动作生成 |
| 2026 | RGMP | P1 | 几何 recurrent prior，主要服务人形操作泛化 |
| 2026 | Embodied-SlotSSM | P0-A/B | 对象级 transient memory 和 LIBERO-Mem benchmark |
| 2026 | Event-based Memory Policy | P2 | 无人机事件流低延迟状态，问题和模态差异较大 |
| 2026 | Conflict-Aware Memory | P2 | LLM planner 的知识库清洗，不是单回合视觉运动记忆 |

## 三、最重要的竞争工作

### 1. MemoryVLA：最危险的正面竞争者

MemoryVLA 使用 working memory、固定容量的感知-认知记忆库和 diffusion action expert。其 MIKASA-Robo 平均成功率为 41.2%，双流记忆明显优于单独的 perceptual 或 cognitive memory。

它意味着 VPWEM 不能再把以下内容单独作为主创新：

- working memory + long-term memory；
- 固定容量记忆；
- 记忆条件化 diffusion action expert。

VPWEM 应改为强调：低层策略、单回合 observation memory、固定 token 预算、长度外推，以及可验证的任务状态保真。

### 2. HAMLET：最应复现的轻量基线

HAMLET 用少量 moment tokens 压缩每个时刻，再通过两层 Transformer 聚合历史。在真实历史依赖任务上将 GR00T N1.5 从 29.2% 提升到 76.4%，且成本显著低于多帧拼接。

VPWEM 必须增加等 token、等 backbone 的 HAMLET-style baseline。真正优势应体现在关键事件已经超出 HAMLET 窗口，以及 test horizon 远大于 train horizon 时。

### 3. MemER：关键帧检索路线

MemER 使用高层 VLM 选择关键帧，并用关键帧历史驱动低层策略。三个真实长时任务平均进度达到 96%，显著高于固定 32 帧历史的 62%。

这说明“选对关键帧”可能比“压缩全部历史”更有效。VPWEM 至少需要比较 uniform frame、change-point、similarity keyframe 和 learned keyframe。

### 4. RoboMME：没有一种记忆表示通吃

RoboMME 在 16 个任务上发现：

- FrameSamp + Modulator 总体最强，平均 44.51%；
- 符号记忆更适合计数和事件显著任务；
- 视觉记忆更适合运动、时间和长视频任务；
- recurrent memory 反而没有明显优势。

因此不能只报一个平均成功率。VPWEM 应按 counting、object、spatial、motion、procedural 等类别报告，并将 FrameSamp 作为最低门槛。

### 5. ALT、AFI、SemanticVLA：三类反证

- ALT：Diffusion Policy 可能通过最近邻定位训练轨迹并召回 action chunk；
- AFI：更强的历史惯性可能让策略陷入错误训练轨迹；
- SemanticVLA：训练内成功不等于语义状态推理，可能只是动作记忆。

因此成功率不能证明 episodic token 保存了任务状态。必须加入轨迹泄漏 probe、错误历史替换、扰动恢复和 OOD 测试。

## 四、与当前代码一致的方法定义

### 1. 研究问题

原 VPWEM 在线推理时按固定间隔接收历史特征，并使用 FIFO 或相邻相似压缩维持有限缓存。PER 的目标是在相同有限预算下，按事件分数决定是否写入，并在缓存满时更新或替换槽位。

新的研究问题是：

> 能否在固定 token 和缓存预算下，依据可计算的特征变化或单步预测误差选择性保留事件，并让策略对正确历史比错误历史产生更低的动作去噪损失？

### 2. 方法结构

```text
historical feature z_t
    |
delta score or one-step prediction error
    |
threshold check or maximum write interval
    |
skip / update similar slot / allocate / replace low-score slot
    |
bounded event bank with K_bank slots
    |
E evidence tokens + (M - E) Q-Former belief tokens
    |
recent observations + M memory tokens
    |
Diffusion Policy action chunk
```

当前在线 PER 的四种操作如下：

- `skip`：分数低于阈值且未达到强制写入间隔时，不更新事件库；
- `update`：若新特征与已有非锚点槽的余弦相似度达到阈值，直接覆盖该槽；
- `allocate`：事件库未满时写入新槽；
- `replace`：事件库已满时替换衰减后分数最低的非锚点槽。

槽位 0 保存回合首个特征，代码不会覆盖其内容。当前实现没有根据显式“状态冲突”标签执行替换，也没有对相似槽做加权平均，因此论文中不应使用“冲突感知修正”或“特征融合式 merge”作为既成事实。

### 3. 写入分数

设历史特征为 `z_t`，预测器为 `g_phi`。代码中的变化分数为：

```text
delta_t = 1 - cosine(z_t, z_(t-1))
```

单步预测误差为：

```text
pred_t = mean((g_phi(z_(t-1)) - z_t)^2)
```

离线训练前向中的事件分数与代码一致地写为：

```text
score_t = delta_t                                      if mode = delta
score_t = pred_t                                       if mode = predictive
score_t = 0.5 * (delta_t / mean(delta) +
                 pred_t  / mean(pred))                 if mode = hybrid
```

首个历史特征的分数设为正无穷，以确保回合锚点被保留；其余位置按分数取 top-K，再恢复时间顺序。

在线推理中的 `hybrid` 当前使用下式：

```text
score_t_online = delta_t + pred_t
```

这与离线归一化公式并不完全一致。修复该训练与推理差异之前，主实验应优先使用 `delta` 或 `predictive` 单一模式，不能声称 hybrid 使用了统一打分函数。

在线写入条件为：

```text
write_t = (score_t >= threshold)
          OR (t - last_write_t >= max_write_interval)
          OR (event_bank_is_empty)
```

相似更新和满库替换规则为：

```text
similarity_i = cosine(bank_i, z_t)

if max(similarity_i) >= merge_threshold:
    overwrite the most similar non-anchor slot
elif event bank is not full:
    allocate a new slot
else:
    replace the lowest decayed-score non-anchor slot
```

当前代码没有独立的 novelty、action influence 或 memory redundancy 加权项。相邻特征变化由 `delta_t` 近似，相似性只用于槽位更新判断。

### 4. 训练目标

在核心配置 `pred_reg_coeff = 0` 的前提下，LongMemDDPM 已接入的总损失为：

```text
L_total = L_action
        + lambda_pred * L_pred
        + lambda_cf   * L_cf
```

其中预测辅助损失为：

```text
L_pred = mean_t(pred_t)
```

批内错误记忆由正确压缩记忆沿 batch 维循环移动一位得到。反事实间隔损失为：

```text
L_cf = max(0, margin + L_action_correct - L_action_wrong)
```

该损失鼓励正确记忆对应的动作去噪损失至少比错误记忆低一个 `margin`。只有当 `memory_aux_coeff` 和 `counterfactual_memory_coeff` 大于 0 时，两项才会加入训练。当前正式 YAML 未设置这些系数，因此现有训练结果只对应 `L_action`。

以下目标目前不在代码中，不得写入已实现的方法公式：

- 写入率或稀疏性损失；
- 任务状态 probe 损失；
- action influence 写入得分；
- 完整历史 teacher 蒸馏损失。

### 5. 暂时不建议加入的模块

- 不先加入 LLM planner 或文本 memory；
- 不同时实现四类 typed slots；
- 不直接引入复杂对象检测和场景图；
- 不把参数量增长包装成 memory 提升。

这些模块会扩大工程量并模糊主贡献。只有当实验明确显示对象身份混淆是主要瓶颈时，才升级为 object slots 和 residual slots。

## 五、因果记忆评测

当前代码已经支持以下三种干预，并保留无干预对照：

| 配置值 | 实际操作 | 可检验问题 |
| --- | --- | --- |
| none | 保留压缩记忆 | 标准性能 |
| zero | 将压缩记忆全部置零 | 策略是否需要记忆内容 |
| shuffle_batch | 将 batch 内记忆循环移动一位 | 策略是否需要样本匹配的记忆 |
| reverse_tokens | 反转压缩 token 顺序 | 策略是否依赖 token 排列 |

注意：`reverse_tokens` 反转的是压缩后的 token，而 Q-Former belief token 不等同于按时间排列的原始帧。因此它不能直接命名为“时间打乱”。

以下干预尚未实现，应保留在实验计划中：

| 计划条件 | 目的 |
| --- | --- |
| 同当前帧、不同隐藏状态的配对历史 | 测是否依赖隐藏任务状态 |
| 只保留关键事件 | 测事件充分性 |
| 只保留最近窗口 | 测长期历史贡献 |
| 环境中途修改对象或目标状态 | 测过期记忆恢复 |

建议指标如下；在结果文件产生之前，它们都只能写成评测定义：

- **Memory Necessity**：正确 memory 换成空 memory 后的性能下降；
- **Memory Specificity**：正确 memory 换成错误 memory 后的性能下降；
- **State Sufficiency**：当前视觉受损时，正确 memory 是否仍能支持决策；
- **Trajectory Leakage**：memory token 预测 trajectory ID 的准确率；
- **Stale-State Recovery**：环境状态被修改后恢复正确行为所需步数。

## 六、必须补齐的强基线

- HAMLET-style moment tokens；
- FrameSamp + Modulator；
- uniform frame sampling；
- feature change-point/keyframe；
- xLSTM recurrent state；
- MemoryVLA-style adjacent similarity merge；
- observation nearest-neighbor action chunk，作为 ALT-style lookup baseline。

公平性要求：

- 相同视觉 backbone；
- 相同 memory token 数；
- 相同历史可见范围；
- 相同训练步数和数据；
- 报告参数量、推理延迟、峰值显存和 FLOPs。

## 七、核心实验矩阵

### 1. 长度外推

```text
train horizon: 32 / 64
test horizon: 1x / 2x / 4x / 8x
关键 cue: 固定出现在 episode 前 10%
memory budget: 所有方法统一为 K tokens
```

主图：

- success vs. test horizon；
- success vs. memory budget；
- latency/peak memory vs. episode length；
- write rate vs. success。

### 2. 任务组合

| 数据集/任务 | 研究目的 |
| --- | --- |
| MIKASA RememberColor3/5/9 | memory capacity scaling |
| MIKASA ShellGameTouch | 遮挡后的对象身份保持 |
| RoboMME counting task | 事件和计数记忆 |
| RoboMME motion task | 运动和时序历史 |
| Robomimic Square | 不需要长期记忆的负对照 |
| LongSquare | 控制记忆长度和 cue delay |
| 中途移动物体/错误动作 | 过期记忆修正与恢复 |

### 3. Event Gate 消融

当前代码可直接形成的第一组消融：

- Q-Former 原始固定间隔写入；
- PER feature delta；
- PER predictive error；
- PER hybrid，但必须先统一训练与在线公式；
- PER delta 加反事实间隔损失；
- PER predictive 加反事实间隔损失。

以下消融尚需实现，不能提前出现在结果表中：

- RGB difference；
- learned event gate；
- 独立 novelty 项；
- action influence 项；
- 稀疏写入正则。

所有方法应固定事件库容量、输出 token 数和平均候选采样频率。若比较不同写入阈值，还应同时报告实际写入率，避免某个方法因保存更多内容而占优。

## 八、论文叙事

建议标题：

> **PE-VPWEM: Predictive Event-Gated Memory for Bounded Non-Markovian Visuomotor Control**

只有完成跨任务 PER 训练、强基线和因果干预后，摘要才可以使用以下目标主张：

> PE-VPWEM 使用特征变化或单步预测误差在有界事件库中选择历史证据，并将证据 token 与 Q-Former belief token 一起条件化 Diffusion Policy。因果干预用于检验策略是否使用样本匹配的记忆。

投稿贡献必须按证据成熟度分层。当前可以报告实现和严格限定的单任务结果：

1. 实现有界事件库，以及 feature delta 和单步预测误差两类写入分数；
2. 实现 zero、batch-shift、token-reverse 三类压缩记忆干预；
3. 在 Transport/mh 的当前设置下，PER-delta 三训练 seed 平均 success 为 42.56%，FIFO 为 39.67%；该结果不代表跨任务优势。

长度外推、Markovian 不退化和因果使用均为待验证结论，不能使用无数据限定的“全面超过”。

## 九、执行顺序

### 第一阶段：先验证现有 VPWEM

1. 在正式配置中启用并跑完 memory intervention harness；
2. 加入 FrameSamp 和 nearest-action lookup baseline；
3. 跑 RememberColor3/5/9 和 ShellGame 的 horizon scaling；
4. 输出 success、latency、memory、FLOPs 四类结果。

这一阶段决定现有 memory 是否真的保存任务状态。如果正确/错误历史互换几乎不影响动作，说明策略没有使用 episodic memory；如果 trajectory-ID probe 很高，说明存在查表风险。

### 第二阶段：验证并补齐 Event-Gated Write

1. 将现有 Hydra 覆盖固化为可版本控制的 PER 训练和评估 YAML；
2. 先跑已实现的 feature delta，再启用 predictor 辅助损失；
3. 统一 hybrid 的离线与在线分数公式；
4. 比较固定预算下的事件 recall 和成功率；
5. 检查是否改善 4x/8x horizon 外推；
6. 只有基础版本有效后，再考虑 learned gate 和稀疏写入正则。

### 第三阶段：根据失败类型决定升级方向

- 若主要问题是旧状态无法纠正：加入 `keep/update/reset` contradiction gate；
- 若主要问题是对象身份混淆：加入 object slots + residual slots；
- 若训练内成功但 OOD 明显下降：加入 trajectory-ID adversary 和 history mixup；
- 若 FrameSamp 始终更强：重新审视 latent compression 是否有必要，不继续堆模块。

## 十、最终研究判断

最务实、最有发表胜率的路线不是继续扩大 VPWEM，而是建立以下完整证据链：

```text
[待验证] 非马尔可夫任务确实需要历史
    -> [工具已实现，结果缺失] memory 被策略因果使用
    -> [probe 未实现] memory 保存任务状态而非轨迹 ID
    -> [单任务初步证据] Transport/mh 上 PER-delta 平均高 2.89 pp
    -> [实验缺失] 固定预算下可以外推到更长 episode
    -> [机制未实现] 状态变化后旧记忆可以被显式修正
```

一句话总结：

> **先用干预证明 VPWEM 确实使用历史，再分别验证 feature delta 和预测误差写入；没有结果支持前，不把“状态记忆”和“长度外推”写成结论。**

## 十一、本地材料索引

- VPWEM 原论文：`literature/vpwem/vpwem.pdf`
- 26 篇 P0 全文精读：`literature/vpwem/vpwem_p0_2024_2026/deep_reading_and_ideas.md`
- P0 PDF：`literature/vpwem/vpwem_p0_2024_2026/pdfs/`
- P0 文本：`literature/vpwem/vpwem_p0_2024_2026/text/`
- 跨领域可迁移机制：`literature/vpwem/cross_domain_memory_transfer_to_vpwem.md`
- VPWEM 方法与实验工作流：`literature/vpwem/vpwem_research_workflow.md`
