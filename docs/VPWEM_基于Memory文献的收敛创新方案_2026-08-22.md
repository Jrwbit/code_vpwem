# VPWEM：基于 Memory 文献与现有实验的收敛创新方案

更新时间：2026-08-22  
文档性质：创新决策、涨点机制分析与最小实验路线  
结论状态：候选方案，尚无新增实验结果

## 0. 最终建议

下一步不再继续调 PER 分数、retention threshold、reservoir、bank size，也不直接实现一个更复杂的 belief memory。

建议把唯一主假设冻结为：

> **VPWEM 当前的第一瓶颈可能不是历史没有被存下，而是普通行为克隆没有迫使策略在“当前观测相似、历史任务状态不同”的决策点使用 memory。**

据此，建议先做一个不更换 VPWEM memory architecture 的最小创新：

> **NCM-VPWEM：Necessity-Conditioned Memory for VPWEM**

它由三个紧密相连、但分阶段验证的部分组成：

1. **Alias-set mining**：在 demonstrations 中找到完整非记忆当前条件相似、但隐藏任务分支不同的样本；
2. **Matched counterfactual diffusion training**：交换这些样本的历史，构造困难且有语义的错误 memory，训练动作生成必须因历史分支而改变；
3. **Necessity-gated read**：只在当前输入具有动作歧义时增强 memory 对 diffusion decoder 的调制，普通/近 Markov 状态保持原策略路径。

第一轮只实现 1 和 2。只有 matched counterfactual supervision 明显优于 random wrong-memory supervision，才实现第 3 项。只有前三项成立后，才考虑把 Q-Former frame summary 换成 compact causal belief。

这条路线的核心不是再回答“保存哪一帧”，而是回答：

> **什么时候 history 是当前动作的必要条件，以及如何让策略在这些时刻因果使用 history。**

---

## 1. 为什么收敛到 memory usage，而不是继续研究 retention

### 1.1 当前结果已经排除了三个直觉

| 直觉 | 当前证据 | 判定 |
| --- | --- | --- |
| 视觉变化大的帧更重要 | delta-PER 在 Transport/mh 三 seed 为 `+2.89pp`，主区间跨 0；跨任务总体近 0，ShellGameTouch 下降 | 不能继续作为主创新 |
| 离线 future-action importance 能指导在线 retention | Learned retention 离线保留约 `80.43%` oracle importance，但 rollout 只有 `39% vs FIFO 38%` | keep 不等于 use |
| 长时覆盖本身能稳定涨点 | 修正后的 Reservoir-2 相对 Recent-2：Square/mh `+1pp`，Transport/ph `-2pp`，LongSquare 均为 0 | coverage 尚未跨任务成立 |

原 Random-2 raw 相对 FIFO-QFormer 的 `+13.33pp` 同时改变 retention、raw/Q-Former representation、参数量和训练/推理分布。严格拆分后没有稳定 coverage 优势。因此不能从这个结果推出 CPB，也不能继续把 random/reservoir 包装成创新。

### 1.2 现有训练目标对 memory 的约束太弱

当前主损失是普通 diffusion behavior cloning：

```text
L_DP = mean((eps_theta(x_t, t, current_condition, memory) - eps)^2)
```

如果近期观测、机器人姿态、演示时间进度或动作惯性已经能把平均训练损失降得很低，优化器没有理由主动学习一条更困难的 long-memory 路径。即使正确 cue 已在 Q-Former 中，策略仍可能忽略它。

当前 `counterfactual_memory_loss` 使用 `batch.roll` 构造随机错误历史。它存在三个问题：

1. 随机历史通常连当前机器人姿态、任务阶段都不匹配，是过于容易的 OOD negative；
2. 模型可以识别“这段 memory 很怪”，但不必学会隐藏分支；
3. 它只要求 wrong-memory loss 变大，没有证明动作会转向错误历史对应的目标分支。

真正困难且有因果意义的 negative 应满足：

```text
current_condition_i ~= current_condition_j
hidden_branch_i != hidden_branch_j
history_i != history_j
```

只有此时，模型不能靠当前帧、关节位置、时间步或明显分布差异解题，必须从 history 恢复隐藏状态。

### 1.3 文献缺口也指向训练监督，而不是新的 memory 名词

- MemoryVLA、MemoAct 已覆盖多级/双流 memory；
- HAMLET、PAM 已覆盖紧凑历史聚合；
- MemER、TRACE 已覆盖关键证据与延迟证据；
- TFP 已覆盖 task-progress belief 和 event-sensitive update；
- MEMBOT 已覆盖显式 latent belief；
- ALT、AFI、SemanticVLA 已提示 trajectory lookup、错误惯性和动作记忆 shortcut。

因此“固定 memory”“compact belief”“event update”“evidence + belief”都不能单独形成安全的新颖性。相对明确的候选差异是：

> 不是先规定 memory 结构，而是从 demonstrations 中识别 memory necessity，并把 matched wrong history 作为动作条件的反事实干预，直接监督 policy 对 history 的因果依赖。

TRACE 明确保持原 imitation objective；NCM-VPWEM 的主要变化恰好在 training objective 和 memory read necessity，而不是再造一种 evidence store。

---

## 2. 因果链：为什么这个方案更可能在 memory benchmark 上涨点

把最终成功拆成：

```text
P(success)
  ~= P(cue captured)
   * P(hidden state decoded | captured)
   * P(policy uses state at branch | decoded)
   * P(motor succeeds | correct branch)
```

现有 PER、learned retention、reservoir 主要改变第一项；结果没有稳定转化为 success。NCM-VPWEM 直接优化第三项，并用 hidden-branch probe 检查第二项。

对 ShellGameTouch、RememberColor、SwapBlocks，这个假设具有明确的可观测预测：

1. 在普通状态上，correct 与 matched-wrong history 的动作分布应接近；
2. 在 branch decision 状态上，两者应明显分离；
3. 换入另一 branch 的 history 后，动作应系统性转向该 branch，而不是仅变得更差；
4. current-only 输入越相似、branch 越不同，matched-CF 的收益越大；
5. 在 Square、Transport、InterceptGrab 等近 Markov 控制任务上，necessity gate 应接近 0，避免 memory 干扰。

这组预测可以同时被离线 loss、branch accuracy、rollout success 和 memory intervention 验证，失败时也能定位是 pair、representation、usage 还是 motor bottleneck。

---

## 3. 方法一：Alias-set mining

### 3.1 完整 current condition

定义：

```text
c_t = 当前策略在不读取 long-term memory 时能看到的全部输入
```

在当前项目里至少包括：

- 最近 `obs_steps` 个视觉 embedding；
- 当前 joints / proprioception；
- gripper state；
- 指令或任务条件（若有）；
- 当前 policy 明确可见的其他变量。

不能只用 RGB embedding 找近邻。否则 joints、gripper 或相位差异会制造“假歧义”。也不能把 previous action 同时放进 current key 和 long history，具体边界必须按实际 policy interface 固定。

### 3.2 隐藏分支标签

第一轮在 MIKASA 使用环境已有的 `oracle_info`：

- ShellGameTouch：`cup_with_ball_number in {0,1,2}`；
- RememberColor：`true_color_indices in {0,...,C-1}`。

这些变量只用于训练数据挖掘、辅助 probe 和评测，测试策略不读取它们。

现有 `.npz` 没有保存 `oracle_info`，但数据采集是确定性的：batched collector 使用 batch seed，之后按 env slot 拆成 1000 条轨迹。因此可以按原采集 seed 和轨迹编号重建标签；更稳妥的长期做法是重新生成一个只增加 `episode_id/seed/env_slot/oracle_info` 的 manifest，不改 RGB/action 原数据。

RMBench 当前先不做 oracle mining。其 action alignment 和在线成功率仍需修复，且现有 zarr 没有可靠 hidden-branch 字段。

### 3.3 决策区间限制

只在 branch 真正开始影响动作的时间段找 pair：

- ShellGameTouch：杯子重新出现后，到 TCP 明确朝某个杯移动前；
- RememberColor：所有颜色物体出现后，到目标接触前；
- SwapBlocks：目标身份不可见、但下一操作对象开始分叉的阶段。

在 cue 展示期或纯移动期挖 pair 会让动作差异与 memory necessity 错位。

### 3.4 Pair 规则

先冻结 encoder，计算：

```text
u_i = EncodeAllNonMemoryInputs(c_i)
```

候选 pair：

```text
episode_i != episode_j
branch_i != branch_j
phase_i == phase_j
cosine(u_i, u_j) >= tau_c
proprio_distance(i, j) <= tau_p
gripper_i == gripper_j
```

使用 mutual top-k，而不是单向最近邻；对 branch 和 timestep 分层采样，避免某一目标或某一相位占满训练 batch。

必须加入两个负对照：

- `same-branch matched`：当前相似且 branch 相同，memory 替换不应产生大影响；
- `different-branch unmatched`：branch 不同但 current 不匹配，用来证明“匹配当前状态”本身有必要。

### 3.5 Pair 质量 gate

训练前必须输出：

- 每个 branch 的 pair 数和样本覆盖率；
- current embedding、joint、gripper、timestep 距离分布；
- pair 的 action 分支差异；
- 至少 100 对可视化人工审计；
- same-branch 与 different-branch 的可分性；
- episode ID / 时间步预测准确率，排查采集捷径。

如果高质量 pair 覆盖不到 decision samples 的 30%，先修 pair 定义或换任务，不训练方法。

---

## 4. 方法二：Matched counterfactual diffusion training

### 4.1 最小版：matched-history ranking

第一轮不改 Q-Former，也不改 diffusion backbone。对 pair `(i,j)`，先分别得到 `M_i` 和 `M_j`，并共享 diffusion timestep、noise 和 noisy action：

```text
ell_i_correct = L_DP(a_i | c_i, M_i; t, eps)
ell_i_wrong   = L_DP(a_i | c_i, M_j; t, eps)

ell_j_correct = L_DP(a_j | c_j, M_j; t, eps)
ell_j_wrong   = L_DP(a_j | c_j, M_i; t, eps)
```

使用对称 soft-margin：

```text
L_MH = softplus((m + ell_i_correct - ell_i_wrong) / tau)
     + softplus((m + ell_j_correct - ell_j_wrong) / tau)

L_total = L_DP + lambda_MH * w_alias * L_MH
```

`w_alias` 只在 decision interval 内启用，并可按 current-match 质量加权。相比当前随机 `batch.roll`，这里的 wrong memory 只改变隐藏分支，不明显改变当前状态分布。

第一轮必须同时比较：

| 方法 | Wrong history | 目的 |
| --- | --- | --- |
| VPWEM | 无 | 原基线 |
| Random-CF | batch roll | 判断普通 OOD wrong memory 是否有用 |
| Unmatched-Branch-CF | branch 不同但 current 不匹配 | 判断 branch label 本身是否足够 |
| Matched-CF | current 匹配、branch 不同 | 核心假设 |
| Same-Branch-CF | current 和 branch 都匹配 | 负对照 |

### 4.2 若 ranking 只拉高 wrong loss：进入 action-branch contrast

单纯 ranking 可能通过破坏 wrong-memory prediction 满足 margin，却不保证动作转向正确 branch。只有出现“离线 gap 增大、rollout branch 不变”时，才增加第二层目标。

定义多噪声层平均 diffusion energy：

```text
E(a; c, M) = mean_(t,eps) L_DP(a | c, M; t, eps)
```

对于 current 高度匹配、branch 不同的 pair，让 `M_i` 更偏好 `a_i` 而非 `a_j`，让 `M_j` 更偏好 `a_j` 而非 `a_i`：

```text
L_AB = CE([-E(a_i;c_i,M_i), -E(a_j;c_i,M_i)] / tau, target=i)
     + CE([-E(a_j;c_j,M_j), -E(a_i;c_j,M_j)] / tau, target=j)
```

它比“wrong loss 变大”更接近最终目标：memory 应改变动作分支选择。为避免演示速度和动作多模态制造假 negative，只在 branch direction 已经分离、current/proprio 高度匹配的短 decision chunk 上使用。

---

## 5. 方法三：Necessity-gated memory read

这一项只有 Matched-CF 通过后才实现。

### 5.1 Gate 的含义

从当前非记忆输入预测：

```text
g_t = P(memory is necessary for action | c_t)
```

然后只调节 memory 对 action decoder 的残差，不修改 memory 的写入：

```text
mem_emb_tilde = g_t * mem_emb
```

或更稳妥地保留一条原始残差：

```text
mem_emb_tilde = (g_min + (1-g_min)*g_t) * mem_emb
```

`g_min` 防止 gate 在训练早期把 long-memory 梯度完全关掉。

### 5.2 Gate 监督

Alias miner 给出软标签：如果 current-neighbor set 中存在多个 branch 且对应动作分布明显分裂，则 necessity 高；如果邻域只有一个 branch 或不同 history 不改变动作，则 necessity 低。

```text
y_nec(c_i) = branch_entropy(neighbors(c_i))
             * action_disagreement(neighbors(c_i))
```

这使 gate 表示“历史是否会改变动作”，而不是“图像变化是否大”。它和 TFP 的 event-sensitive write gain 不同：TFP 决定何时更新 belief；这里决定当前动作何时必须读取 history。

### 5.3 为什么 gate 有望避免普通 benchmark 退化

Robomimic Square/Transport 等任务中，当前视觉和 proprio 通常已足够。始终把历史 token 强注入 decoder 会引入 stale state、轨迹身份和动作惯性。Necessity gate 在这些状态应接近 0；在 ShellGame/RememberColor 的目标分支时接近 1。

因此它的价值不是让所有任务都大涨，而是形成预期交互：

```text
memory-intensive ambiguous states: memory gain ↑
ordinary Markov-like states: regression ~= 0
```

---

## 6. 为什么暂时不改成 compact belief

PAM、TFP、MEMBOT、TRACE 已经使“compact belief”本身非常拥挤；现有 VPWEM 的 Q-Former 也尚未被证明是主要瓶颈。

正确顺序是：

1. 固定 Q-Former，验证 matched supervision 是否让现有 memory 真正改变动作；
2. 固定 matched supervision，比较 Q-Former、GRU belief、gated slots；
3. 只有新 representation 在等 token、等 backbone 下更强或等效但更快，才保留架构贡献。

若 `VPWEM + Matched-CF` 已经最好，论文就收缩成 memory-necessity training，不强行保留 ACSM/compact belief 名称。这比一次加入 selector、belief、revision、probe 和 adversary 更容易归因，也更可能快速找到有效涨点。

---

## 7. Benchmark 选择与涨点预期

### 7.1 先做 oracle upper bound，不满足就停止

每个任务先跑四级 bottleneck ladder：

| 级别 | 条件 | 回答的问题 |
| --- | --- | --- |
| M0 | current/recent only | 无 long memory 下界 |
| M1 | oracle branch one-hot memory token | 只要知道隐藏状态，低层控制能否完成 |
| M2 | full/cue-only history | 现有 encoder 能否从视觉证据解题 |
| M3 | VPWEM correct/zero/matched-wrong | 当前 memory 是否携带并被使用 |

只有 `M1 - M0 >= 10pp` 且 M1 的 motor success 足够高，才值得做 NCM。否则成功率瓶颈不在 memory，继续训练 memory 不会稳定涨点。

### 7.2 任务优先级

| 优先级 | Benchmark | 角色 | 当前判断 |
| ---: | --- | --- | --- |
| P0 | MIKASA RememberColor5 | 主涨点任务 | 5-way hidden branch，随机猜测率低、headroom 大；先修 baseline parity |
| P0 | MIKASA ShellGameTouch | necessity/因果诊断 | GT branch 最干净，但本地 strict success 已约 91%，最终 success headroom 小 |
| P1 | MIKASA RememberColor3 | 机制复现 | 3-way alias 清晰；当前本地约 2.22% 与论文严重不一致，修复前禁做创新 |
| P1 | RememberColor9 / capacity scaling | 长度与容量外推 | 用于证明方法随 branch 数增加仍有效，不作为第一轮 smoke |
| P1 | RMBench SwapBlocks | 跨 benchmark | hidden identity 合适，但当前在线 0 success、数据无 branch 字段，先修 action/env pipeline |
| P2 | LongSquare | delayed cue 对照 | 当前 Recent/Reservoir 均 0、严格复现低；先做 oracle motor upper bound |
| Control | Square/Transport/InterceptGrab | 负控制 | 不要求涨点，只要求不退化超过预设 margin |

### 7.3 不预先承诺虚假点数

不同任务的可涨空间由 `M1-M0` 决定，不能在 oracle upper bound 前写“预计 +10pp/+20pp”。预注册时使用以下机制标准：

- BranchAcc：Matched-CF 相对 VPWEM 至少 `+10pp`；
- Correct vs Matched-Wrong success gap：至少 `15pp`，且动作方向转向 wrong branch；
- Overall Success：development 至少不低于 VPWEM `-5pp`，正式 claim 平均至少 `+5pp`；
- 至少两个 memory-intensive tasks、至少 4/5 training seeds 为正；
- Markov negative control 不下降超过 `5pp`；
- matched-CF 必须优于 random-CF 和 unmatched-branch-CF。

如果 BranchAcc 上涨但 Success 不涨，说明 motor 是剩余瓶颈；此时可以报告 memory decision 的机制结果，但不能继续通过调 memory 把它包装成 success 改进。

---

## 8. 最小实现位置

### 8.1 数据层

- 新增离线脚本：从 MIKASA 采集 seed/env slot 重建 `oracle_info`；
- 为每个 timestep 保存 `episode_id/timestep/branch/phase/current_embedding`；
- 生成 train-only matched-pair manifest；
- `LongMemMikasaDataset` 返回 paired history、paired action、alias weight；
- 验证 pair 不跨 train/validation/test split。

不应直接重写原 `.npz`；优先使用独立 manifest，避免破坏已有数据和 checkpoint 可追溯性。

### 8.2 模型与损失层

- 扩展 `LongMemDDPM.loss()`：接收显式 `matched_memory/matched_action/alias_weight`；
- correct/wrong 比较复用 `t/epsilon/x_t`；
- 支持 `random / unmatched_branch / matched_branch / same_branch` negative mode；
- 日志记录 correct loss、wrong loss、margin violation、`Delta_CF`、`R_CF`；
- Stage 2 再在 `LongMemChiTransformerPTP` 增加 necessity gate；第一阶段不改 decoder。

### 8.3 评测层

- MIKASA eval 保存每个 episode 的 `oracle_info`；
- 定义 branch-decision window 和 BranchAcc；
- 增加 Correct、Zero、Random、Matched-Wrong、Same-Branch-Matched intervention；
- 不只看 success drop，还记录首次目标方向、TCP 目标杯/颜色、wrong-branch attraction rate；
- 所有方法复用环境初态、diffusion noise 表和 pair manifest。

### 8.4 单元测试

- pair 不跨 episode/split；
- current/proprio threshold 正确执行；
- symmetric loss 两个方向都产生梯度；
- correct/wrong 共用 noise；
- same-branch negative 不被当作 hard negative；
- gate=0 等价于 no-memory，gate=1 等价于原 VPWEM；
- batch 顺序变化不改变 pair 关系；
- inference 不读取 oracle branch。

---

## 9. 最省算力的执行顺序

### Gate A：零训练诊断

1. 修复/确认 ShellGameTouch 和 RememberColor baseline；
2. 重建 oracle branch manifest；
3. 跑 M0/M1/M2，测 motor upper bound；
4. 构建并人工审计 100 对 matched pairs；
5. 计算现有 VPWEM 的 correct/random/matched-wrong offline energy gap。

停止条件：oracle branch 不能显著提高 branch choice，或 pair 覆盖不足。

### Gate B：10k adaptation smoke

任务先用 ShellGameTouch 做因果诊断，再用修复后的 RememberColor5 看 success headroom。

只比较：

```text
VPWEM
VPWEM + Random-CF
VPWEM + Unmatched-Branch-CF
VPWEM + Matched-CF
```

单 seed、100 paired episodes，只用于筛选。Matched-CF 必须在 BranchAcc、matched-wrong gap 和 overall success 三项中至少满足预注册门槛，才扩 seed。

### Gate C：加入 necessity gate

只有 Gate B 成立才加入 gate，并比较：

```text
Matched-CF
Necessity gate only
Matched-CF + necessity gate
Oracle necessity gate upper bound
```

如果 learned gate 不如 oracle gate，问题在 ambiguity label/gate learning；如果 oracle gate 也无收益，删除 gate，保留 Matched-CF。

### Gate D：representation

只有现有 Q-Former memory 已被证明被因果使用、但 hidden-state probe 或效率仍差，才比较：

```text
Q-Former frame memory
GRU causal belief
gated belief slots
```

此阶段固定 matched pairs、loss、token 数和 decoder，避免监督与架构混杂。

---

## 10. 最终论文主张的三个层级

### 当前只能写

> 现有 retention 扩展没有稳定转化为在线成功率，说明 memory availability 不等于 policy reliance；VPWEM 需要从“该存什么”转向“何时 history 必须改变动作”。

### Matched-CF 成立后可以写

> Observationally matched、hidden-branch-mismatched histories 是比随机错误历史更有效的 counterfactual training intervention；它提高 ambiguous branch decision，并使动作对正确 history 形成方向明确的因果依赖。

### Necessity gate 和多任务成立后可以写

> NCM-VPWEM 从 demonstrations 中估计 memory necessity，在 fixed-token VPWEM 上通过 matched counterfactual supervision 和 necessity-conditioned read，使 long memory 在观测别名决策点改变动作，同时避免在近 Markov 状态引入历史干扰。

禁止提前声称：

- 首次解决 state ambiguity；
- 首次提出 compact belief；
- NCM 已经稳定涨点；
- oracle branch supervision 是测试时输入；
- 所有 long-horizon task 都需要 memory；
- matched-wrong 导致性能下降就等于 memory 编码了正确状态。

---

## 11. 一句话决策

> **停止在“怎样保存更多/更关键的帧”上继续堆模块；先用 matched observational aliases 让现有 VPWEM 在真正需要 history 的动作分支上必须使用 memory，再用 necessity gate 控制何时读取。只有这条监督主线成立，才值得升级 compact belief 或 revision architecture。**

## 12. 证据入口

本地材料：

- `literature/vpwem/vpwem_p0_2024_2026/deep_reading_and_ideas.md`
- `literature/vpwem/cross_domain_memory_transfer_to_vpwem.md`
- `literature/vpwem/vpwem_p0_idea_fusion_feishu.md`
- `docs/VPWEM_DP记忆机制_创新审计与研究重构_2026-08-21.md`
- `docs/VPWEM_创新点宏观重构与整合_2026-08-21.md`
- `docs/acsm_vpwem_full_research_and_experiment_roadmap_2026-08-21.md`
- `artifacts/per_vpwem/formal_eval/FORMAL_RESULTS.md`
- `artifacts/learned_retention_seed100/SCREENING_REPORT.md`
- `artifacts/random_reservoir_crossbench_strict/CROSSBENCH_RESULTS.md`

同期工作：

- VPWEM: https://arxiv.org/abs/2603.04910
- PAM: https://arxiv.org/abs/2512.24638
- TRACE: https://arxiv.org/abs/2606.14551
- TFP: https://arxiv.org/abs/2607.08283
- MEMBOT: https://arxiv.org/abs/2509.11225
- MemoAct: https://arxiv.org/abs/2603.18494

