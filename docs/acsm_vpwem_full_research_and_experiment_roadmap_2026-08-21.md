# ACSM–VPWEM 完整研究路线与全量实验清单

更新时间：2026-08-21  
项目定位：从 VPWEM 的 memory retention 问题，推进到 ACSM 的 memory necessity 与 action-relevant belief 问题

> **VPWEM addresses memory retention, whereas ACSM addresses memory necessity.**

本文不是结果报告，而是后续研究、实现、筛选、正式评测和论文写作的统一执行方案。所有尚未运行的内容均为计划，不得提前写成实验结论。

## 一、项目的出发点

### 1.1 VPWEM 已经回答的问题

VPWEM 的基本链路是：

```text
历史 observation
    ↓
历史采样、缓存与压缩
    ↓
Q-Former memory tokens
    ↓
Diffusion Policy action chunk
```

它主要研究：

> 在有限的 memory/token 预算下，哪些历史视觉证据应该保留或压缩？

### 1.2 现有扩展实验揭示的问题

已经实现和筛选过的方向包括：

- delta-PER：按表征变化写入或替换事件；
- Learned retention：预测历史帧对未来动作的离线影响；
- Random retention / Reservoir sampling；
- Recent/FIFO、Oracle ranking 等对照。

当前证据不支持把它们写成已经有效的新方法：

- PER 在 Transport/mh 三训练 seed 上相对 FIFO 的平均差为 `+2.89pp`，区间跨过零；
- Learned retention 在单 seed 筛选中为 `39%`，FIFO 为 `38%`，基本持平；
- Random 相对原 Q-Former/FIFO 曾出现三 seed 平均 `+13.3pp`，但表示结构与保留规则混杂；
- 严格的 raw-token Reservoir-2 vs Recent-2 对照中，Square/mh 为 `+1pp`，Transport/ph 为 `-2pp`，没有稳定优势；
- Oracle 只说明离线存在可排序的历史帧，不说明策略在线真正依赖这些信息。

因此，失败根因不能再简单归结为 retention score、bank size 或淘汰规则不够好。

### 1.3 新的核心判断

```text
Correct evidence retained
        ≠
Policy causally relies on the evidence
```

也就是：

```text
Memory availability ≠ Memory necessity
```

下一步需要回答：

> 当前观测何时不足以决定动作？在这些时刻，如何通过训练使历史内容成为动作分支选择的必要因果条件？

## 二、ACSM 的最终问题定义

ACSM 暂定全称：

> **Ambiguity-aware Counterfactual State Memory**

完整方法最终由三部分组成：

```text
Observational Alias Mining
            ↓
Matched Counterfactual History Training
            ↓
Compact Action-Relevant Belief
```

但研究必须分阶段推进。第一轮只验证 matched counterfactual supervision，不能提前声称已经验证完整 ACSM。

### 2.1 当前非记忆输入

对时刻 `i`，定义：

```text
c_i = policy 在不读取 long-term memory 时，
      当前决策步骤能够直接获得的全部输入
```

它可以包含：

- 当前或固定短窗口视觉输入；
- 当前 proprioception / joint state；
- 当前 gripper state；
- 当前指令或任务条件；
- 原策略接口明确提供的其他非长期记忆变量。

不要在论文定义中固定写 `recent actions`。某个输入是否属于 `c_i`，必须以实际 policy interface 为准；如果 previous action 只通过长期 memory 获得，它就属于历史 `h_i`，不能同时进入 matching key。

### 2.2 历史信息

```text
h_i = 只有通过 memory 路径才能获得的过去信息
```

严格的 observational alias pair 满足：

```text
c_i ≈ c_j
z_i ≠ z_j
```

其中 `z` 是当前不可直接观测、但会改变正确动作分支的任务状态，例如：

- ShellGameTouch 的正确目标杯；
- RememberColor 的目标颜色；
- SwapBlocks 的物体身份或交换阶段；
- 已完成/未完成的子任务；
- 早期 cue 指定但当前已不可见的目标。

其概率含义是：

```text
P(a | c_i, h_i) ≠ P(a | c_i, h_j)
```

### 2.3 不应使用的宽松定义

以下条件不能单独证明 observational aliasing：

- 当前 RGB embedding 相似；
- action chunk 的原始 L2 距离较大；
- 两个时刻来自不同 episode；
- 两个动作方向短时不同。

这些差异可能来自 proprioception、时间错位、演示风格、速度差异或 action multimodality，而非隐藏任务状态。

## 三、与近期工作的边界

### 3.1 已被近期工作覆盖的方向

- PAM：state ambiguity、长历史工作记忆和紧凑上下文表征；
- TRACE：delayed evidence、相似当前观察、固定大小 latent memory；
- TFP：连续 task-progress belief、event-sensitive update 和 hidden-state intervention；
- MEMBOT：显式 latent belief encoder，并整合 observation/action history；
- MemoAct：层级 sensory/short-term/long-term memory。

因此不能声称：

- 首次解决视觉歧义；
- 首次提出 compact memory；
- 首次维护 task belief；
- 首次用 hidden-state intervention 分析机器人策略。

### 3.2 ACSM 应主张的差异

候选主张是：

> Existing methods primarily optimize what to store or how to update memory. ACSM targets when memory is causally necessary for action by mining observationally matched yet task-state-mismatched demonstrations and using their histories as counterfactual training interventions.

中文表述：

> ACSM 不把记忆容量或更新结构作为第一矛盾，而是从 demonstration 中自动识别“当前可观测输入相似、隐藏任务状态不同”的决策点，并交换匹配历史作为反事实监督，迫使策略形成历史条件下的动作分歧。

这仍是候选新颖性判断，投稿前必须继续核对同期工作，尤其是 TRACE、PAM 和后续 observational aliasing / counterfactual imitation 工作。

## 四、总研究路线

```text
Stage 0：VPWEM failure diagnosis
    │     证明 keep ≠ use，清理历史实验中的混杂
    ▼
Stage 1：Matched-CF hypothesis test
    │     不换 memory architecture，只换监督方式
    │
    ├── 不成立：停止 ACSM 架构开发，排查 pair mining / loss
    │
    └── 成立
          ▼
Stage 2：Compact causal belief
    │     用固定大小 belief 表示 action-relevant task state
    ▼
Stage 3：Full ACSM
    │     Alias Mining + Matched-CF + Compact Belief
    ▼
Stage 4：跨任务、多 seed、长度外推与论文级验证
```

## 五、Stage 0：VPWEM 失败诊断与实验地基

### 5.1 目的

- 固化已有结果，避免把不同架构的比较误解为 retention 机制比较；
- 确认哪些任务真的要求长期记忆；
- 建立 paired rollout、统一随机性和干预评测框架；
- 为 Stage 1 提供同一 checkpoint、同一数据、同一评测 episode 的公平起点。

### 5.2 必须保留的已有对照

| 类别 | 方法 | 角色 |
|---|---|---|
| 无长期记忆 | DP / short-window DP | 下界 |
| 原方法 | VPWEM Q-Former | 主基线 |
| 原始历史 token | Recent-K | 等 token 近期历史基线 |
| 随机历史 | Reservoir-K | 无信息选择基线 |
| 事件选择 | delta-PER | 已有候选方法 |
| 学习选择 | Learned retention | 已有失败候选 |
| 离线上界 | Oracle ranking | 诊断，不是在线方法 |

### 5.3 Stage 0 输出

- 每个任务是否真正存在长时依赖的判定；
- equal-token、equal-backbone、equal-training-budget 的可比配置；
- correct / zero / shuffle / reverse 等现有干预结果；
- 明确标记无效 checkpoint、配置错误和不能用于论文的历史结果。

## 六、Stage 1：Matched-CF 假设检验

### 6.1 本阶段只验证什么

本阶段只验证：

> 将随机错误历史替换为“当前可观测输入相似、隐藏任务状态不同”的错误历史，是否能使现有 VPWEM memory 变得 action-relevant？

本阶段不验证：

- 新 memory architecture；
- belief slots；
- learned write gate；
- 新 retention score；
- 完整 ACSM。

### 6.2 第一任务：MIKASA ShellGameTouch

选择原因：

- 存在明确的早期 cue 和后期目标选择；
- 当前决策点可能视觉相似；
- 环境能提供正确目标杯的 GT branch label；
- 不需要用 action L2 猜测行为分支；
- 已有工作表明该任务对 task-progress memory 敏感。

### 6.3 Pair mining

#### Step A：构造完整 current condition

用冻结的当前 policy encoder 得到：

```text
u_i = EncodeAllNonMemoryInputs(c_i)
```

必须审计 encoder 输入，确保包含所有当前非记忆 policy 输入，而不混入 long-term history。

#### Step B：限定候选决策区间

只在真正可能发生 branch decision 的时间区域挖 pair，避免把大量无关运动阶段当成歧义样本。候选区间可由以下方式确定：

- 环境阶段标记；
- 目标接触前的固定窗口；
- 首次进入目标区域；
- GT subgoal transition。

#### Step C：建立 matched pair

对样本 `i`，从不同 episode 中寻找 `j`：

```text
sim(u_i, u_j) ≥ τ_c
branch_label_i ≠ branch_label_j
split_i = split_j = train
episode_i ≠ episode_j
```

优先使用 mutual nearest neighbor 或 top-k 后再做阈值过滤，防止单向近邻造成密度偏差。

#### Step D：质量审计

在训练前输出：

- pair 总数、覆盖率和每个 branch 的数量；
- current-condition 相似度分布；
- proprioception / gripper 差异分布；
- 时间位置差异；
- 人工可视化不少于 100 对；
- 错误分支、重复帧和明显不同当前状态的比例。

必须先通过 pair 质量审计，再开始 CF 训练。

### 6.4 三组核心方法

| 方法 | Memory | CF negative | 目的 |
|---|---|---|---|
| VPWEM | 原 Q-Former | 无 | 基线 |
| VPWEM + Random-CF | 原 Q-Former | batch random roll | 判断普通错误历史监督是否有效 |
| VPWEM + Matched-CF | 原 Q-Former | matched wrong history | 验证核心假设 |

三者必须满足：

- 相同基础 checkpoint；
- 相同训练数据与训练步数；
- 相同 token 数、Q-Former 和 diffusion backbone；
- 相同 optimizer、learning rate、batch size；
- 相同训练 seed；
- 仅 CF negative 构造不同。

### 6.5 对称反事实损失

对 matched pair `(i,j)`：

```text
L_CF(i,j)
  = [m + loss(c_i, B_i, a_i) - loss(c_i, B_j, a_i)]_+
  + [m + loss(c_j, B_j, a_j) - loss(c_j, B_i, a_j)]_+
```

总损失：

```text
L_total = L_DP + λ_CF · L_CF
```

### 6.6 Diffusion 配对约束

每个 correct / wrong belief 比较必须共享：

- 相同 diffusion timestep `t`；
- 相同噪声 `epsilon`；
- 相同 noisy action `x_t`；
- 相同 loss mask 和 timestep weighting。

当前 `LongMemDDPM` 的 random batch-roll 分支已经复用同一个 `x_t`、`t` 和 target；修改时必须保持这一性质。

### 6.7 第一轮 smoke test

配置：

- 任务：ShellGameTouch；
- training seed：100；
- 统一 evaluation seed；
- 每种方法：相同 10k adaptation；
- 每种方法：100 paired rollout episodes；
- 每个 episode 保存 branch decision、success、reward 和 intervention 结果。

第一轮指标：

1. Overall Success；
2. Branch Decision Accuracy；
3. Counterfactual Memory Gap：

```text
Delta_CF = loss_wrong - loss_correct
```

4. Normalized Counterfactual Memory Gap：

```text
R_CF = (loss_wrong - loss_correct) / (loss_correct + epsilon)
```

5. 四种 intervention：

- Correct history；
- Zero history；
- Random history；
- Matched-wrong history。

计算 `Delta_CF` 和 `R_CF` 时，应对多个固定的 `(t, epsilon)` 抽样取均值，并让所有模型复用相同抽样表，降低 Monte Carlo 方差。

### 6.8 Stage 1 预注册晋级条件

Matched-CF 相对 VPWEM：

- `Delta BranchAcc ≥ +10pp`；
- Correct 相对 Matched-Wrong 的 success drop `≥15pp`；
- `Delta_CF` 与 `R_CF` 明显高于 VPWEM 和 Random-CF；
- overall success 的筛选差值 `≥-5pp`；
- 结果不是由少数 episode 或单一 branch 驱动。

`-5pp` 只是 smoke-test 筛选阈值，不能写成统计非劣效结论。正式非劣效需要预先定义 margin、足够样本量和置信区间。

### 6.9 Stage 1 停止条件

出现以下任一情况，不进入 belief architecture：

- matched pair 人工审计显示大量假歧义；
- Matched-CF 不优于 Random-CF；
- BranchAcc 无明显提高；
- Matched-Wrong 与 Random history 的干预差异很小；
- `Delta_CF` 上升但 rollout branch choice 不变；
- overall success 明显下降。

停止后优先排查：

1. `c_i` 是否遗漏当前可观测变量；
2. branch label 是否与真正决策阶段对齐；
3. pair 是否存在时间错位；
4. margin 与 loss scale 是否匹配；
5. CF loss 是否只学会放大错误历史损失而破坏正确动作建模；
6. 现有 VPWEM memory 是否根本没有携带可区分历史。

## 七、Stage 1.5：Matched-CF 稳健性验证

只有 Stage 1 smoke test 达标后执行。

### 7.1 多训练 seed

- training seed：100、200、300；
- 每个 seed 使用相同 paired evaluation initial states；
- 推断单位是 training seed，不把全部 episode 当独立训练重复；
- 同时报 paired episode bootstrap 和跨训练 seed 汇总。

### 7.2 第二和第三任务

优先顺序：

1. MIKASA RememberColor3；
2. RMBench SwapBlocks；
3. MIKASA RememberColor5；
4. 其他有明确 hidden branch label 的 memory task。

Transport、Square 等普通 Robomimic 任务不作为首要涨点场景，只作为“非记忆任务不应退化”的控制任务。

### 7.3 Pair mining 消融

| 变体 | 目的 |
|---|---|
| random negative | 普通错误历史基线 |
| visual-only matched | 暴露遗漏 proprioception 的假歧义 |
| full-current matched | 正式方案 |
| same branch matched | 负对照；相似历史不应被强迫分离 |
| unmatched different branch | 判断 current-state matching 是否必要 |
| nearest vs top-k sampled | 检查近邻过拟合 |

### 7.4 Loss 消融

- 单向 CF vs 对称 CF；
- fixed margin vs relative margin；
- 不同 `λ_CF`；
- 不同 matched-pair sampling ratio；
- only ambiguous batches vs ambiguous/ordinary mixed batches；
- hard-negative curriculum：先宽松、后严格。

## 八、Stage 2：Compact Causal Belief

### 8.1 进入条件

只有当 Stage 1 证明 matched supervision 明显优于 random supervision，才实现新 memory architecture。

### 8.2 研究假设

Stage 1 若成立，说明第一矛盾是监督；Stage 2 再回答：

> 相比压缩历史帧，一个显式、固定容量、因果更新的 task belief 是否更适合承载 matched-CF 学到的动作必要信息？

### 8.3 最小 belief 实现

输入：

```text
当前 observation embedding
+ 当前 robot state
+ previous action（仅作为 causal updater 输入）
+ 上一步 belief
```

更新：

```text
B_t = CausalUpdater(B_{t-1}, e_t, a_{t-1})
```

输出：固定 `K` 个 belief tokens，例如 `K=4`、维度 `256`，直接复用 VPWEM 的 memory-token 接口输入 Diffusion Policy。

首版只选一种简单 updater：GRU、轻量 causal transformer 或 gated slot update 三者择一。不要同时创新 updater、slot routing 和 decoder fusion。

### 8.4 Belief 监督

#### Matched-CF necessity supervision

继续使用 Stage 1 的对称反事实损失，作为核心目标。

#### Privileged state sufficiency supervision

在模拟任务中，训练时可增加：

```text
L_state = CE / regression(g(B_t), z_t)
```

预测：

- correct cup / target color；
- object identity；
- subtask phase；
- completed-step counter；
- environment-provided hidden state。

测试时不使用 privileged label。

`L_state` 是 sufficiency supervision，不是核心新颖性；matched-CF 才检验 belief 是否 action-relevant。

### 8.5 Stage 2 方法矩阵

| 方法 | Memory representation | Matched-CF | State supervision |
|---|---|---:|---:|
| VPWEM | Q-Former frame memory | 否 | 否 |
| VPWEM + Matched-CF | Q-Former frame memory | 是 | 否 |
| Recurrent DP | single recurrent state | 否 | 否 |
| Belief-only | compact belief | 否 | 否 |
| Belief + State | compact belief | 否 | 是 |
| Belief + Matched-CF | compact belief | 是 | 否 |
| ACSM-Full | compact belief | 是 | 是 |

### 8.6 Stage 2 成功判据

ACSM-Full 至少应满足：

- 在 memory task 上优于 `VPWEM + Matched-CF` 或以更低计算成本持平；
- belief probe 能恢复 GT task state；
- Matched-Wrong intervention 对 branch choice 有显著、方向正确的影响；
- state supervision 单独使用不如 state + matched-CF；
- ordinary/non-memory task 不出现系统性退化。

如果 `VPWEM + Matched-CF` 已经最好，而 compact belief 无优势，则论文可以收缩为 matched counterfactual history training，不强行保留 ACSM architecture 主张。

## 九、Stage 3：完整 ACSM

### 9.1 完整方法

```text
Demonstrations
    │
    ├── all non-memory current inputs ──► Alias Miner
    │                                      │
    │                                      └── matched wrong histories
    │
    └── causal history ──► Compact Belief Updater ──► belief tokens
                                                    │
current condition ──────────────────────────────────┼──► Diffusion Policy
                                                    │
matched belief swap ────────────────────────────────┘
```

训练目标：

```text
L = L_DP + λ_CF L_symmetric_matched_CF + λ_state L_state
```

可选正则项必须通过单独消融后才保留：

- belief temporal stability；
- event-boundary update sparsity；
- uncertainty calibration；
- slot diversity；
- belief bottleneck。

不要一开始全部加入。

### 9.2 在线推理约束

- 只使用当前观测、当前 robot state、过去动作和已有 belief；
- 不访问未来动作、未来状态或 GT branch；
- memory/token 数固定；
- episode 长度增长时，memory 占用保持常数；
- 明确 reset 条件；
- 对 belief update、readout 和 policy inference 分别计时。

## 十、Stage 4：论文级全量实验

### 10.1 任务组

#### A. 强记忆任务：主要结论

- MIKASA ShellGameTouch；
- MIKASA RememberColor3；
- MIKASA RememberColor5；
- RMBench SwapBlocks；
- 后续选择具有 GT hidden state / branch label 的 delayed-evidence task。

#### B. 弱记忆或近 Markov 任务：安全控制

- Robomimic Square；
- Robomimic Transport；
- MIKASA InterceptGrabMedium；
- 其他当前观察已基本充分的任务。

这些任务的目标不是要求 ACSM 大幅涨点，而是验证不会因强制使用 memory 而退化。

#### C. 长度外推任务

- cue-to-decision delay 为训练长度的 `1×、2×、4×`；
- 插入无关动作或等待阶段；
- 增加 distractor events；
- 延长稳定/遮挡阶段。

### 10.2 完整基线

| 组别 | 基线 |
|---|---|
| 无记忆 | DP、short-window DP |
| 简单历史 | frame stacking、Recent-K、Reservoir-K |
| 循环状态 | GRU/LSTM recurrent DP |
| 原项目 | VPWEM Q-Former |
| 已有扩展 | PER、Learned retention |
| 同期强基线 | PAM-style、TRACE-style、TFP-style，按可获得代码与公平性决定 |
| 本方法 | VPWEM + Matched-CF、Belief + Matched-CF、ACSM-Full |

若同期方法代码不可得，必须明确标记为概念比较，不能用不等价的简化实现声称胜过原方法。

### 10.3 主指标

- Overall Success；
- Branch Decision Accuracy；
- Ambiguous-State Success；
- Subgoal completion；
- 平均 reward；
- `Delta_CF`；
- `R_CF`；
- belief/state probe accuracy；
- inference latency；
- peak GPU memory；
- memory/token budget；
- 随 episode length 的计算和显存曲线。

### 10.4 因果干预矩阵

每个正式模型至少评估：

| Intervention | 含义 | 期望 |
|---|---|---|
| Correct | 正确历史/belief | 正常性能 |
| Zero | 删除 memory | memory task 明显下降 |
| Random | 随机其他 episode history | 中等下降 |
| Matched-Wrong | 当前状态匹配、branch 错误 | 最大且方向明确的下降 |
| Same-Branch Matched | 当前匹配、branch 相同 | 应接近 Correct |
| Early-only | 只保留早期 cue | 检验 cue sufficiency |
| Late-only | 删除早期 cue | delayed-evidence task 应下降 |
| Belief slot swap | 交换指定 slot | 检验 slot 语义与因果作用 |

不仅报告 success drop，还应报告错误动作是否转向被换入历史对应的 branch。否则只能说明 memory 扰动有害，不能说明它编码了正确任务状态。

### 10.5 必做消融

#### 监督消融

- 无 CF；
- Random-CF；
- Matched-CF；
- Matched-CF + state supervision；
- state supervision only；
- 单向 vs 对称 CF。

#### Mining 消融

- visual-only key；
- full-current-condition key；
- GT branch label；
- action-mode proxy；
- random negative；
- same-branch negative control；
- 不同 similarity threshold / top-k。

#### Memory 消融

- Q-Former frame memory；
- recurrent state；
- compact belief；
- `K=1/2/4/8` slots；
- with/without previous action in causal updater；
- with/without positional/time input。

#### 训练消融

- 不同 `λ_CF`；
- 不同 margin；
- CF sample probability；
- hard-negative curriculum；
- 从头训练 vs VPWEM checkpoint adaptation；
- frozen encoder vs joint fine-tuning。

### 10.6 统计规范

- smoke test：seed100、100 paired episodes，只用于筛选；
- 正式结果：至少 3 个 training seeds；
- 每个训练 seed 使用相同的一组 evaluation initial states；
- success 使用 paired episode difference 和 bootstrap interval；
- 二分类配对可报告 McNemar exact test；
- 跨 seed 推断以 training seed 为重复单位；
- 不把 300 个 episode 当作 300 个独立训练重复；
- 同时报告点估计、区间和原始成功数；
- 多任务/多消融时说明多重比较处理或将结果明确标为探索性。

## 十一、数据泄漏和伪因果风险

### 11.1 Split 泄漏

- pair index 只能在 train split 内建立；
- validation/test 不得参与训练近邻库；
- normalization、action-mode clustering 和 threshold 调参只能用 train/validation；
- test branch labels 只用于评测，不参与训练。

### 11.2 假歧义

必须排除：

- proprioception 差异足以解释动作；
- gripper 状态不同；
- instruction 不同；
- camera/view 不同；
- 时间错位导致 action chunk 不同；
- demonstrator style 或速度不同；
- diffusion 多模态导致两个动作都正确。

### 11.3 捷径学习

模型可能通过 episode identity、时间步、padding pattern 或数据采集噪声识别 branch。需要：

- 跨 episode pairing；
- 时间/长度平衡；
- 去除无意义 ID；
- same-branch matched control；
- 新 episode、新 cue arrangement 和长度外推测试。

### 11.4 “干预有害”不等于“状态正确”

Matched-Wrong 导致性能下降只是必要证据。更强证据是：动作系统性转向错误历史对应的 branch，并且 Same-Branch Matched 不造成同等下降。

## 十二、实现任务清单

### 12.1 Stage 1 数据层

- 审计 ShellGameTouch policy 的全部 non-memory inputs；
- 提取决策时刻与 GT branch label；
- 生成冻结 current-condition embeddings；
- 构建 train-only approximate/exact nearest-neighbor index；
- 生成 matched-pair manifest；
- 保存 pair similarity、episode、timestep、branch 和质量标记；
- 实现 pair-aware sampler/collate；
- 输出 pair 可视化审计报告。

### 12.2 Stage 1 模型层

- 扩展 `LongMemDDPM.loss()` 接收 matched history；
- 保留 shared `t/epsilon/x_t`；
- 实现对称 CF；
- 分别记录 correct、wrong、margin violation、`Delta_CF`、`R_CF`；
- 支持 random、matched、same-branch 三种 negative mode；
- 确保 EMA 和 checkpoint 包含所需状态。

### 12.3 Stage 1 评测层

- 统一 paired rollout initial states；
- 正常与四类 intervention；
- 记录 branch decision；
- 保存逐 episode 结果；
- 自动生成 bootstrap、McNemar 和干预方向分析；
- 失败视频按 branch 分类。

### 12.4 Stage 2 实现

- 新增 `CompactBeliefMemory`；
- 明确 reset、update、readout；
- 接入 previous action 的因果路径；
- 增加 privileged state head；
- 增加 belief trace 日志；
- 单元测试 shape、causality、reset、checkpoint round-trip；
- 测试未来信息泄漏和 batch independence。

## 十三、实验执行顺序与计算预算控制

### Phase A：零成本/低成本检查

1. current-condition 输入审计；
2. GT branch label 可用性；
3. pair coverage；
4. 100 对人工审计；
5. loss 单元测试；
6. 极小 batch overfit，确认 correct/wrong gap 能被优化。

任何一项失败都不进入 rollout。

### Phase B：Stage 1 smoke

1. VPWEM；
2. Random-CF；
3. Matched-CF；
4. ShellGameTouch seed100；
5. 100 paired episodes；
6. 按预注册阈值 go/no-go。

### Phase C：Stage 1 正式化

1. training seed100/200/300；
2. ShellGameTouch；
3. RememberColor3；
4. SwapBlocks；
5. mining 与 loss 关键消融。

### Phase D：Stage 2 belief

1. 只实现一个最小 updater；
2. 对比 VPWEM + Matched-CF；
3. 添加 state supervision；
4. 决定论文是 supervision-only 还是 full ACSM。

### Phase E：论文级扩展

1. 多任务、多 seed；
2. 长度外推；
3. 完整 intervention；
4. 效率；
5. 强基线；
6. 真实机器人或更高保真环境验证（若资源允许）。

## 十四、每阶段交付物

### Stage 0

- VPWEM/PER/Random/Learned/Reservoir 历史实验审计；
- 可比较性表；
- 失败结论和禁止使用的主张。

### Stage 1

- pair manifest；
- pair 质量报告；
- 三方法 checkpoint；
- 逐 episode rollout；
- BranchAcc、`Delta_CF`、`R_CF`；
- 四类干预报告；
- go/no-go 结论。

### Stage 2

- CompactBeliefMemory 实现；
- 单元测试；
- belief probe；
- state supervision / matched-CF 消融；
- 是否保留 belief architecture 的结论。

### Stage 4

- 多任务三 seed 主表；
- ambiguous-state 专项表；
- intervention 因果表；
- 长度外推图；
- 计算效率图；
- 失败案例与视频；
- 完整配置、checkpoint、原始结果和统计脚本。

## 十五、论文叙事与可写主张

### 15.1 当前只能写的问题陈述

- 保存历史证据不保证策略会使用它；
- 现有 VPWEM retention 扩展没有获得稳定收益；
- 需要直接训练和检验 memory necessity。

### 15.2 Stage 1 成立后可写

- observationally matched、task-state-mismatched history 是比随机错误历史更有效的训练干预；
- matched-CF 提高 ambiguous branch decision，并扩大 counterfactual memory gap；
- memory supervision 是当前系统的关键瓶颈之一。

### 15.3 Full ACSM 成立后可写

- ACSM 从 demonstrations 中识别 observational aliasing；
- ACSM 学习固定预算、action-relevant 的 causal belief；
- matched history interventions 同时用于训练和机制验证；
- ACSM 在 delayed-evidence/memory tasks、长度外推和 matched-wrong intervention 上优于公平基线。

### 15.4 禁止提前使用的主张

- “ACSM 已经有效”；
- “ACSM 显著优于 VPWEM”；
- “首次解决视觉歧义”；
- “首次提出 compact belief memory”；
- “适用于所有长时机器人任务”；
- “Robomimic 普通任务必须明显涨点”；
- “Matched-Wrong 下降自动证明因果状态正确”。

## 十六、最终决策树

```text
Matched pair 是否真实且覆盖足够？
    ├── 否：修 pair 定义、当前输入或任务选择
    └── 是
         ↓
Matched-CF 是否优于 Random-CF？
    ├── 否：停止架构开发，排查监督假设
    └── 是
         ↓
BranchAcc 和 matched-wrong intervention 是否同时成立？
    ├── 否：可能只优化了离线 loss，不足以形成论文主张
    └── 是
         ↓
Compact belief 是否优于/等效且更高效于 Q-Former memory？
    ├── 否：保留 Matched-CF 作为主要贡献
    └── 是：形成 Full ACSM
         ↓
多任务、多训练 seed、长度外推是否成立？
    ├── 否：收缩为诊断或 workshop 结论
    └── 是：形成完整投稿证据链
```

## 十七、近期第一行动项

现在不应继续调 retention score、PER 阈值、bank size 或 belief slot 数量。

唯一优先实验是：

> 在 MIKASA ShellGameTouch 上，保持 VPWEM memory architecture、backbone、token 数、checkpoint 和训练预算不变，仅将 random `batch.roll` 错误历史替换为 full-current-state matched、GT-branch-mismatched 的错误历史，比较 VPWEM、Random-CF 和 Matched-CF 的 BranchAcc、Success、`Delta_CF`、`R_CF` 及 Correct/Zero/Random/Matched-Wrong 干预。

这一实验成立，才进入 compact belief；不成立，就优先修正 pair mining 和 counterfactual objective。

## 十八、相关工作入口

- VPWEM：本仓库原始方法与实现；
- PAM：https://arxiv.org/abs/2512.24638
- TRACE：https://arxiv.org/abs/2606.14551
- TFP：https://arxiv.org/abs/2607.08283
- MEMBOT：https://arxiv.org/abs/2509.11225
- MemoAct：https://arxiv.org/abs/2603.18494

---

最终一句话：

> **VPWEM 研究“该记住什么”，ACSM 研究“什么时候记忆必须改变动作”；先用 matched counterfactual history 证明 necessity，再决定是否需要新的 compact belief architecture。**
