# PER-VPWEM 创新点决策报告

日期：2026-08-12

## 决策摘要

当前证据不支持把 PER-VPWEM（delta 事件选择记忆）作为“稳定提升 VPWEM
性能”的论文核心创新。它可以保留为探索性机制或诊断分支，但应立即停止把
大规模跨基准计算资源继续投入到该性能主张上，除非先通过下述确认性 gate。

这不是“原论文 FIFO 全部复现失败，所以 PER 无效”的结论。两个结论需要
分开：

1. 当前基线复现质量不均衡，不能把与论文数字的差异完全归因于 PER；
2. 在当前可配对的 FIFO 对照中，PER 没有呈现跨任务、一致的正向效应，故
   不能支持其作为性能创新点。

## 数据范围

- Transport/mh 正式对照：3 个训练 seed，每个模型在公共 300 个 episode
  上评测；FIFO 和 PER 均从同 seed 基础 checkpoint 等预算适配 10k steps。
- MIKASA / MoMaRT：服务器上的 seed-100 筛选；每个已完成方法—任务组合
  为 100 episodes，公共环境 seed 为 `20260807`。
- 跨基准中，MoMaRT `table_setup_from_dishwasher_expert` 的 FIFO 评测因 OOM
  缺失，故不能进入配对汇总。

## PER 相对 FIFO 的证据

### Transport/mh 正式评测

| 指标 | FIFO | PER | PER - FIFO |
| --- | ---: | ---: | ---: |
| 成功率（900 rollouts） | 357/900，39.67% | 383/900，42.56% | +26，+2.89 pp |
| exact McNemar | - | - | p=0.203 |
| paired bootstrap 95% CI | - | - | [-1.33, +7.11] pp |
| 平均累计 reward | 7.191 | 8.918 | +1.727 |

该任务有正向均值和 reward 信号，但成功率主检验不显著，且训练 seed 100
为负向。因此它是候选信号，不是性能结论。

### 跨基准 seed-100 原始配对重算

| 基准 | 已完成配对 | FIFO 成功 | PER 成功 | 差值 |
| --- | ---: | ---: | ---: | ---: |
| MoMaRT | 9 | 561/900，62.33% | 562/900，62.44% | +1，+0.11 pp |
| MIKASA | 3 | 113/300，37.67% | 109/300，36.33% | -4，-1.33 pp |
| 合计（描述性） | 12 | 674/1200，56.17% | 671/1200，55.92% | -3，-0.25 pp |

合并逐 episode 的 discordant pairs 为 FIFO-only=181、PER-only=178，精确
McNemar p=0.916。该 pooled p 值不能替代多任务、多训练 seed 的正式统计，
但它没有显示任何总体正向证据。

任务层面有 6 个负差与 6 个正差；幅度从 -12 pp（ShellGameTouch）到 +12 pp
（unload-dishwasher-to-dresser/suboptimal）。这说明效应高度异质，且没有稳定
方向。

## 与原论文的关系

服务器 FIFO 并非每项都低于论文：ShellGameTouch 为 90%，与计划文件记录的
论文约 91% 接近；MoMaRT 9 个有配对的 FIFO 均值为 62.33%，与计划文件中
逐图读取的论文值约 65.92% 接近但偏低。相反，RememberColor3 的 FIFO 仅 6%，
远低于记录的论文约 82%；本地 LongSquare 也未复现论文约 70%。

因此，不能把 PER 的无增益解释为“FIFO 基线普遍太差”。更准确的解释是：
当前复现实验存在任务相关的基线偏差，而在同一任务、同一 checkpoint 预算的
FIFO-PER 配对内，PER 仍未显示一致优势。

原论文数值在 `docs/vpwem_experiment_plan_2026-07-26.md` 中部分标为“论文图约”，
不能当作严格逐项统计对比。更重要的是，仓库审计发现论文、发布配置与保存
checkpoint 间存在层数、compressor、采样历史和训练设置的配置漂移。

## 机制层面的判断

当前 PER 的正式候选是非参数 delta 启发式：按相邻 embedding 的
`1 - cosine_similarity` 写入容量为 8 的 bank，超过阈值或达到最大写入间隔时
更新，满库时按衰减分数淘汰槽位，再由原 Q-Former 输出 2 个 token。

这会保留“表征变化大”的观测，但没有证据表明这些变化与动作决策所需的长期
信息相关。预测误差、evidence token 和反事实损失在本轮并未作为有效正式候选
（或未启用），因此目前的创新实质是一个未验证的手工保留规则，而不是学习到
的任务相关 memory selection。

单 seed 小样本干预显示错配 memory 会明显伤害策略，说明策略会使用 memory；
它不能推出“delta 事件库优于 FIFO”。

## 建议

### 立即决策

不要把“PER 优于 FIFO”写作当前论文的贡献，也不建议继续扩大当前 delta 候选的
MoMaRT/MIKASA 大规模 seed 矩阵。主创新方向应准备切换。

### 唯一值得保留的确认性 gate

在决定彻底删除 PER 前，只做一次小而严格的确认：

1. 先复现 FIFO：固定论文级配置，优先修复并验证 RememberColor3 或 LongSquare；
   保存 config、checkpoint hash、逐 episode 结果。
2. 在一个已通过 FIFO parity 的 memory-intensive 任务上，冻结一个 PER 候选，
   用至少 5 个训练 seed、每 seed 300 个未见公共 episode 做 FIFO-PER 配对。
3. 同时记录 bank 写入率、平均 token age、被替换 slot、以及 zero/shuffle memory
   干预；否则无法解释“为何会涨/不涨”。
4. 若成功率差的 95% CI 下界仍不大于 0，或机制指标显示 delta 与任务成功无关，
   则正式放弃 delta-PER 作为性能创新点。

### 候选替代方向

若继续做 memory innovation，优先探索 query/task-conditioned retention：以当前
policy query、动作不确定性或未来 action/value 预测误差评价历史槽位，而不是用
相邻视觉 embedding 的变化量作为代理。该方向必须在开发集完成消融后冻结，并
通过上述确认性协议测试；它是待验证假设，不能预设会优于 FIFO。

## 证据位置

- `artifacts/per_vpwem/formal_eval/FORMAL_RESULTS.md`
- `artifacts/per_vpwem/RESULTS.md`
- `docs/vpwem_claim_code_experiment_audit_2026-08-09.md`
- `docs/vpwem_experiment_plan_2026-07-26.md`
- 服务器：`/home/jrw/vpwem_outputs/per_vpwem_screen_seed100/eval/{momart,mikasa}/`
