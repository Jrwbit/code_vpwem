# No-History strict、Direct-History strict 与 VPWEM strict 对比说明

> 盘点时间：2026-09-09 15:11（Asia/Shanghai）  
> 只讨论 Robomimic 标准四任务：Square/ph、Square/mh、Transport/ph、Transport/mh。  
> 目标：说明三者分别改了什么、当前是否构成公平对比，以及相对 VPWEM 的差距。

## 1. 一句话结论

- `No-History strict` 是“删除全部长期历史”的必要下界。
- `Direct-History strict` 是“删除 memory bank 和 Q-Former，把冻结视觉骨干的空间历史 token 直接交给 diffusion decoder”的替代系统。
- `VPWEM strict` 是“历史全局 embedding → FIFO memory bank → 2-layer Q-Former → 2 memory tokens”的正式基线。
- 当前三者可以比较最终成功率，但只能回答“整个方案谁更好”；由于历史采样和 token 预算不一致，暂时不能严格归因成“Q-Former 压缩本身带来了多少提升”。
- 截至15:11，No-History和Direct-History strict都没有在线成功率结果，因此相对VPWEM的正式差距仍是待测值。

## 2. 每个方法具体改了什么

| 项目 | No-History strict | Direct-History strict | VPWEM strict |
| --- | --- | --- | --- |
| 当前观测 | 最近2帧，每帧一个256维 `dp_emb` | 与No-History相同 | 与No-History相同 |
| 长期历史 | 完全删除 | 保留4帧：episode首帧anchor + 最近3帧 | 从更早历史采样，经FIFO bank保存，最多8个bank槽位 |
| 历史表示 | 无 | 每相机每帧4×4=16个空间tokens；另加1个proprio token/帧 | 每个历史帧先成为一个融合RGB与proprio的256维全局embedding |
| 历史压缩 | 无 | 无memory bank、无Q-Former | 2-layer Q-Former压缩为固定2 tokens |
| 给decoder的历史token数 | 0 | Square 132；Transport 260 | 2 |
| diffusion decoder | 8层PTP Transformer，`Tm=0` | 8层Direct-History PTP Transformer | 8层PTP Transformer，`Tm=2` |
| 条件序列 | diffusion step + 2 current obs | diffusion step + 2 current obs + direct history | diffusion step + 2 memory + 2 current obs |
| 训练目标 | 原始diffusion action denoising loss | 同一diffusion action denoising loss | 同一diffusion action denoising loss |
| memory辅助损失 | 无 | 无 | 无 |

### 2.1 No-History strict

它只保留当前两帧的 `dp_emb`，并把 `LongMemChiTransformerPTP` 的 `Tm` 设置为0。配置文件里虽然还显示Q-Former相关字段，但代码没有实例化或调用Q-Former，这些字段对该实验不起作用。

这个实验回答：不看长期历史时，标准DP/PTP骨干本身能做到多高成功率。

### 2.2 Direct-History strict

它删除：

- VPWEM FIFO memory bank；
- Q-Former；
- 固定2个episodic memory tokens。

它新增：

- 从与当前 `dp_emb` 相同的冻结EMA视觉骨干提取空间feature map；
- 每个相机feature map做4×4池化；
- 空间线性投影、camera embedding、spatial position embedding；
- 每个历史帧一个proprio token和时间embedding；
- 支持变长历史padding mask的Direct-History decoder。

历史固定为最多4帧。当历史超过4帧时，实际保留首帧anchor和最近3帧。

Square有2个相机，所以每帧为 `2×16+1=33` tokens，4帧共132 tokens。Transport有4个相机，所以每帧为 `4×16+1=65` tokens，4帧共260 tokens。

### 2.3 VPWEM strict

VPWEM把当前窗口之前的历史抽样为256维全局embedding，送入容量8的FIFO bank，再由2-layer Q-Former压成固定2个memory tokens。这2个tokens与当前2帧观测共同条件化8层diffusion decoder。

这个实验回答：学习式固定容量压缩的完整VPWEM系统性能。

## 3. 已经匹配的实验条件

| 条件 | 是否匹配 | 证据/说明 |
| --- | --- | --- |
| 任务和数据demonstrations | 是 | 四个实验均来自同一Robomimic ph/mh数据源 |
| 当前观测 `dp_emb` | 是 | Direct数据直接复制原 `dp_emb`；重新计算校验最大误差约 `3e-5` |
| 冻结视觉骨干 | 是 | Direct空间特征来自生成 `dp_emb` 的同一个DP-EMA context checkpoint |
| 训练seed | 是 | seed100 |
| decoder深度 | 是 | 8 layers |
| obs/action/horizon | 是 | obs=2、action=8、horizon=9 |
| diffusion设置 | 是 | DDPM、50 sample steps |
| 学习率 | 是 | `1e-4` |
| optimizer更新预算 | 计划匹配 | 都以1M optimizer steps为终点 |
| effective batch | 是 | 64；Direct通过梯度累积达到64 |
| dropout | 基本匹配 | short=0.3；VPWEM/Direct long=0.3；No-History没有长期分支 |

Direct的空间特征不是来自另一个视觉模型：manifest明确记录了同一context checkpoint，并验证重建的当前 `dp_emb` 与原数据一致。因此“视觉骨干不同”不是当前主要混杂因素。

## 4. 当前仍未匹配的条件

| 未匹配项 | 当前差异 | 对结论的影响 | 修正方式 |
| --- | --- | --- | --- |
| 历史时间采样 | Direct为stride1、anchor+recent3；VPWEM为每20帧抽样并经FIFO8 | 无法区分时间覆盖与压缩结构的贡献 | 给三者生成同一组固定history timestamps |
| 历史token预算 | Direct为132/260；VPWEM仅2 | 无法判断差异来自结构还是上下文带宽 | 增加Direct→2 tokens的token-matched对照 |
| 历史模态组织 | Direct是空间RGB tokens + 独立proprio token；VPWEM是融合后的全局embedding | 比较中同时改变了空间表示和压缩器 | 增加global-history direct与spatial-history compressed对照 |
| DataLoader采样 | No-History普通shuffle；Direct/VPWEM按history length排序采样 | batch顺序是非预期混杂 | No-History也使用同一个LongMem sampler但丢弃history |
| decoder实现 | Direct使用单独的Direct-History类和padding mask | 除history输入外还改变了少量实现细节 | 统一成同一个decoder类、仅切换history adapter |
| 参数量 | No-History显示9.62M；VPWEM约11.86M；Direct约9.47M/9.77M | 可以比较系统，但不是参数匹配比较 | 同时报告总参数与实际激活参数；必要时做parameter-matched |
| 评测入口 | 当前没有strict No-History/Direct-History在线eval pipeline | 训练结束仍不能直接产出同口径成功率 | 训练完成前补齐统一evaluator |

其中最关键的是前两项：同一history evidence和同一token budget。若不修正，最终只能写“系统A优于系统B”，不能写“Q-Former比直接历史压缩更有效”。

## 5. 当前训练状态

| 方法 | Square/ph | Square/mh | Transport/ph | Transport/mh | 是否已有在线成功率 |
| --- | ---: | ---: | ---: | ---: | --- |
| No-History strict | 566k | 510k | 565k | 524k | 否 |
| Direct-History strict | 33k | 26k | 15k | 11k | 否 |
| VPWEM strict | 已完成1M | 已完成1M | 已完成1M | 已完成1M | 是 |

No-History已保存100k至500k中间checkpoint。Direct-History尚未到第一个100k保存点。由于Direct每次更新要处理132/260个历史tokens，训练墙钟时间远大于No-History和VPWEM。

训练loss不能直接横向比较：三者条件信息量和decoder输入长度不同，最终结论必须来自相同episode上的在线Success和效率指标。

## 6. VPWEM正式参考值

每点30个rollouts，固定评测seed：

| 任务 | VPWEM 800k | VPWEM 900k | VPWEM 1M |
| --- | ---: | ---: | ---: |
| Square/ph | 83.3% | 93.3% | 90.0% |
| Square/mh | 70.0% | 70.0% | 83.3% |
| Transport/ph | 83.3% | 93.3% | 80.0% |
| Transport/mh | 70.0% | 36.7% | 63.3% |
| 四任务宏平均 | 76.7% | 73.3% | 79.2% |

checkpoint波动明显，特别是Transport/mh。因此最终不应只挑一个最优checkpoint；应至少报告800k/900k/1M曲线，主对比预先固定1M。

## 7. 当前相对VPWEM差多少

### 7.1 Strict正式差距

| 任务 | No-History strict − VPWEM 1M | Direct-History strict − VPWEM 1M |
| --- | ---: | ---: |
| Square/ph | 待评测 | 待评测 |
| Square/mh | 待评测 | 待评测 |
| Transport/ph | 待评测 | 待评测 |
| Transport/mh | 待评测 | 待评测 |

当前不能用训练loss填写这些差值。

### 7.2 非strict 400k Direct-History初筛，仅作风险提示

| 任务 | Direct-History 400k | VPWEM 1M | 表面差值 |
| --- | ---: | ---: | ---: |
| Square/ph | 63.3% | 90.0% | -26.7pp |
| Square/mh | 20.0% | 83.3% | -63.3pp |
| Transport/ph | 33.3% | 80.0% | -46.7pp |
| Transport/mh | 13.3% | 63.3% | -50.0pp |
| 四任务宏平均 | 32.5% | 79.2% | -46.7pp |

这组Direct使用raw-image联合训练且只有400k，不能当作strict最终结论。它只能说明Direct-History存在较高失败风险，不支持现在扩seed。

## 8. 三个正式比较应该怎样解释

1. `VPWEM − No-History`：长期历史整体带来的价值。
2. `Direct-History − No-History`：直接空间历史相对完全无历史的价值。
3. `VPWEM − Direct-History`：两个完整history系统的性能、效率差异。

若要进一步声称第3项就是“Q-Former压缩收益”，必须先补同timestamp、token-matched和统一sampler版本。

## 9. 最小闭环方案

1. 先为No-History现有500k checkpoint补统一在线评测，尽快得到历史下界。
2. Direct-History不要等到1M才评；保存并评测50k或最迟100k。
3. 三者使用相同的30个episode做快速筛选，正式结论扩到至少100个paired episodes。
4. 主表固定比较1M；同时报告800k/900k/1M，避免checkpoint cherry-picking。
5. 同时报告Success、Wilson 95% CI、推理延迟、峰值显存、history token数和训练吞吐。
6. 补一个关键的token-matched Direct-History-2tok实验，再讨论Q-Former本身的贡献。

## 10. 当前可下与不可下的结论

可以下的结论：

- 三条线的当前观测表示和主要训练超参数已经较好匹配。
- No-History是有效的必要下界。
- Direct-History是一个高带宽、无Q-Former的完整替代系统。
- 早期非strict结果显示Direct-History明显落后，值得设置早停门槛。

暂时不能下的结论：

- No-History比VPWEM低多少。
- Direct-History strict比VPWEM低多少。
- Q-Former本身贡献了多少百分点。
- Direct-History失败是因为无压缩、时间采样、token过多还是decoder优化困难。
