# Direct、Direct2、NoHistory、VPWEM 完整实验计划与审计

> 服务器快照：2026-09-20 14:56 CST  
> 审计范围：只保留 `direct`、`direct_token2`、`no_history`、`vpwem`。下文统一把 `direct_token2` 写成 **Direct2**。  
> 原始证据：BHR checkpoint、Hydra config、逐 episode `result.npy`、当前代码和 VPWEM 论文 PDF。

## 一、结论

当前没有任何一组达到“论文级四方法完整对照”。登记系统中的等级统计为：A 级 0 条、B 级 4 条、C 级 8 条、D 级 5 条。

执行更新：按用户要求取消等待，修复后的 BHR VPWEM 复评于14:53在物理 GPU7与现有 MoMaRT Direct 并行启动，14:56完成，无 OOM。正式结果为92%（92/100），平均推理时间0.3862秒。旧29%文件已改名保留为 `result_600000_paired.invalid_reset_bug_20260911.npy`。

MIKASA ShellGameTouch seed100 四方法现已在 BHR 同机闭环：NoHistory 88%、Direct 95%、Direct2 94%、修复后的 VPWEM 92%。原 VPWEM 29% 因 evaluator 没有在 episode 间清空 online/EMA Q-Former memory 而无效。Direct 相对 VPWEM `+3pp`，Direct2 `+2pp`，但配对检验均不显著，且只有一个训练 seed，不能证明优于 VPWEM。

MoMaRT 的 400k/1M 数字可以定位系统表现，但不能作为 memory 结构公平消融：现有 VPWEM 是预计算 embedding policy、effective batch 256，Direct/Direct2 是 raw-image spatial policy、effective batch 64。1M 的 Direct2 71.67% 对 VPWEM 65.33% 是 `+6.34pp`，这个涨点含 observation encoder、训练 batch 和 history adapter 三类变化。

Robomimic 目前只有 checkpoint，没有新的 strict 四方法在线对照。NoHistory 四任务已到 1M；Direct 仅 Square 到 900k、Transport 到 400k；Direct2 strict 尚未跑。旧的非 strict Direct 400k 宏平均 32.5%，只能作为风险信号。

因此，近期不应再开新的大矩阵。P0 原计划的1,520个评测 episode 已完成100个，还应先用已有 checkpoint 补完剩余1,420个，再决定哪些训练值得继续。

## 二、四个方法的固定定义

| 名称 | 代码名 | 长期历史 | 给 diffusion decoder 的历史 token | 科学问题 |
|---|---|---|---:|---|
| NoHistory | `no_history` | 丢弃 | 0 | 没有长期历史的下界 |
| Direct | `direct` | raw-image fair 线保留每帧 4x4 空间 token；embedding fair 线直接保留每帧全局 token；旧 strict Robomimic 为 4 帧 | MIKASA 512；raw MoMaRT 1024；embedding MoMaRT 16；旧 Robomimic Square 132、Transport 260 | 不压缩的高带宽历史是否有效 |
| Direct2 | `direct_token2` | 与 Direct 同一历史，使用无参数 adaptive average pooling | 2 | 相同 token 数下，简单池化能否替代 Q-Former |
| VPWEM | `vpwem` | 全局 embedding 写入 FIFO cache，再经 2-layer Q-Former | 2 | 学习式固定容量记忆是否优于无历史和简单压缩 |

注意：当前 fair-history 实现的 Direct 不含旧文档所写的独立 proprio token，所以 MIKASA 两相机为 `16帧 x 2相机 x 4x4 = 512` 个历史 token，不是旧 Robomimic strict 的 132/260 token 口径。旧 strict 代码和新 fair-history 代码不能混成同一个 Direct。

## 三、已经跑过什么，哪些有效

### 3.1 MIKASA ShellGameTouch fair-history 600k

| 方法 | 训练 | 评测 | Success | 对 NoHistory | 对有效 VPWEM 诊断值 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| NoHistory | 600k，seed100 | 100 paired episodes | 88% | 0 | -4pp | B：有效单 seed |
| Direct | 600k，seed100 | 100 paired episodes | 95% | +7pp | +3pp | B：有效单 seed |
| Direct2 | 600k，seed100 | 100 paired episodes | 94% | +6pp | +2pp | B：有效单 seed |
| VPWEM，BHR 原评测 | 600k，seed100 | 100 paired episodes | 29% | -59pp | 不适用 | D：reset bug，无效 |
| VPWEM，BHR reset-fixed | 600k，seed100 | 100 paired episodes | 92% | +4pp | 0 | B：有效单 seed |
| VPWEM，本机 reset-fixed | 同一 checkpoint | 100 paired episodes | 92% | +4pp | 0 | C：独立诊断，与BHR正式值一致 |

BHR 四个有效结果的 paired exact McNemar 检验：Direct 对 VPWEM `p=0.4531`，Direct2 对 VPWEM `p=0.625`，NoHistory 对 VPWEM `p=0.3438`；Direct 对 NoHistory `p=0.0654`，Direct2 对 NoHistory `p=0.0703`，Direct 对 Direct2 `p=1.0`。全部未达到0.05。

### 3.2 MoMaRT 十配置多任务

| Checkpoint | VPWEM | Direct | Direct2 | NoHistory | 当前能否严格比较 |
|---|---:|---:|---:|---:|---|
| 400k | 63.67% | 54.33% (`-9.34pp`) | 68.00% (`+4.33pp`) | 未评 | 否：缺 NoHistory，且 VPWEM/Direct encoder 与 batch 不同 |
| 1M | 65.33% | 988k 在训（最新保存900k），尚无结果 | 71.67% (`+6.34pp`) | checkpoint 完成，正在评 | 否：矩阵未齐且架构混杂 |

括号内是相对同表 VPWEM 的百分点。400k Direct2 对 Direct 为 `+13.67pp`；从 400k 到 1M，VPWEM 增加 `1.66pp`，Direct2 增加 `3.67pp`。这些都是描述性数字，不是 memory 机制因果效应。

各方法每个 checkpoint 都是 10 个任务配置、每配置 30 episodes，即 300 episodes/方法。所有现有结果只有训练 seed100。

### 3.3 Robomimic

| 实验 | Square/ph | Square/mh | Transport/ph | Transport/mh | 判定 |
|---|---:|---:|---:|---:|---|
| NoHistory strict | 1M | 1M | 1M | 1M | D：训练完成，无 result |
| Direct strict | 900k | 900k | 400k | 400k | D：未完成，无 result |
| Direct2 strict | 未跑 | 未跑 | 未跑 | 未跑 | 缺失 |
| VPWEM strict 1M 参考 | 90.0% | 83.3% | 80.0% | 63.3% | seed100 有效参考，宏平均 79.2% |
| 旧非 strict Direct 400k | 63.3% | 20.0% | 33.3% | 13.3% | C：宏平均 32.5%，与 VPWEM 1M 不同预算 |

旧 Direct 400k 相对 VPWEM strict 1M 宏平均表面为 `-46.7pp`，但同时改变训练预算、数据入口和历史协议，禁止当作正式差距。

### 3.4 其他已跑但不应进入主表的结果

- MIKASA RememberColor3 旧 Direct：20k/100k/400k 分别为 3.3%/10.0%/6.7%，600k checkpoint 未评；它使用 4-frame stride1 旧协议，不是当前 16-frame fair-history。
- RememberColor5 和旧 ShellGameTouch Direct 各有 400k checkpoint，但没有同轮正式 result。
- MIKASA-All Direct2 的 BS256/128 可行性运行失败或停止；当前 BS64 正在训练。这是吞吐/显存工程测试，不是性能结果。
- MoMaRT embedding fair-history 只有 VPWEM 1M checkpoint；Direct、Direct2、NoHistory 只在 step0 后退出，所以没有四方法对照。

## 四、超参数是否一致

### 4.1 MIKASA fair-history：方法间基本一致，但不等于论文设置

| 项目 | NoHistory | Direct | Direct2 | VPWEM | 论文 DP-VPWEM |
|---|---:|---:|---:|---:|---:|
| training seed | 100 | 100 | 100 | 100 | 100/200/300 |
| optimizer steps | 600k | 600k | 600k | 600k | 600k |
| batch size | 16 | 16 | 16 | 16 | **64** |
| diffusion layers | 8 | 8 | 8 | 8 | 8 |
| embedding dim | 256 | 256 | 256 | 256 | 256 |
| obs/action/horizon | 2/8/9 | 2/8/9 | 2/8/9 | 2/8/9 | L=2、H=8 |
| DDPM sample steps | 50 | 50 | 50 | 50 | 50 |
| learning rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| dropout | 0.3 | 0.3 | 0.3 | 0.3 | memory dropout 0.3 |
| history ratio/max frames | 5/16 | 5/16 | 5/16 | 5/16 | ratio 5；cache max 8 |
| history output tokens | 0 | 512 | 2 | 2 | 2 |

结论：四方法内部共享的训练超参一致，能做 seed100 系统比较；但 batch16 与论文 batch64 不一致，且只有一个训练 seed，因此不能称为论文配置复现。

### 4.2 MoMaRT：Direct 三法内部一致，和 VPWEM 不一致

| 项目 | NoHistory/Direct/Direct2 | 当前表中的 VPWEM | 论文 |
|---|---:|---:|---:|
| 输入 | raw RGB/depth/proprio/scan | 预计算 `dp_emb` + task-specific context checkpoint | frozen short-context visual encoder |
| effective batch | 64 | 256 | 64 |
| physical batch | 64 / 32 / 32 | 128 | 未限定 |
| optimizer steps | 1M | 1M | 1M |
| layers / dim | 8 / 256 | 8 / 256 | 8 / 256 |
| history ratio/max frames | 20/16 | 20/16 | ratio 20、cache max 8 |
| train seed | 100 | 100 | 100/200/300 |

Physical batch 不同本身可以由梯度累积补偿；真正不可忽略的是 effective batch 64 对 256，以及 raw-image 与 embedding policy 的不同。现有 MoMaRT 表只能比较完整系统，不能写“Direct2 的 2-token pooling 优于 Q-Former”。

### 4.3 Robomimic strict：主训练超参接近，但历史协议不一致

共同项是 seed100、8-layer、1M 目标、effective batch64、obs/action/horizon=2/8/9、50 DDPM steps、lr=1e-4、crop 76x76。不同项如下：

| 项目 | NoHistory | Direct strict | VPWEM strict |
|---|---|---|---|
| sampler/history timestamps | 普通无历史入口 | stride1、anchor+recent3 | stride20、FIFO8 |
| max history frames | 0 | 4 | cache 8 |
| history tokens | 0 | Square 132、Transport 260 | 2 |
| 当前状态 | 四任务 1M | 900k/400k | 四任务 1M 参考 |

所以它们能比较完整系统，却不能把差距单独归因于 Q-Former；正式矩阵必须统一 history timestamps 和 sampler，并增加 Direct2。

## 五、与论文的比较

论文 Table I 的统一主设置为 DDPM、50 sampling steps、action chunk 8、8-layer、dim256、L2S2、Q-Former 2-layer、cache8、batch64、lr1e-4、EMA 0.999、MIKASA 600k、其他 benchmark 1M、训练 seeds 100/200/300。

| Benchmark | 论文 DP-VPWEM | 当前最可比 VPWEM | 差值 | 边界 |
|---|---:|---:|---:|---|
| MIKASA ShellGameTouch | 约91% | BHR reset-fixed 92% | +1pp | 当前 batch16、单 seed；不是论文三seed设置 |
| MIKASA RememberColor3 | 约82% | 旧 VPWEM 三 seed 约2.22%；旧 Direct 最好10% | -79.78pp / -72pp | 复现失败；旧 Direct 非 fair 协议 |
| MoMaRT 十配置平均 | DP-VPWEM 69%；MaIL-VPWEM 71% | embedding multitask VPWEM 65.33% | -3.67pp | 论文逐任务三 seed；当前多任务单 seed |
| MoMaRT 十配置平均 | 同上 | raw Direct2 71.67% | 对 DP-VPWEM +2.67pp | 与 MaIL-VPWEM 数值接近不代表同方法 |
| Robomimic 标准任务平均 | 约72% | strict seed100 参考 79.2% | +7.2pp | 当前仅单 seed；新三法尚未评 |

论文没有报告这里定义的 Direct2，也没有与当前四方法完全相同的公平矩阵。因此“与论文比”主要用于检查 VPWEM 复现是否合理，不用于证明 Direct/Direct2 超过论文。

## 六、完整执行计划表

### P0：不扩新矩阵，先消费现有 checkpoint

| ID | 工作 | 新训练 | 评测量 | 完成标准 |
|---|---|:---:|---:|---|
| P0-01 | **已完成**：BHR 重评 ShellGameTouch VPWEM 600k | 否 | 100 | 92/100；online/EMA memory 每批 reset；逐 episode结果已保存 |
| P0-02 | 完成 MoMaRT NoHistory 1M | 否 | 300 | 按训练架构走纯 DPPTP（obs=2、Tm=0、无 memory）；10/10 tasks、seed20260917；GPU2 已验证 rollout |
| P0-03 | 收尾 MoMaRT Direct 1M 并评测 | 少量续训 | 300 | 1M checkpoint + 10/10 results |
| P0-04 | Robomimic NoHistory strict 1M 四任务 | 否 | 400 | 每任务 100 paired episodes |
| P0-05 | Robomimic Direct 现有节点快速 gate | 否 | 120 | 4任务x30；宏平均达到60%才续训 |
| P0-06 | MoMaRT NoHistory 400k | 否 | 300 | 与现有 400k 相同入口和 seed |
| **剩余合计** |  |  | **1,420 episodes** | 先完成再启动 P1 |

### P1：构造真正可归因的 seed100 主矩阵

| ID | Benchmark | 训练矩阵 | 正式评测 | 关键控制 |
|---|---|---:|---:|---|
| P1-01 | MoMaRT embedding | 缺 3 个：NoHistory/Direct/Direct2；复用 VPWEM | 4方法x10任务x100 = 4,000 episodes | 同 `dp_emb`、effective batch256、1M、同任务顺序 |
| P1-02 | MIKASA ShellGameTouch | batch64 下四方法x3 seeds = 12 runs | 1,200 episodes | 同 history manifest；P0-01 已完成 |
| P1-03 | Robomimic 四标准任务 | 四方法x4任务xseed100 = 16 runs | 1,600 episodes | 同 timestamps/sampler、8-layer、batch64、1M |

Robomimic 是否立刻跑满 16 个训练，应由 P0-05 gate 决定。如果 Direct 在现有 checkpoint 上宏平均低于60%，先只实现统一协议的 100k 小试和 Direct2，不直接消耗完整 1M 预算。

### P2：统计和效率闭环

只有 seed100 通过预注册 gate 的 benchmark 才扩 seed200/300。正式输出必须同时包含：

- 三训练 seed 的 mean±std；
- 每 seed 固定 100 个相同 episode 的 Wilson 95% CI；
- 同 episode 的 McNemar 或 paired bootstrap；
- 固定 1M 主 checkpoint，附 400k/600k/800k/1M 曲线，禁止事后挑最好点；
- 参数量、history token 数、峰值显存、训练 step/s、单步推理时间；
- checkpoint SHA256、代码 commit/diff hash、Hydra config、数据 manifest hash。

## 七、结果管理系统

本次新增以下可执行登记系统：

- `experiments/history4/registry.json`：唯一事实源，保存方法定义、全部已跑结果、有效性等级、证据路径和执行计划；
- `tools/history4_report.py`：校验登记表并生成 dashboard/CSV；
- `reports/history4/status.csv`：可筛选的结果总表；
- `reports/history4/plan.csv`：可筛选的执行队列；
- `reports/history4/README.md`：自动生成的当前 dashboard。

更新命令：

```bash
python3 tools/history4_report.py
python3 tools/history4_report.py --check
```

有效性固定分四级：A=统一协议、有效 evaluator、三训练 seed；B=统一协议但单 seed；C=有真实数字但有架构/数据/预算混杂；D=evaluator 无效、缺结果或未完成。任何 `result.npy` 的存在都不能自动升级有效性。

## 八、立即决策

1. 保留 MIKASA Direct 95%、Direct2 94%、NoHistory 88% 为有效单 seed；删除所有将 VPWEM 29% 当正式成绩的图表。
2. VPWEM 92% 标为 BHR `reset-fixed` 有效单-seed结果；Direct `+3pp`、Direct2 `+2pp` 均不显著，不能写成胜出。
3. MoMaRT 71.67% 只标“raw Direct2 multitask diagnostic”，不写成优于 VPWEM 或达到 MaIL-VPWEM。
4. Robomimic 先评现有 checkpoint，不先续训；旧 Direct 32.5% 不进入 strict 主表。
5. 下一次汇报直接从 `reports/history4/README.md` 和两个 CSV 取数，禁止从旧文档手工拼表。
