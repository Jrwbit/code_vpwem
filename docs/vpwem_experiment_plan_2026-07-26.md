# VPWEM 实验训练计划与结果盘点

更新时间：2026-07-26 22:06（Asia/Shanghai）

## 结论

- 本机没有训练进程，但远端 `bhr-server`（8×RTX 4090D）有 VPWEM 任务运行。远端状态与 ETA 见下文。
- 论文核心矩阵包含 17 个任务配置：MIKASA 2 个、MoMaRT 10 个、Robomimic 5 个。每个配置按 3 个随机种子（100/200/300）计算，共 51 个 VPWEM 正式训练。
- 仓库扩展矩阵另有 MIKASA 2 个、Robomimic ToolHang、RMBench 和 Real-Zarr，共 5 个配置。
- 本机已完成 Robomimic 5 个配置的三 seed 训练与评估；这些 checkpoint 中四个标准任务均为 4-layer diffusion，而论文 Table I 为 8-layer，因此只能称为本地变体，不能称为严格论文配置复现。其中 LongSquare 的 4-layer 三 seed 结果只有 `1.11% +/- 1.92%`，唯一 8-layer seed 100 也为 0%，与论文主配置约 70% 严重不符，应视为失败/回归。
- RMBench 的 SwapBlocks 已有 95k checkpoint，但现有所有正式/调试评估记录均为 0%。
- Real-Zarr 已跑完 160k（约 7.14 小时），但目录中只有 `scheduler_160000.pt`，缺少 `model_160000.pt`，所以当前 checkpoint 不可用于部署。
- MIKASA 的论文口径 8-layer ShellGameTouch 三 seed 已在远端训练至 600k，但尚未评估；RememberColor3 的 seed 100/200 正在训练，seed 300 已排队、尚未启动。训练池完成后会顺序评估全部 6 个 MIKASA 运行。
- MoMaRT 的 10 份原始 hdf5 均在远端；当前两个 unload seed-100 VPWEM 在训。10 个任务现已全部被当前队列或修正后的重试队列覆盖。

## 远端服务器实时状态

快照时间：2026-07-26 22:06。ETA 根据 12 秒内实际 step 增量估算；共享 GPU、I/O 或任务重启会改变 ETA。

| Bench | 当前运行 | 进度 | 实测速率 | 预计训练完成 |
|---|---|---:|---:|---|
| MIKASA | RememberColor3 8-layer seed 100 | 493k / 600k（82.17%） | 1.789 step/s | 2026-07-27 14:44 |
| MIKASA | RememberColor3 8-layer seed 200 | 367k / 600k（61.17%） | 1.802 step/s | 2026-07-28 10:03 |
| MoMaRT | unload expert 8-layer VPWEM seed 100 | 127k / 1M（12.70%） | 8.631 step/s | 2026-07-28 02:13 |
| MoMaRT | unload suboptimal 8-layer VPWEM seed 100 | 103k / 1M（10.30%） | 7.000 step/s | 2026-07-28 09:43 |

同时运行的 MoMaRT 短上下文前置训练有 6 个：dresser setup expert/suboptimal、dishwasher cleanup expert/suboptimal、sink cleanup expert/suboptimal。

队列覆盖情况：

- MIKASA：6/6 正式运行均在同一 supervisor 队列中；当前 2 个训练、1 个等待，之后自动开始 6 个顺序评估。
- MoMaRT unload expert/suboptimal：seed 100 已启动成功；seed 200/300 和三 seed 评估在各自 worker 后续队列中。
- MoMaRT sink cleanup expert/suboptimal：短上下文已启动；embedding、VPWEM seed 100/200/300、评估均在 GPU1 队列中。
- MoMaRT dresser setup 与 dishwasher cleanup 共 4 个配置：短上下文已启动；修正后的 GPU7 retry supervisor 正在等待旧队列结束，之后会接管 VPWEM 三 seed和评估。
- MoMaRT dishwasher setup expert/suboptimal：embedding 已完成，VPWEM 曾因缺少 `+inference_action_steps=8` 启动失败。2026-07-26 22:20 已创建独立的 GPU1 retry 队列；它会等待当前 GPU1 sink 队列结束，再自动运行两任务的 seed 100/200/300 和评估。

## 估时口径

- 硬件口径：当前单张 RTX 5090 24 GB。
- “单 seed”包含 VPWEM 训练，不包含短上下文 DP-PTP 预训练和 embedding 重写。
- “三 seed”按单 GPU 串行估算；有 3 张同等 GPU 时，墙钟时间接近单 seed。
- Robomimic、RMBench、Real-Zarr 使用本机实测；MIKASA 已改用远端日志实测；MoMaRT 同时参考论文 `0.09 s/step × 1,000,000 step ≈ 25 h` 和远端实时速率。
- 从零开始时还需要短上下文策略训练和 embedding 重写，整体预算通常为表中 VPWEM 阶段的约 1.8–2.2 倍。

## 论文核心实验矩阵

| Bench | 任务配置 | Steps | 单 seed 训练 | 三 seed 串行 | 评估耗时（每 seed） | 当前本机状态 | 结果 |
|---|---|---:|---:|---:|---:|---|---|
| MIKASA | ShellGameTouch-v0 | 600k | 20–21 h（远端实测） | 已并行完成 | 10–30 min* | 8-layer 三 seed 训练完成，待评估 | 旧 200k 变体四 seed：`94.17% +/- 6.87%`；论文 91% |
| MIKASA | RememberColor3-v0 | 600k | 约 93 h（远端独占进程时间） | 三 seed 已排队 | 10–30 min* | seed 100/200 在训；300 等待 | 旧 200k 变体四 seed：`7.50% +/- 3.19%`；论文 82% |
| MoMaRT | table_cleanup_to_dishwasher / expert | 1M | 24–40 h** | 72–120 h | 1–3 h* | 原始数据有；短上下文在训 | 论文图约 65% |
| MoMaRT | table_cleanup_to_dishwasher / suboptimal | 1M | 24–40 h** | 72–120 h | 1–3 h* | 原始数据有；短上下文在训 | 论文图约 37% |
| MoMaRT | table_cleanup_to_sink / expert | 1M | 24–40 h** | 72–120 h | 1–3 h* | 原始数据有；短上下文在训 | 论文图约 82% |
| MoMaRT | table_cleanup_to_sink / suboptimal | 1M | 24–40 h** | 72–120 h | 1–3 h* | 原始数据有；短上下文在训 | 论文图约 27% |
| MoMaRT | table_setup_from_dishwasher / expert | 1M | 24–40 h** | 72–120 h | 1–3 h* | embedding 已有；VPWEM 启动失败 | 论文图约 88% |
| MoMaRT | table_setup_from_dishwasher / suboptimal | 1M | 24–40 h** | 72–120 h | 1–3 h* | embedding 已有；VPWEM 启动失败 | 论文图约 80% |
| MoMaRT | table_setup_from_dresser / expert | 1M | 24–40 h** | 72–120 h | 1–3 h* | 原始数据有；短上下文在训 | 论文图约 95% |
| MoMaRT | table_setup_from_dresser / suboptimal | 1M | 24–40 h** | 72–120 h | 1–3 h* | 原始数据有；短上下文在训 | 论文图约 72% |
| MoMaRT | unload_dishwasher_to_dresser / expert | 1M | 当前 seed-100 ETA 约 28 h | 三 seed 已排队 | 1–3 h* | seed 100：127k，在训 | 论文图约 77% |
| MoMaRT | unload_dishwasher_to_dresser / suboptimal | 1M | 当前 seed-100 ETA 约 36 h | 三 seed 已排队 | 1–3 h* | seed 100：103k，在训 | 论文/表：58.3%（主对比设置） |
| Robomimic | Square / ph | 1M | 8–9 h | 25–27 h | 2–5 min | 已完成 3 seed | 本机 `94.44% +/- 1.92%` |
| Robomimic | Square / mh | 1M | 7–11 h | 21–33 h | 2–5 min | 已完成 3 seed | 本机 `76.67% +/- 3.33%` |
| Robomimic | Transport / ph | 1M | 8–11 h | 24–33 h | 5–15 min | 已完成 3 seed | 本机 Q-Former 2-layer 一致口径：`90.00% +/- 5.77%` |
| Robomimic | Transport / mh | 1M | 10–12 h*** | 30–36 h | 10–20 min | 已完成 3 seed | 本机 `36.67% +/- 6.67%` |
| Robomimic | LongSquare / ph | 1M | 8–11 h | 24–33 h | 2–5 min | 已完成但结果无效 | 本机 `1.11% +/- 1.92%`；论文主 L2S2 约 70% |

\* 尚无完整评估样本，是计划估计，不是实测 ETA。  
\** 论文理论值约 25 h；远端当前两个 unload 任务共享资源后的外推总时长约 32–40 h。  
\*** 两个 Transport/mh seed 曾并发争抢资源，各自墙钟约 33 小时；正常独占 GPU 的 seed 100 为 10.95 小时，因此计划采用 10–12 小时。

论文汇总结果：MoMaRT 的 DP-VPWEM 平均约 69%，MaIL-VPWEM 平均约 71%；Robomimic 的 DP-VPWEM 平均约 72%。当前仓库没有 MaIL-VPWEM 的训练 pipeline，只有普通 MaIL 和 DP-VPWEM，因此 MaIL-VPWEM 不能直接纳入可执行队列。

## 仓库扩展实验

| Bench | 任务配置 | Steps | 单 seed 训练 | 三 seed 串行 | 当前状态 | 当前结果 |
|---|---|---:|---:|---:|---|---|
| MIKASA | RememberColor5-v0 | 配置不一致：200k/600k emb | 约 54 h（按远端 124k 外推） | 约 162 h | 旧任务停在 118k–124k | 无 |
| MIKASA | InterceptGrabMedium-v0 | 配置不一致：200k/600k emb | 6–10 h* | 18–30 h | 未训练 | 无 |
| Robomimic | ToolHang / ph | 1M | 9–12 h* | 27–36 h | 未训练 | 无 |
| RMBench | SwapBlocks / demo_clean | 当前正式运行 80k–95k | 15–25 min（优化后实测） | 45–75 min | checkpoint 已有 | 所有现有评估记录均为 0% |
| Real-Zarr | block_2cam | 160k | 7.14 h（实测） | 21–23 h | 训练日志完成但模型文件缺失 | 无自动/实机成功率 |

## 本机 Robomimic 结果明细

所有结果均为每个 seed 30 episodes，表中为 success rate。

| 任务 | Seed 100 | Seed 200 | Seed 300 | Mean +/- sample std | Mean reward |
|---|---:|---:|---:|---:|---:|
| Square/ph | 93.33% | 96.67% | 93.33% | `94.44% +/- 1.92%` | 41.72 |
| Square/mh | 73.33% | 76.67% | 80.00% | `76.67% +/- 3.33%` | 26.09 |
| Transport/ph | 93.33% | 83.33% | 93.33% | `90.00% +/- 5.77%` | 29.19 |
| Transport/mh | 30.00% | 36.67% | 43.33% | `36.67% +/- 6.67%` | 7.52 |
| LongSquare/ph | 0.00% | 0.00% | 3.33% | `1.11% +/- 1.92%` | 0.13 |

Square/ph 和 Transport/ph 的 seed 100 均采用与 seed 200/300 一致的 Q-Former 2-layer 运行。旧表把 Transport/ph 的 Q-Former 1-layer seed 100（83.33%）混入汇总，现已改用 `transport_ph/vpwem_reproduction_qformer2/100`（93.33%）。上述标准任务 checkpoint 仍为 4-layer diffusion，不能作为论文 8-layer 设置的严格复现结果。

## 推荐执行顺序与总工期

以下为单张 RTX 5090、已有短上下文 checkpoint 时的墙钟计划；`T` 表示真正启动队列的时间。

| 阶段 | 内容 | 完成窗口 | 验收条件 |
|---|---|---|---|
| P0 | 修复/验证 Real-Zarr checkpoint 保存；对 LongSquare 做 100-step smoke + 1 个短评估 | T + 0.5 天 | 模型文件存在；LongSquare 非全零动作/奖励 |
| P1 | RMBench 3 seed 训练及至少 10 episodes/seed 评估 | T + 1 天 | 输出三 seed success rate；若仍为 0%，暂停扩量 |
| P2 | LongSquare 3 seed 正式重跑 | T + 2–2.5 天 | 接近论文 70%；若仍低于 50%，转入回归诊断 |
| P3 | MIKASA：完成 RememberColor3 100/200/300，随后 supervisor 自动评估两个任务全部三 seed | 当前 100/200 预计 7 月 27–28 日完成；seed 300 随后启动，约再需 4 天 | 两任务都必须有三 seed 正式结果 |
| P4 | Robomimic ToolHang + MIKASA 两个扩展任务 | T + 6–8 天 | 三个扩展配置各有三 seed |
| P5 | MoMaRT：完成当前 6 个短上下文任务及 2 个 unload VPWEM | unload 预计 7 月 28 日完成；其余取决于前置训练 | 10 个任务均有可用 embedding |
| P6 | MoMaRT 10 任务 VPWEM；当前仅排了部分 seed 100 | seed-100 矩阵约再需 4–7 天；seed 200/300 尚未排队 | 论文 10 配置全部三 seed |
| P7 | Real-Zarr 三 seed训练和实机评估 | 可与数据等待期穿插；训练约 1 天 | 至少 20 个实机 trials，报告成功率 |

因此：

- 论文核心 VPWEM 矩阵的“VPWEM 阶段”总预算约 39 GPU-days；Robomimic 已有训练可复用后，剩余约 33–37 GPU-days。
- 加上仓库扩展实验和必要回归重跑，约 37–43 GPU-days。
- 若所有短上下文 DP-PTP、embedding 都从零重做，建议预留 70–85 GPU-days。
- 远端 MoMaRT 的 10 个原始 hdf5 已齐；当前限制是短上下文 checkpoint、embedding 生成和 seed 200/300 尚未排队，而不是缺数据。

## 开跑前必须处理的问题

1. 配置中仍有 `PATH_TO_DATASET`、`PATH_TO_SAVE_MODEL` 占位符；批量脚本中也有未定义变量和错误的绝对路径前缀。
2. Real-Zarr 完成日志有 `step=160000`、loss `0.03217`，但模型权重未保存，只保存了 scheduler。
3. LongSquare 本机评估与论文差距约 69 个百分点，需先核对数据版本、目标 peg 定义、context checkpoint、Q-Former 层数和 eval wrapper，不能直接继续做消融。
4. RMBench 现有大量评估均为 0%，应先做动作归一化/坐标系/agent-pos 对齐检查，再增加 seeds 或 episodes。
5. MIKASA 的非 embedding 与 embedding YAML 在 `gradient_steps`、`obs_steps` 等字段上不一致，正式队列应冻结为论文口径：600k steps、batch 64、3 seeds。
6. README 顶部含明文 W&B API key，应立即吊销并改用环境变量。
7. MoMaRT 的两个 dishwasher-setup VPWEM 启动曾因缺少 Hydra 覆盖 `+inference_action_steps=8` 失败；修正后的 GPU1 retry 队列已于 2026-07-26 22:20 创建，等待当前 sink 队列结束后自动接管。
