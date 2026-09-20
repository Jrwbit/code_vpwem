# MoMaRT 公平历史对比实验运行记录

> 快照时间：2026-09-14 20:48 CST  
> 服务器：`bhr-server`（RTX 4090 D）  
> 实验目录：`/home/jrw/vpwem_outputs/fair_history_momart_multitask`  
> 统一目标：1,000,000 optimizer steps，effective batch size = 64，seed = 100

## 当前结论

- 正在运行的三项实验为 `direct_token2`、`direct` 和 `no_history`。
- `no_history` 当前最快，最近约 0.195 秒/step，预计约 2.1 天后完成。
- `direct_token2` 最近约 0.492 秒/step，预计约 5.5 天后完成。
- `direct` 最近约 0.593 秒/step，预计约 6.7 天后完成。
- `vpwem` 已异常停止，当前不会自行完成；退出前只记录到 1,000 step，且尚未达到 100,000 step 的首次 checkpoint 保存点。

## 实验状态

| GPU | 方法 | Physical BS | 梯度累积 | Effective BS | 最新进度 | 最近速度 | 预计剩余 | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 3 | `direct_token2` | 32 | 2 | 64 | 32,000 / 1,000,000 | 约 0.492 秒/step | 约 5.5 天 | 运行中 |
| 7 | `direct` | 32 | 2 | 64 | 29,000 / 1,000,000 | 约 0.593 秒/step | 约 6.7 天 | 运行中 |
| 7 | `no_history` | 64 | 1 | 64 | 70,000 / 1,000,000 | 约 0.195 秒/step | 约 2.1 天 | 运行中 |
| 3 | `vpwem` | 1 | 64 | 64 | 最后记录 1,000 / 1,000,000 | 首个千步约 7.75 秒/step | 若从头重跑约 90 天 | 已异常停止 |

上述 ETA 使用各任务最近约 10,000 step 的实际耗时外推；服务器竞争、数据加载、checkpoint 写盘或后续 OOM 都可能改变完成时间。按当前速度，`no_history` 约在 9 月 16 日晚完成，`direct_token2` 约在 9 月 20 日上午完成，`direct` 约在 9 月 21 日中午完成。

## 配置与变更历史

### GPU 3

- `vpwem` 最初以 physical BS=1、gradient accumulation=64 启动。
- 运行到至少 1,000 step 后异常退出。日志末尾出现 W&B `MsgRouterThr` 的 `OSError: handle is closed`，但未留下可恢复的模型 checkpoint。
- `direct_token2` 以 physical BS=32、gradient accumulation=2 独立运行，有效 batch size 保持 64。
- GPU 3 同时还有其他用户的 Instinct Parkour 训练，占用部分显存与算力，因此 ETA 可能波动。

### GPU 7

- `no_history` 使用 physical BS=64、gradient accumulation=1，有效 batch size 64。
- `direct` 最初使用 physical BS=1、gradient accumulation=64，后按要求停止并尝试改为 physical BS=64、gradient accumulation=1。
- `direct` BS=64 与 `no_history` 同卡并行时两次在首步发生 CUDA OOM；即使启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 仍无法共存。
- 当前 `direct` 改为 physical BS=32、gradient accumulation=2，已稳定运行且有效 batch size 仍为 64。
- 已设置自动兜底：若当前 `direct` BS=32 因 CUDA OOM 退出，则自动停止 `no_history`，等待显存释放，并以 physical BS=64、gradient accumulation=1 重启 `direct`。截至本次快照，兜底尚未触发。

## 最新训练指标

| 方法 | Step | Avg diffusion loss | Task index |
|---|---:|---:|---:|
| `direct_token2` | 32,000 | 0.105864 | 0 |
| `direct` | 29,000 | 0.114796 | 0 |
| `no_history` | 70,000 | 0.114575 | 7 |
| `vpwem` | 1,000（停止前最后记录） | 0.264050 | 8 |

这些 loss 来自不同 task index，仅用于运行健康检查，不应直接当作方法最终优劣结论。正式比较应等待相同 checkpoint、相同评测任务及相同评测预算下的成功率结果。

## 后续事项

1. 关注 100,000 step 的首批 checkpoint 是否正常生成。
2. `no_history` 完成后，重新测量卡 7 上 `direct` 的单卡速度，更新 ETA。
3. 决定是否以更大的 physical batch 重启 `vpwem`；以 BS=1 重跑预计成本过高。
4. 所有方法完成后，使用相同 checkpoint 集和评测协议生成最终对比表。
