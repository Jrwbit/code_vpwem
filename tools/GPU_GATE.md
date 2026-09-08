# GPU Gate：共享服务器资源监控与自动接入

这台机器没有 Slurm/PBS 等调度器，因此普通进程没有“预计完成时间”或“待运行队列”可查。`gpu_gate.py` 能查看当前所有用户的 GPU 计算/图形进程、GPU/CPU 内存和已运行时间；它还给通过本工具提交的实验建立一个轻量队列，并在资源持续满足要求后自动启动。

## 查看状态

```bash
python3 tools/gpu_gate.py status
watch -n 10 python3 tools/gpu_gate.py status
```

输出中的“剩余=未知”不是监控缺失：Linux 和 NVIDIA 驱动不知道任意训练程序还要迭代多少步。通过本工具启动时传入 `--estimate`，状态页会显示预计结束时间。

## 等空闲后自动跑

例如，任意 GPU 空闲显存至少 18 GiB、利用率不超过 10%，连续三次采样满足条件后运行实验：

```bash
nohup python3 tools/gpu_gate.py run \
  --gpu auto \
  --min-free 18432 \
  --max-util 10 \
  --stable 3 \
  --poll 30 \
  --estimate 12h \
  -- python pipelines/longmemdpptp_robomimic_image_emb.py seed=100 \
  > logs/gpu_gate_seed100.log 2>&1 &
```

也可限定候选 GPU：`--gpu 0` 或 `--gpu 0,2,3`。工具在启动子进程前设置 `CUDA_VISIBLE_DEVICES`；程序内部看到的所选卡编号会变成 `cuda:0`，这是正常行为。

在服务器1的 VPWEM 实验中不要使用 `--gpu auto`。当前授权范围固定为物理 GPU `1,2,7`，并且每条长队列应使用单个编号（`--gpu 1`、`--gpu 2` 或 `--gpu 7`）永久绑定；不要在队列阶段之间重新选卡。

## 连续队列与持有权

资源阈值只用于队列第一次启动。任务一旦获得 GPU 并开始运行，`gpu-gate` 会在整个子进程生命周期持有该卡的软件锁，不会因为其他用户后来并卡而暂停或退出。

由多个训练、评测命令组成的 shell 队列应 source `tools/gpu_queue_policy.sh`，并在每个阶段前调用 `wait_for_gpu_turn "$gpu"`。第一次调用等待空闲；第一次成功后，该队列获得持有权，后续调用只记录当前显存压力并立即继续，不会重新等待 GPU 全空。

这不是显存隔离：其他用户仍可能绕过工具直接并卡。如果因此发生真实 OOM，实验脚本仍需依靠 checkpoint 和“结果存在则跳过”实现恢复；调度器不会主动停止已经启动的实验。

## 边界与注意事项

- GPU 锁只对同样使用 `gpu_gate.py` 的任务有效。其他用户直接执行训练命令时仍可能在检测与启动之间抢走显存；严格的多人排队需要管理员部署 Slurm。
- `/tmp/gpu-gate` 是全机队列目录。正常 Linux 的 `/tmp` 可能在重启时清空，所以等待命令建议配合 `nohup`、`tmux` 或 systemd 使用。
- `--estimate` 只用于展示预计结束时间，不会到点终止实验。
- 查看其他用户的命令行受 `/proc` 挂载权限影响；权限不足时仍可看到 NVIDIA 驱动公开的 PID 和显存。
