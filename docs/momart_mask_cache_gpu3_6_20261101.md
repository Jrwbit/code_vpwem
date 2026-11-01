# MoMaRT 新 Mask 共享缓存：BHR GPU3–6 启动记录

2026-11-01 19:55（北京时间），按用户指令在 BHR 物理 GPU3、4、5、6 启动四个持久预处理进程。启动前四卡显存均不超过6MiB，未停止其他用户或既有评测进程。仅构建供 A1/A2/A3 共享的全量对象缓存，不启动策略训练。此前“半天到一天”为无全量测速的预留估计，现由下述实测替代。

## 工作分配和首批证据

| 物理GPU | 数据集队列 | 总帧数 | worker PID | 首批已提交帧 | 早期速度（帧/秒） |
|---|---|---:|---:|---:|---:|
| 3 | cleanup_to_dishwasher suboptimal；setup_from_dresser expert | 157867 | 758477 | 1408 | 47.39 |
| 4 | cleanup_to_sink suboptimal、expert | 139136 | 758478 | 1472 | 48.27 |
| 5 | unload_dishwasher_to_dresser suboptimal；setup_from_dresser suboptimal；setup_from_dishwasher expert | 178055 | 758482 | 1344 | 44.79 |
| 6 | cleanup_to_dishwasher expert；unload_dishwasher_to_dresser expert；setup_from_dishwasher suboptimal | 179402 | 758483 | 1344 | 44.96 |

总计10数据集、654460帧，无重复数据集。上表简称的完整名称见运行目录 `plan.json`。快照为19:55:48左右：全部四个状态均为 `perception`，合计5568个已提交帧；不是缓存完成。首批四卡显存约843–884MiB。supervisor PID为758235，运行时 `status.json` 为 `stage=preparing`、`policy_training_started=false`。

按当前首批速度，最慢队列的纯感知部分约1.1小时。此估算未覆盖CPU因果时序回放、压缩写盘及吞吐波动；完整缓存完工时间需继续根据进度修正，不再沿用“半天到一天”，也不把纯推理ETA当作完整缓存ETA。

## 协议与并行方式

生产入口仍为 `momart_object_memory.prepare_train --prepare-only`。新增 `prepare_multigpu` 只按数据集长度做最长优先分配；每卡单进程，完整数据集独占SQLite/Zarr写入，进程使用独立work/status/log。数据集内按原顺序生成感知结果并执行全部训练相位的因果回放，A1/A2/A3共用一份缓存。

使用原BHR数据，核对十数据集的episode名单、边界、物理帧映射和帧数与已打包清单一致；没有迁移数据，没有重新声称全数组哈希复算。启动前核对包内输入清单哈希、15个生产源码哈希、冻结Mask/ROI权重哈希及感知协议。沿用生产验收的 `detector_compare` 环境（torch2.6.0+cu118、torchvision0.21.0+cu118），不使用本机5090产生的异版本缓存。

所有子进程均显式使用 `--prepare-only`。一个worker失败时其他独立worker继续，supervisor等待所有worker退出后将失败退出码写入状态。单卡中断后可使用同一配置恢复，SQLite按已提交连续帧、对象Zarr按episode边界续跑；禁止同时启动同一队列的两个writer。

本次启动不表示策略训练已开始，不表示Gate B或策略任务成功率通过。

## 路径与恢复

- 代码：`/home/jrw/code_vpwem_momart_association_20261030`。
- Python：`/home/jrw/vpwem_envs/detector_compare/bin/python`。
- 运行目录：`/home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101`。
- 缓存目录：`/home/jrw/vpwem_data/momart_objects_mask8_multigpu_20261101`。
- 总日志：运行目录加 `.log`；分卡日志为 `gpu3.log` 至 `gpu6.log`。
- 分卡状态：运行目录下 `gpu3/pipeline_status.json` 至 `gpu6/pipeline_status.json`；配置为 `gpu3.yaml` 至 `gpu6.yaml`。
- 完成判定：supervisor正常退出且 `stage=cache_complete`，并核查10个对象Zarr的complete标记、completed_frames与感知/stream签名。当前尚未完成，不宣称这项末尾核验已执行。

仅在对应worker已退出后，以下示例可恢复GPU3队列；其余卡对应替换编号：

```bash
cd /home/jrw/code_vpwem_momart_association_20261030
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/jrw/vpwem_envs/detector_compare/bin/python -u \
  -m momart_object_memory.prepare_train \
  --config /home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101/gpu3.yaml \
  --device cuda:0 --prepare-only
```

## 验证与Git

本机及BHR各运行2项针对性检查，均通过：分配覆盖每个数据集一次、总帧数和负载、重复GPU拒绝、子进程强制prepare-only、失败状态记录。`git diff --check` 通过。启动器提交 `e0b1a43`，生产感知/缓存源码未更改。数据、权重、缓存、状态及运行日志不进入Git。

飞书已追加并回读核验：[GPU3–6启动记录](https://ccnuoe6sry36.feishu.cn/docx/RfXudCiBOoeG3Bx0D34cbU1rn0g#doxcniYqEXiT9NDAfIb558CfGrg)。

补充：19:56:50最后一次成功日志读取，GPU3–6分别已提交4480、4480、4032、4288帧，合计17280帧，速度约48.45、48.57、43.98、46.29帧/秒。随后连续两次SSH返回 `Connection refused`，无法继续核验远端最新状态。启动已用nohup/setsid持久化，但SSH断连本身不能证明任务仍在运行或已经停止；不能把最后成功快照写成持续实时状态。
