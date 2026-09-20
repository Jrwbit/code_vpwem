# MoMaRT History-4 SH2 transfer

目标根目录：

`/oss/sh2/vpwem_transfer/momart_history4_20260920`

## 目录

- `code/`：BHR 实际运行仓库的源码与 Git 历史；排除重复的数据集、旧输出、旧 artifacts 和缓存。
- `design/`：本机最新审计、设计文档、报告、实验登记表及关键配置、pipeline、启动脚本。
- `experiments/fair_history_momart_multitask/`：raw-observation Direct、Direct-token2、No-History 的 Hydra 配置、metrics、离线日志和 checkpoints。
- `experiments/fair_history_momart_multitask_emb/`：embedding VPWEM 实验及 checkpoints。
- `context_checkpoints/`：VPWEM 相关 DP-PTP/context encoder 的完整 400k–1M checkpoint 序列及 Hydra 配置；包含当前实际评测依赖的十个任务专属 8-layer 1M checkpoint。
- `evaluations/`：四方法 1M、十任务、每任务 30 episodes 的逐任务 `result.npy`、日志和代表性视频。
- `manifests/`：文件清单、大小、SHA-256 与迁移完成标记。

## 四方法口径

- VPWEM：预计算 `dp_emb` 的多任务后端；effective batch 256；Q-Former 输出 2 tokens。
- Direct-token2：raw observation 多任务策略；effective batch 64；历史自适应池化到 2 tokens。
- Direct：raw observation 多任务策略；effective batch 64；保留完整空间历史 tokens。
- No-History：raw observation 多任务策略；effective batch 64；长期历史关闭。

当前 VPWEM 与另外三法的 observation encoder 和 effective batch 不同，因此这些结果属于跨系统诊断，不是严格的 memory 单变量消融。

## 1M 最终评测结果

评测协议为训练 seed 100、eval seed 20260917、10 个任务配置、每配置 30 episodes：

| 方法 | 宏平均成功率 |
|---|---:|
| VPWEM | 65.33% |
| Direct-token2 | 71.67% |
| Direct | 72.00% |
| No-History | 72.33% |

No-History 的 GPU2 启动环境未设置 `JOB_END=5`，因此 GPU2 实际运行 0–10，GPU3 同时运行 5–10；后五项被同 checkpoint、同 seed、同 episode 数重复评测并由 GPU2 最终文件覆盖。迁移包保存的是所有进程自然结束后的最终文件。该问题不改变单次评测协议，但浪费了重复计算量，正式统计时仍按每任务最终 30 episodes 计数，不能把两次运行合并成 60 episodes。

## 数据集

训练数据本身不重复收进本迁移包。核对结果为：BHR 的完整原始 MoMaRT 数据 `/home/jrw/vpwem_data/momart` 有 10 个 HDF5、约 209G；SH2 的 `/oss/sh2/vpwem_data/momart` 当前只有其中 3 个 HDF5、约 70G。BHR 的预计算 embedding 数据 `/home/jrw/vpwem_data/momart_emb_8layer` 有 10 个 HDF5、约 210G，SH2 尚无等价目录。这约 419G 数据不属于本次“代码、实验、设计、checkpoint”迁移范围；若要求迁移包可在 SH2 独立完整重训，还需另行复制这些数据。

## 完整性

大模型文件经 BHR 直接上传 OSS；每个单 PUT 对象均返回 HTTP 200，且 OSS ETag 与源文件 MD5 一致。最终交付共约 117G；`manifests/` 内保存目标端文件清单与 checkpoint/result 的 SHA-256，并以 `TRANSFER_COMPLETE` 标记完成。

主策略 checkpoint 数量为：VPWEM 7 组（400k–1M）、Direct-token2 9 组（200k–1M）、Direct 9 组（200k–1M）、No-History 3 组（800k–1M）。此外完整迁移源端现存的 248 个 ≥400k context 文件（124 个 model、124 个 scheduler，覆盖 400k–1M；源端个别 variant 本身没有 1M）。评测结果共 40 个 `result.npy`（4 方法 × 10 任务）。

严格源—目标审计结果：主策略 ≥400k 文件为 48/48、SHA-256 差异 0；context ≥400k 文件为 248/248、路径/大小差异 0、SHA-256 差异 0。对应清单为 `manifests/strategy_ge400_sha256.tsv` 和 `manifests/context_ge400_sha256.tsv`。
