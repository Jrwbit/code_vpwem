# MoMaRT 带框 No-History 175k / 200k / 225k 评测

2026-10-28 17:00（北京时间）已在 BHR 启动，当前为运行记录，尚无最终成功率。

| checkpoint | 物理 GPU | 并行环境 | 回合 |
|---|---|---|---|
| 175000 | 0 | 10 | 10 配置 × 30 |
| 200000 | 1 | 10 | 10 配置 × 30 |
| 225000 | 7 | 10 | 10 配置 × 30 |

沿用 250k 评测协议：boxed_no_history、combined_blend 在线检测/跟踪、eval seed=20260917、执行块8、无视频。训练 seed100；未重新训练。总计计划900回合。三个队列独立运行，逐任务保存 result.npy 和成功率 TSV。

训练目录：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100`。

输出目录：训练目录下 `eval_online_combined_blend_175k_200k_225k_20261028`。

评测代码：`/home/jrw/code_vpwem_boxed_continue_20261027`，复用 `tools/run_momart_boxed_checkpoint_eval_bhr.sh`。可复现启动器：本仓库 `tools/run_momart_boxed_intermediate_eval_20261028.sh`，已复制至服务器 `/home/jrw/vpwem_jobs/`，分别传入0、1、7。

核验：三个 checkpoint 文件存在；启动器通过 bash -n；三个队列均记录 START、对应 checkpoint 和 GPU，日志已进入 iGibson 环境加载。队列 PID 分别为2714402、2714403、2714404。GPU gate 在启动前检查空闲状态，评测持有共享 GPU 锁。SSH 曾短暂拒绝连接，恢复后成功启动，没有重复提交。

本记录仅确认任务已启动，不代表900回合完成或模型质量提升。完成状态以各 gpu*_status.tsv 的 FINISH 和30份有效 result.npy 为准。
