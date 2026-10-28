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

## 完成复核与GPU复用（2026-10-28 21:45）

以上为启动历史。已重新读取全部30份原始result.npy，验证每项30回合、checkpoint、eval_seed=20260917、二值成功标志及均值一致；三条状态TSV均为FINISH、failures=0、missing=0。

| GPU | checkpoint | 成功数/回合数 | 成功率 | 队列完成时间 |
|---|---|---|---|---|
| 0 | 175000 | 165/300 | 55.00% | 2026-10-28 21:36:52+08:00 |
| 1 | 200000 | 171/300 | 57.00% | 2026-10-28 21:40:19+08:00 |
| 7 | 225000 | 188/300 | 62.67% | 2026-10-28 21:36:33+08:00 |

三张卡的上述评测均完成，核查时已无对应评测进程且显存空闲。按用户“统筹安排”的要求，将已排队的250k四步评测转移到空出的0/1/7：eval seed100→GPU0、200→GPU1、300→GPU7；不再等待3/4/6上的八步队列。原结果保留，不重跑已完成的中间checkpoint。详见 `docs/momart_boxed_250k_action4_queue_20261028.md` 最新调整记录。
