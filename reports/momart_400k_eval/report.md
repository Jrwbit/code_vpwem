# MoMaRT 多任务 400k checkpoint 评测

完成时间：2026-09-19T07:28:57+08:00

评测设置：seed=20260917；每个任务配置 30 episodes；每个 episode 最长 1504 steps；共 3 个方法 × 10 个任务配置 = 900 episodes。

注意：VPWEM 使用 embedding policy 与任务对应的 context encoder；direct 和 direct_token2 使用 raw visual policy，因此这是相同训练步数与任务协议下的结果，并非完全相同 observation encoder 架构的严格消融。

## 汇总

| 方法 | 配置数 | Episodes | 平均成功率 | 平均 reward | 平均推理时间/次 (s) |
|---|---:|---:|---:|---:|---:|
| vpwem | 10 | 300 | 0.6367 | 0.0000 | 0.3316 |
| direct | 10 | 300 | 0.5433 | 0.0000 | 0.7299 |
| direct_token2 | 10 | 300 | 0.6800 | 0.0000 | 0.3979 |

## 分任务结果

| 方法 | 任务配置 | Episodes | 成功率 | Reward | 推理时间/次 (s) |
|---|---|---:|---:|---:|---:|
| direct | table_cleanup_to_dishwasher_expert | 30 | 0.1667 | 0.0000 | 0.7299 |
| direct | table_cleanup_to_dishwasher_suboptimal | 30 | 0.1000 | 0.0000 | 0.7255 |
| direct | table_cleanup_to_sink_expert | 30 | 0.5000 | 0.0000 | 0.7365 |
| direct | table_cleanup_to_sink_suboptimal | 30 | 0.6667 | 0.0000 | 0.7349 |
| direct | table_setup_from_dishwasher_expert | 30 | 0.9000 | 0.0000 | 0.7319 |
| direct | table_setup_from_dishwasher_suboptimal | 30 | 0.8333 | 0.0000 | 0.7291 |
| direct | table_setup_from_dresser_expert | 30 | 0.7000 | 0.0000 | 0.7120 |
| direct | table_setup_from_dresser_suboptimal | 30 | 0.5000 | 0.0000 | 0.7440 |
| direct | unload_dishwasher_to_dresser_expert | 30 | 0.7333 | 0.0000 | 0.7232 |
| direct | unload_dishwasher_to_dresser_suboptimal | 30 | 0.3333 | 0.0000 | 0.7319 |
| direct_token2 | table_cleanup_to_dishwasher_expert | 30 | 0.6667 | 0.0000 | 0.3952 |
| direct_token2 | table_cleanup_to_dishwasher_suboptimal | 30 | 0.3333 | 0.0000 | 0.3970 |
| direct_token2 | table_cleanup_to_sink_expert | 30 | 0.7333 | 0.0000 | 0.4050 |
| direct_token2 | table_cleanup_to_sink_suboptimal | 30 | 0.5667 | 0.0000 | 0.3936 |
| direct_token2 | table_setup_from_dishwasher_expert | 30 | 0.8667 | 0.0000 | 0.3906 |
| direct_token2 | table_setup_from_dishwasher_suboptimal | 30 | 0.9333 | 0.0000 | 0.3936 |
| direct_token2 | table_setup_from_dresser_expert | 30 | 0.8333 | 0.0000 | 0.3992 |
| direct_token2 | table_setup_from_dresser_suboptimal | 30 | 0.6333 | 0.0000 | 0.3884 |
| direct_token2 | unload_dishwasher_to_dresser_expert | 30 | 0.7000 | 0.0000 | 0.4059 |
| direct_token2 | unload_dishwasher_to_dresser_suboptimal | 30 | 0.5333 | 0.0000 | 0.4103 |
| vpwem | table_cleanup_to_dishwasher_expert | 30 | 0.5667 | 0.0000 | 0.3318 |
| vpwem | table_cleanup_to_dishwasher_suboptimal | 30 | 0.3000 | 0.0000 | 0.3398 |
| vpwem | table_cleanup_to_sink_expert | 30 | 0.6667 | 0.0000 | 0.3303 |
| vpwem | table_cleanup_to_sink_suboptimal | 30 | 0.1667 | 0.0000 | 0.3343 |
| vpwem | table_setup_from_dishwasher_expert | 30 | 0.7667 | 0.0000 | 0.3352 |
| vpwem | table_setup_from_dishwasher_suboptimal | 30 | 0.8000 | 0.0000 | 0.3372 |
| vpwem | table_setup_from_dresser_expert | 30 | 0.9333 | 0.0000 | 0.3125 |
| vpwem | table_setup_from_dresser_suboptimal | 30 | 0.8667 | 0.0000 | 0.3465 |
| vpwem | unload_dishwasher_to_dresser_expert | 30 | 0.8000 | 0.0000 | 0.3382 |
| vpwem | unload_dishwasher_to_dresser_suboptimal | 30 | 0.5000 | 0.0000 | 0.3105 |

## 视频

共 30 个代表性 MP4，已作为飞书文档附件上传；每个方法、每个任务配置 1 个。

BHR 原始结果目录：`/home/jrw/vpwem_outputs/fair_history_momart_multitask_eval_400k`
