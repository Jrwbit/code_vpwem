# MoMaRT 新检测框带框 No-History 250k 两次复测

2026-10-28，北京时间。按用户要求在物理卡3、4各运行一次250k复测，判断跨评测seed的成功率波动。当前为启动记录，不是完成报告。

| 轮次 | checkpoint | GPU | eval seed | 并行环境 | 回合数 |
|---|---|---|---|---|---|
| 既有基准 | 250000 | 7 | 20260917 | 10 | 300，194成功（64.67%） |
| 新复测1 | 250000 | 3 | 20260918 | 10 | 10配置×30，运行中 |
| 新复测2 | 250000 | 4 | 20260919 | 10 | 10配置×30，运行中 |

“新检测框”沿用既有250k评测的冻结RGB-only combined_blend在线检测/跟踪，bundle：`/home/jrw/vpwem_outputs/bbox_strategies_20261020/comparison_v2/detector_bundle.json`。track=combined，max_age=3，refine_alpha=0.5，use_roi=true。不修改检测器、不重新训练。训练仍为seed100、原离线训练框；策略执行块保持8。此次增加两个独立评测seed，不是新增训练seed，也不是同seed确定性复现。

代码目录：`/home/jrw/code_vpwem_boxed_continue_20261027`。

训练目录：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100`。

输出：训练目录下 `eval_online_combined_blend_250k_repeat_seed20260918_20261028` 和 `eval_online_combined_blend_250k_repeat_seed20260919_20261028`。各自保留队列日志、gpu*_status.tsv、gpu*_success_rates.tsv、逐配置result.npy。

启动脚本：`tools/run_momart_boxed_250k_repeats_20261028.sh`，服务器副本在 `/home/jrw/vpwem_jobs/`。通过bash -n，分别传3和4启动。GPU gate检查空闲后运行，评测共享锁避免重复占用。启动前卡3、4显存各3MiB、利用率0%。队列PID为2938821、2938824。

完成后应核对每轮10份有效结果、每份30回合、checkpoint和eval_seed一致，再比较三轮总成功率及逐配置差值。跨评测seed的差异不能解释为重新训练后的模型方差。
