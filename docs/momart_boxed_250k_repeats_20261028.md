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

## 用户指定 seed 后的调整（2026-10-28 19:22）

以上20260918、20260919两轮被用户指定的新方案取代：已停止两轮旧队列及其评测进程组，保留原始日志，不将未完成结果计入比较。原文仅保留为启动历史。

最终方案是同一个250k checkpoint跑三个**评测 seed：100、200、300**；训练 seed 仍为100。seed控制评测的随机数序列，不是训练步数，也不代表模型重新训练。每轮仍为10个任务配置×30回合，每卡并行10环境，总共900回合，采用相同combined_blend新框和8步执行块。

| 最终评测seed | GPU | 排队顺序 |
|---|---|---|
| 100 | 3 | 立即启动 |
| 200 | 4 | 立即启动 |
| 300 | 3 | seed100结束后自动启动 |

三个输出目录分别为训练目录下的 `eval_online_combined_blend_250k_repeat_seed100_20261028`、`eval_online_combined_blend_250k_repeat_seed200_20261028`、`eval_online_combined_blend_250k_repeat_seed300_20261028`。启动脚本参数现为seed值100/200/300。GPU3串行执行100和300，GPU4执行200。此前卡0/1/7的175k/200k/225k评测未改动。

这里只能确认新队列已提交，成功率待完整评测后复算。

## 最新安排：先同seed三次，再不同seed（2026-10-28 19:23）

用户进一步指定先测试三次100，再跑200和300，使用卡3、4、6。此安排取代上一节：已停止刚启动的seed100/200队列和待执行300；旧输出保留，不计入本轮。此前64.67%的训练seed是100，但评测seed是20260917。

| 阶段 | 独立运行目录 | 评测seed | GPU |
|---|---|---|---|
| 1 | seed100_repeat1 | 100 | 3 |
| 1 | seed100_repeat2 | 100 | 4 |
| 1 | seed100_repeat3 | 100 | 6 |
| 2 | seed200 | 200 | 3 |
| 2 | seed300 | 300 | 4 |

三次seed100各自重新执行并保存独立结果，避免复用已有result.npy导致跳过。阶段1三个队列全部成功完成后才进入阶段2；任何队列失败则暂不启动阶段2，保留失败日志。总计五轮、1500回合，每轮10任务配置×30回合、并行10环境。权重、在线combined_blend检测框、执行块8均保持不变。前三轮比较同seed重复运行与不同GPU的执行差异；它们不是独立随机seed样本。随后200/300用于观察跨seed变化。

最终输出根目录：训练目录下 `eval_online_combined_blend_250k_seed100_triplicate_20261028`，五轮各用表中独立子目录。启动器：`tools/run_momart_boxed_250k_seed100_triplicate_20261028.sh`，服务器副本 `/home/jrw/vpwem_jobs/`。bash -n通过，supervisor PID=2944211，持有独占锁，记录阶段切换和退出状态。

当前是已提交运行计划与启动核验，尚无五轮最终成功率。之前0/1/7上的中间checkpoint评测保持运行。

## 后续新增四步队列（2026-10-28 21:23）

用户新增250k、预测9步执行4步、评测seed100/200/300各一轮。已另建协调器PID3051605，等待本页既有五轮八步队列的supervisor锁释放后再按GPU3/4/6排队；没有替换或停止本页任务。新任务共900回合，沿用本页旧tracker与计分协议，便于按执行长度对照。详情及结果入口见 `docs/momart_boxed_250k_action4_queue_20261028.md` 和[飞书排队记录](https://feishu.cn/docx/Jhhed9zpcoNFdsxS1dvcC5SFntd)；飞书正文及表格已回读核对。提交时新任务仍为waiting_predecessor，未产生成功率。
