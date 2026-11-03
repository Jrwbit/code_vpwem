# 固定模型错误占比与主视角视频，2026-11-03

## 先看视频

- [全部30段错误回放，2分10秒](https://ccnuoe6sry36.feishu.cn/file/JqgDbbCcIozSUXxe7Skcdyd9nug)：开发集1段，加确认6–9全部20段抓碗错类及9次切换首查询未选对，不只挑几个极端案例。
- [连续四次把碗选成碗架](https://ccnuoe6sry36.feishu.cn/file/X7qpbO0ETo5QlextplEcBmG0nmd)：确认7的一段连续错误。
- [切换时拒选，随后恢复](https://ccnuoe6sry36.feishu.cn/file/M9oZb5S9LoGnQOx3J2Sci2PHnKf)：确认9，1208步拒选，1216步恢复。
- [开发集抓碗时错选把手](https://ccnuoe6sry36.feishu.cn/file/JtEJbDBSiobdn8xUcI3c68tGnrQ)：这不是碗架错误，区分其他类别。

红框是模型真实接受框，绿框是原离线弱参考目标。左边为原始主视角连续示范，右边保留最近一次实际评估查询的原图与原框，明确标记held；没有给中间帧伪造预测或插值框。两边都是同一主相机，不是双相机。影片是原示范叠加归档选择器结果，不是策略闭环执行。播放速度20源step/秒，不宣称已核验真实控制频率；未来画面仅离线审阅。

## 影响范围

统计同一个完整冻结配置，开发24条单列，确认6–9四批共64条且episode彼此不重合。四批均核验冻结配置文件SHA相同；确认6用其primary，确认7–9用retention，未把失败新模型混入。分母均为已知弱意图标签的查询，未知意图不计入准确率；没有按错误删样本。确认6–9是既有检查过的确认数据，当前汇总属于回顾审计，不是新留出试验。

| 数据 | 抓碗错类/抓碗查询 | 其中碗架 | 错类事件/涉及轨迹 | 切换首查询未选对/切换 |
|---|---:|---:|---:|---:|
| 开发24条 | 2/224（0.89%） | 0 | 1段/1条 | 0/16 |
| 确认6：16条 | 10/161（6.21%） | 1 | 5段/4条 | 3/9 |
| 确认7：16条 | 13/178（7.30%） | 9 | 6段/5条 | 0/4 |
| 确认8：16条 | 3/161（1.86%） | 3 | 3段/2条 | 3/7 |
| 确认9：16条 | 13/200（6.50%） | 7 | 6段/6条 | 3/8 |
| 确认6–9合计：64条 | 39/700（5.57%） | 20/700（2.86%） | 20段/17条 | 9/28（32.14%） |

- 39次抓碗错类占所有3510条已知查询的1.11%；但以抓碗阶段为分母是5.57%，不能只用全轨迹大量携碗正确帧稀释。20次是碗架，19次是其他非碗类。
- 17/64条轨迹出现过抓碗错类（26.56%），20段中12段仅1次错误查询，8段有连续错误；最长5个查询、首末观测相差32step。这是离散观测跨度，不是精确连续错误持续时间。
- 切换首查询9次未选对中，4次仍选旧类别、5次拒选；不是9次都主动选错物体。3次当时没有匹配新目标框，6次有框。
- 9次中6次在随后8或16step（通常1–2个查询）出现正确框，1次24step后恢复；余下2次在原标签连续可追踪窗口内没有更多观测，属于随访不足，不能写成永久失败。
- 切换首查询未选对涉及8/64条轨迹（12.5%）。抓碗错类与切换未选对的轨迹并集20/64（31.25%），两者有重叠，不能把17+8相加当受影响轨迹。

## 对“能不能用”的更正

开发集只有2/224次抓碗错类、集中1条轨迹，切换16/16正确；若只看开发，确实像少数局部问题。确认数据则抓碗错类涉及17条，不能说只剩一两个特例。但错误中的短暂拒选或一两查询后恢复，也不能直接等同任务失败。此前单凭完整离线门槛未过便笼统说“不能用”过于绝对；正确结论是可作为研究实验版本，实际任务影响仍待带/不带单目标条件的策略闭环对照。

这些视频也保留原监督可能提前切换的证据，绿色弱参考不等于人工意图真值。应由视频审阅和真实策略结果决定可接受程度，不能把31.25%出现过选择问题的轨迹解释成31.25%任务失败，也不能断言短暂错误完全不影响。现有单框选择器尚未接到动作策略在线adapter，不能把原示范顺利完成当成该选择器闭环成功。

## 视频索引

| 合集开始 | 片段 | 数据/轨迹 | 问题 | 源step |
|---|---:|---|---|---:|
| 00:00.00 | 1 | dev / table_setup_from_dresser_expert / demo_101 | 抓碗错类 | 376 |
| 00:04.45 | 2 | confirm6 / table_cleanup_to_dishwasher_expert / demo_14 | 抓碗错类 | 448 |
| 00:08.50 | 3 | confirm6 / table_setup_from_dishwasher_expert / demo_89 | 抓碗错类 | 232 |
| 00:12.55 | 4 | confirm6 / table_setup_from_dresser_expert / demo_14 | 抓碗错类 | 240 |
| 00:17.80 | 5 | confirm6 / table_setup_from_dresser_expert / demo_17 | 抓碗错类 | 256 |
| 00:21.85 | 6 | confirm6 / table_setup_from_dresser_expert / demo_17 | 抓碗错类 | 272 |
| 00:26.70 | 7 | confirm6 / table_setup_from_dishwasher_expert / demo_89 | 切换首查询未选对 | 232 |
| 00:30.75 | 8 | confirm6 / unload_dishwasher_to_dresser_suboptimal / demo_5 | 切换首查询未选对 | 360 |
| 00:34.80 | 9 | confirm6 / unload_dishwasher_to_dresser_suboptimal / demo_5 | 切换首查询未选对 | 368 |
| 00:38.85 | 10 | confirm7 / table_setup_from_dishwasher_expert / demo_109 | 抓碗错类 | 160 |
| 00:44.10 | 11 | confirm7 / table_setup_from_dishwasher_expert / demo_16 | 抓碗错类 | 216 |
| 00:48.15 | 12 | confirm7 / table_setup_from_dresser_expert / demo_2 | 抓碗错类 | 256 |
| 00:53.40 | 13 | confirm7 / unload_dishwasher_to_dresser_suboptimal / demo_27 | 抓碗错类 | 320 |
| 00:57.45 | 14 | confirm7 / unload_dishwasher_to_dresser_suboptimal / demo_77 | 抓碗错类 | 224 |
| 01:01.50 | 15 | confirm7 / unload_dishwasher_to_dresser_suboptimal / demo_77 | 抓碗错类 | 240 |
| 01:05.95 | 16 | confirm8 / table_setup_from_dishwasher_expert / demo_95 | 抓碗错类 | 208 |
| 01:10.00 | 17 | confirm8 / unload_dishwasher_to_dresser_suboptimal / demo_25 | 抓碗错类 | 568 |
| 01:14.05 | 18 | confirm8 / unload_dishwasher_to_dresser_suboptimal / demo_25 | 抓碗错类 | 736 |
| 01:18.10 | 19 | confirm8 / table_setup_from_dishwasher_expert / demo_95 | 切换首查询未选对 | 208 |
| 01:22.15 | 20 | confirm8 / table_setup_from_dresser_expert / demo_57 | 切换首查询未选对 | 360 |
| 01:26.20 | 21 | confirm8 / unload_dishwasher_to_dresser_suboptimal / demo_25 | 切换首查询未选对 | 728 |
| 01:30.25 | 22 | confirm9 / table_cleanup_to_dishwasher_expert / demo_79 | 抓碗错类 | 624 |
| 01:34.30 | 23 | confirm9 / table_setup_from_dishwasher_expert / demo_22 | 抓碗错类 | 216 |
| 01:38.35 | 24 | confirm9 / table_setup_from_dresser_expert / demo_11 | 抓碗错类 | 392 |
| 01:42.80 | 25 | confirm9 / table_setup_from_dresser_expert / demo_23 | 抓碗错类 | 232 |
| 01:47.65 | 26 | confirm9 / table_setup_from_dresser_expert / demo_56 | 抓碗错类 | 272 |
| 01:51.70 | 27 | confirm9 / unload_dishwasher_to_dresser_suboptimal / demo_42 | 抓碗错类 | 288 |
| 01:57.35 | 28 | confirm9 / table_setup_from_dishwasher_expert / demo_8 | 切换首查询未选对 | 272 |
| 02:01.40 | 29 | confirm9 / table_setup_from_dresser_expert / demo_11 | 切换首查询未选对 | 296 |
| 02:05.45 | 30 | confirm9 / unload_dishwasher_to_dresser_suboptimal / demo_42 | 切换首查询未选对 | 1208 |

## 复现与验证

`build_error_videos.py --root OUTPUT_ROOT --output NEW_DIRECTORY`读取固定归档，不运行新模型、不占GPU；`error_impact.py`按全episode相邻查询识别事件，未知/间隔>16step截断，不跨不明时间声称连续错误。恢复追踪止于未知、超间隔或主体再变。统计与既有sequence_metrics逐组核验切换、未选对及旧类滞留计数完全一致。视频仅读取obs/rgb，350条重复包含的查询逐一核验源字节SHA；全部30事件均导出，2590视频帧，129.5秒。76测试通过（5条既有依赖弃用警告），包括连续错误、未知中断、恢复/随访不足。

高清合集61,914,883字节，飞书上传接口返回file size beyond limit；改用同帧同分辨率CRF34压缩为9,448,503字节，成功上传。压缩版完整解码0错误，并已抽取开发把手/确认碗架画面检查文字及框；飞书下载副本SHA与本地一致。
高清合集SHA256：3e17bba817e7aab94fcbf3e2d4730423de1345408e3f86fbdb63546772b4f898。分享版SHA256：06b6080c784e713484b0c4172fed4d649b66a681f6f9d4fa134923d9ed0b1020。

压缩复现：imageio_ffmpeg.get_ffmpeg_exe()对应ffmpeg，参数`-i all_error_events.mp4 -an -c:v libx264 -preset fast -crf 34 -pix_fmt yuv420p -movflags +faststart all_error_events_share.mp4`。原视频、数据、预测归档、运行JSON不入Git，视频索引及代码记录在Git。固定配置SHA：8d84bb4e0eb649daec882481c9b60c874b7a03818907cebe6d2c62c5960a0a5a。

同时核验前两项研究已terminal：检测权重回缩18/18无合格，保留原检测器；等价框监督12拟合/62记录全部新模型不合格，保留当前候选。累计完整选择器比较记录1927+62=1989（含控制/状态重复），感知18回缩配方单列，不当新训练。后续碗类别辅助实现已提交7937e2f、75测试通过，但本次按用户要求优先视频与影响审计，尚未启动该轮训练，不把计划当结果。确认10仍未使用。

## 补充分开统计：稳定阶段自行跳类

“需要换目标时没选对”和“目标没变却自行换类”分开计算。固定配置在确认6–9的稳定主体、前后均选框的2985对相邻查询中，类别改变47次（1.57%），涉及27/64条轨迹；开发是10/1073（0.93%）、7/24条。这个指标只检查类别，不能判断同类不同碗之间的实例跳动，也不是任务失败率。

按确认6/7/8/9分别为12/758、9/709、12/721、14/797。完整30段短合集覆盖抓碗错类与必需切换首查询未选对，没有把全部稳定阶段跳类、同类框偏差等其他错误都独立剪成片段。这样保留短视频可读性，并把统计范围讲清楚。

`build_error_videos.py --audit-only`可在新目录只重算全部计数，不重做视频。第二次完整审计逐组复现第一版所有帧/事件/恢复计数，并与既有sequence_metrics的稳定跳类及分母完全一致。77项测试通过（5条既有依赖弃用警告）。视频链接、占比表、30项时间索引和SHA已在飞书回读核验，分享视频下载副本SHA相同；本轮无策略闭环成功率结论。
