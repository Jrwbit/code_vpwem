# MoMaRT 评测总对比与250k重复性验证｜2026-10-29

按用户要求重新编排的百分比表格和柱状图见[结果重整版](https://feishu.cn/docx/A4mVdzwCZovWOlxKxXNci1fynze)，本页保留完整历史审计记录。

最新状态：八步补跑已于13:49全部完成。seed100三轮均177/300，seed200为176/300，seed300为177/300；四步/八步完整对照见文末15:03更新。前文“缺项/未启动”为08:00历史快照。


北京时间08:00开始只读核查；原始结果快照时间：2026-10-29T08:00:32.552432+08:00。所有下列已有成绩均重新加载远端 result.npy 复算，未启动、修改或终止训练/评测，未补跑实验。

## 结果与状态

250k八步 seed100 第1、2轮均为 **177/300（59.00%）**，且逐episode成功标志完全一致。第3轮两项评测发生CUDA显存不足，只完成 **125/240（52.08%）**；该部分比例不能与完整300回合直接排名。supervisor已在00:03:07标记第一阶段失败，因此八步seed200/300没有启动。

250k四步三seed均完成：**100：177/300（59.00%）；200：188/300（62.67%）；300：176/300（58.67%）**。目前唯一完整的同eval seed执行长度对照为seed100：四步和两轮完整八步总分相同，尚无稳定提分证据。既有eval seed20260917的250k八步 **194/300（64.67%）** 已重新核验；不能拿它直接替代seed100/200/300的八步对照。

## 实验协议与可比性

| 实验组 | 训练/评测seed | 训练步/执行块 | 输入与可比性 |
|---|---|---|---|
| 带框训练节点 | 100 / 20260917 | 150k、175k、200k、225k、250k / 8 | 同离线训练框、在线combined_blend；150k后保留Adam/EMA续训并延长cosine至250k |
| 250k八步复测 | 100 / 100重复3次，200、300 | 250k / 8 | 同权重，物理GPU3/4/6；200/300原定GPU3/4，未启动 |
| 250k四步 | 100 / 100、200、300 | 250k / 4 | 物理GPU0/1/7；旧tracker，reset修复未应用，与八步一致 |
| 150k四步 / 坐标ROI | 100 / 20260917 | 150k / 4或8 | 四步是执行长度消融；ROI是另一结构，不能与250k混作同预算 |
| 普通No-History、Direct、Direct2 | 100 / 20260917 | 1M / 8 | 无红框，raw-image，有效batch64；1M与带框150k/250k预算不同 |
| VPWEM | 100 / 20260917 | 1M / 8 | 预计算embedding、任务context encoder、有效batch256，与raw-image存在混杂 |

带框组预测horizon=9、最近观测2帧、DDPM50、EMA、10并行环境、10配置各30回合，上限1500环境步。主RGB画1像素红框；训练为离线Grounding DINO框，评测为冻结combined_blend（max_age=3、refine_alpha=0.5）。四步仅执行动作索引1:5，training_action_steps仍为8；八步执行1:9。旧计分wrapper只回传动作块末两个info，缩短执行块同时增加观测、tracker更新和成功检查频率，不能把变化单独归因于预测质量。八步episode_steps=1504为外层计数，环境实际1500截断。

## 逐轮完成情况

| 轮次/评测seed | 物理GPU | 成功/实际回合 | 状态 |
|---|---|---|---|
| 250k八步100重复1 | 3 | 177/300，59.00% | 10/10；完成 |
| 250k八步100重复2 | 4 | 177/300，59.00% | 10/10；完成 |
| 250k八步100重复3 | 6 | 125/240，52.08% | 8/10；2项OOM失败 |
| 250k八步200 | 3 | 0/0；无成功率 | 未启动，前序失败阻断 |
| 250k八步300 | 4 | 0/0；无成功率 | 未启动，前序失败阻断 |
| 250k四步100 | 0 | 177/300，59.00% | 10/10完成，退出码0 |
| 250k四步200 | 1 | 188/300，62.67% | 10/10完成，退出码0 |
| 250k四步300 | 7 | 176/300，58.67% | 10/10完成，退出码0 |

四步queue_status、summary、completion均确认完成，最后完成时间05:34:50。八步重复1于00:00:31、重复2于00:03:07结束，均failures=0；重复3于23:13:32结束，failures=2。重复3的missing=0是队列结束字段，不能据此认定10项都有结果：原始文件实际仅8份。

## 250k十任务对照

每格为成功/实际回合；R1/R2/R3均为eval seed100的八步重复。八步200/300所有任务均未运行，因此没有可填的分项成绩。

| 任务配置 | 八步R1 | 八步R2 | 八步R3 | 四步100 | 四步200 | 四步300 |
|---|---|---|---|---|---|---|
| table_cleanup_to_dishwasher_expert | 13/30 | 13/30 | 13/30 | 17/30 | 16/30 | 14/30 |
| table_cleanup_to_dishwasher_suboptimal | 8/30 | 8/30 | 8/30 | 9/30 | 11/30 | 7/30 |
| table_cleanup_to_sink_expert | 20/30 | 20/30 | 20/30 | 15/30 | 16/30 | 13/30 |
| table_cleanup_to_sink_suboptimal | 14/30 | 14/30 | 14/30 | 18/30 | 18/30 | 16/30 |
| table_setup_from_dishwasher_expert | 27/30 | 27/30 | 缺失 | 25/30 | 22/30 | 22/30 |
| table_setup_from_dishwasher_suboptimal | 25/30 | 25/30 | 缺失 | 26/30 | 26/30 | 25/30 |
| table_setup_from_dresser_expert | 17/30 | 17/30 | 17/30 | 18/30 | 23/30 | 20/30 |
| table_setup_from_dresser_suboptimal | 16/30 | 16/30 | 16/30 | 16/30 | 14/30 | 19/30 |
| unload_dishwasher_to_dresser_expert | 20/30 | 20/30 | 20/30 | 17/30 | 21/30 | 19/30 |
| unload_dishwasher_to_dresser_suboptimal | 17/30 | 17/30 | 17/30 | 16/30 | 21/30 | 21/30 |

## 同seed重复性与逐episode一致性

三轮完整300回合成功率的均值、标准差、极差：**不可计算**，因为R3缺60回合。仅R1/R2完整轮：均值59.00%，样本标准差0.00个百分点，极差0.00个百分点；这不是三个独立seed统计。

三轮共同8配置、各240回合：成功数分别为[125, 125, 125]；成功率均值52.08%，样本标准差0.00个百分点，极差0.00个百分点。只描述共同子集，不能当完整基准。

| 对比 | 任务成功数相同 | 存储序号成功标志一致 | 不一致episode数 |
|---|---|---|---|
| R1 vs R2 | 10/10 | 300/300（100.00%） | 0 |
| R1 vs R3 | 8/8 | 240/240（100.00%） | 0 |
| R2 vs R3 | 8/8 | 240/240（100.00%） | 0 |

| 任务 | 三轮均值% | 样本SD百分点 | 极差百分点 | 三轮标志全一致 |
|---|---|---|---|---|
| table_cleanup_to_dishwasher_expert | 43.33 | 0.00 | 0.00 | 30/30 |
| table_cleanup_to_dishwasher_suboptimal | 26.67 | 0.00 | 0.00 | 30/30 |
| table_cleanup_to_sink_expert | 66.67 | 0.00 | 0.00 | 30/30 |
| table_cleanup_to_sink_suboptimal | 46.67 | 0.00 | 0.00 | 30/30 |
| table_setup_from_dishwasher_expert | 缺R3 | — | — | — |
| table_setup_from_dishwasher_suboptimal | 缺R3 | — | — | — |
| table_setup_from_dresser_expert | 56.67 | 0.00 | 0.00 | 30/30 |
| table_setup_from_dresser_suboptimal | 53.33 | 0.00 | 0.00 | 30/30 |
| unload_dishwasher_to_dresser_expert | 66.67 | 0.00 | 0.00 | 30/30 |
| unload_dishwasher_to_dresser_suboptimal | 56.67 | 0.00 | 0.00 | 30/30 |

以上按各文件episode_success存储顺序逐项比较；episode_id已核对为0–29。相同序号和seed不证明模拟器初始状态、扩散噪声、并行调度或GPU浮点轨迹完全一致；没有状态快照可作严格配对认证。总成功数相等也不等于逐episode复现。统计单位是重复执行，不是独立训练或独立评测seed；不据此构造独立样本置信区间或显著性结论。

## 训练节点与150k执行长度、ROI对照

| 版本 | 成功/回合/百分比 | 状态 |
|---|---|---|
| 150k八步 | 178/300，59.33% | 10/10完成，原始复核 |
| 175k八步 | 165/300，55.00% | 10/10完成，原始复核 |
| 200k八步 | 171/300，57.00% | 10/10完成，原始复核 |
| 225k八步 | 188/300，62.67% | 10/10完成，原始复核 |
| 250k八步 | 194/300，64.67% | 10/10完成，原始复核 |
| 150k四步 | 183/300，61.00% | 10/10完成，原始复核 |
| 150k坐标/ROI八步 | 160/300，53.33% | 10/10完成，原始复核 |

| 任务 | 150k/8 | 175k/8 | 200k/8 | 225k/8 | 250k/8 | 150k/4 | 150k ROI |
|---|---|---|---|---|---|---|---|
| table_cleanup_to_dishwasher_expert | 15/30 | 11/30 | 11/30 | 14/30 | 14/30 | 13/30 | 13/30 |
| table_cleanup_to_dishwasher_suboptimal | 9/30 | 6/30 | 6/30 | 5/30 | 11/30 | 6/30 | 0/30 |
| table_cleanup_to_sink_expert | 14/30 | 15/30 | 17/30 | 19/30 | 18/30 | 14/30 | 17/30 |
| table_cleanup_to_sink_suboptimal | 15/30 | 12/30 | 16/30 | 15/30 | 19/30 | 9/30 | 13/30 |
| table_setup_from_dishwasher_expert | 25/30 | 26/30 | 26/30 | 25/30 | 25/30 | 24/30 | 26/30 |
| table_setup_from_dishwasher_suboptimal | 30/30 | 28/30 | 25/30 | 27/30 | 23/30 | 27/30 | 22/30 |
| table_setup_from_dresser_expert | 23/30 | 19/30 | 22/30 | 25/30 | 24/30 | 28/30 | 22/30 |
| table_setup_from_dresser_suboptimal | 12/30 | 12/30 | 18/30 | 21/30 | 19/30 | 19/30 | 13/30 |
| unload_dishwasher_to_dresser_expert | 16/30 | 18/30 | 14/30 | 21/30 | 19/30 | 19/30 | 19/30 |
| unload_dishwasher_to_dresser_suboptimal | 19/30 | 18/30 | 16/30 | 16/30 | 22/30 | 24/30 | 15/30 |

同eval seed20260917的曲线是59.33%→55.00%→57.00%→62.67%→64.67%，并非单调提高；250k相对150k多16次成功、+5.33个百分点。它描述已有续训路径，不是架构改动效果。150k四步相对八步多5次、+1.67个百分点；150k坐标/ROI比纯画框少18次、−6.00个百分点，均只覆盖单训练seed。

## 跨seed与执行块对比

250k四步三独立评测seed等权均值60.11%，样本标准差2.22个百分点，极差4.00个百分点；总计541/900（60.11%），仅为该三seed汇总，不是900个独立训练样本。训练seed始终只有100。

| eval seed | 八步 | 四步 | 可解释差值 |
|---|---|---|---|
| 100 | R1/R2各177/300；R3不完整 | 177/300 | 相对两完整轮均0.00个百分点 |
| 200 | 未运行 | 188/300 | 无法计算 |
| 300 | 未运行 | 176/300 | 无法计算 |

seed100四步与R3共同8配置的比较仅作子集诊断：126/240，52.50%，R3为125/240；缺失两配置不补零。八步不存在完整100/200/300三seed均值；禁止把R1/R2/R3当三个seed，也不把旧取消目录中的结果借来补齐。

## 历史1M方法：独立预算分组

| 方法 | 原始复算 | 说明 |
|---|---|---|
| 普通No-History | 217/300，72.33% | 10/10完整，checkpoint=1M，eval seed20260917 |
| VPWEM | 196/300，65.33% | 10/10完整，checkpoint=1M，eval seed20260917 |
| Direct | 216/300，72.00% | 10/10完整，checkpoint=1M，eval seed20260917 |
| Direct2（Direct-token2） | 215/300，71.67% | 10/10完整，checkpoint=1M，eval seed20260917 |

| 任务 | No-History 1M | VPWEM 1M | Direct 1M | Direct2 1M |
|---|---|---|---|---|
| table_cleanup_to_dishwasher_expert | 16/30 | 19/30 | 19/30 | 18/30 |
| table_cleanup_to_dishwasher_suboptimal | 12/30 | 4/30 | 6/30 | 9/30 |
| table_cleanup_to_sink_expert | 22/30 | 21/30 | 22/30 | 18/30 |
| table_cleanup_to_sink_suboptimal | 20/30 | 10/30 | 26/30 | 21/30 |
| table_setup_from_dishwasher_expert | 28/30 | 25/30 | 29/30 | 27/30 |
| table_setup_from_dishwasher_suboptimal | 29/30 | 24/30 | 28/30 | 29/30 |
| table_setup_from_dresser_expert | 28/30 | 30/30 | 28/30 | 29/30 |
| table_setup_from_dresser_suboptimal | 18/30 | 23/30 | 21/30 | 24/30 |
| unload_dishwasher_to_dresser_expert | 25/30 | 24/30 | 20/30 | 19/30 |
| unload_dishwasher_to_dresser_suboptimal | 19/30 | 16/30 | 17/30 | 21/30 |

四种方法原始文件本次均可读取，分项与既有report_feishu.md一致。普通No-History 217/300（72.33%）是1M无框成绩，不能当作150k或250k画框成绩；预算与batch不等，不能据其高于64.67%推断画框有害。Direct和Direct2仅相差1次成功；VPWEM输入编码与batch不同，无法把分差归因于长期记忆机制。历史No-History后五项曾重复执行并覆盖，当前按最终每配置30回合计，未合并成60。

## 失败、缺失与证据边界

- R3缺失table_setup_from_dishwasher_expert及suboptimal，各30回合；两份日志均为torch.OutOfMemoryError，任务退出码1。错误发生在在线Grounding DINO/BERT推理路径。日志写GPU 0是进程可见设备编号，调度物理卡为6；不能据日志编号断言物理GPU0故障，也未进一步证明其他进程归属或根因。
- 八步seed200/300因阶段屏障未启动，各缺300回合；不会自动因本报告补跑。
- 取消的eval seed20260918/20260919以及旧seed100/200/300目录全部排除。
- 本次表列168份可用result.npy均验证二值成功标志、30回合、均值、checkpoint、eval seed和顺序ID；共168份（28+30+10+20+10+30+40）、5040个已存储回合，不合并为独立统计样本。

## 证据与复算方法

服务器：ssh bitbot421；读取Python：/home/jrw/anaconda3/envs/vpwem/bin/python。对各result.npy的episode_success求和，以实际数组长度作分母；按env_name和env_type对齐，不依赖目录遍历顺序。150k/250k四步summary逐任务计数与原始文件一致；250k四步summary中SHA256也逐项匹配。中间节点、既有250k与重复轮读取全部gpu*_status.tsv并与原始文件对照。

- repeats：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100/eval_online_combined_blend_250k_seed100_triplicate_20261028`
- intermediate：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100/eval_online_combined_blend_175k_200k_225k_20261028`
- 250k：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100/eval_online_combined_blend_250k`
- 150k：`/home/jrw/vpwem_outputs/momart_boxed_online_150k_combined_blend_20261026_gpu1`
- 150k_action4：`/home/jrw/vpwem_outputs/momart_boxed_no_history_150k_action4_20261027_gpu356`
- 250k_action4：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100/eval_online_combined_blend_250k_action4_seeds100_200_300_20261028`
- 1m：`/home/jrw/vpwem_outputs/fair_history_momart_multitask_eval_1m`

复算脚本：tools/collect_momart_evaluation_20261029.py（SSH stdin只读执行），tools/report_momart_evaluation_20261029.py（本地离线生成）。完整数组、结果哈希、状态快照保存在被Git忽略的artifacts/momart_evaluation_20261029/evidence.json；原始模型/数据不复制、不提交。

历史解释来源：docs/momart_boxed_250k_repeats_20261028.md末节、docs/momart_boxed_250k_action4_queue_20261028.md、docs/momart_boxed_intermediate_eval_20261028.md、docs/momart_boxed_nohistory_150k_improvement_20261027.md、docs/assets/momart_main_figures_20261028/evidence.md、reports/momart_1m_four_methods/report_feishu.md及reports/momart_history4_sh2_transfer_20260920.md。本报告为2026-10-29实时结果，旧文档的“运行中”仅保留历史含义。

新飞书文档：https://feishu.cn/docx/LJD4dbDkTo8jwsxuPVTch9RunSz

## 09:14更新：按用户要求用GPU0/1/3恢复八步评测

本节是08:00结果快照后的新操作，不改变上述历史结果。用户确认补跑三个未完成轮次，已启动独立恢复协调器PID3530881，提交时间2026-10-29 09:14:31+08:00，三个队列并行，不再受旧第一阶段失败屏障阻断。

- GPU0：eval seed100第3轮，仅补跑table_setup_from_dishwasher_suboptimal与expert两项，各30回合；原八项240回合继续保留。
- GPU1：eval seed200，完整10配置×30回合，共300回合。
- GPU3：eval seed300，完整10配置×30回合，共300回合。

合计新增660回合。策略保持250k、horizon9、执行8步、obs_steps2、DDPM50、10个并行环境、原combined_blend检测器与旧计分协议；不重新训练，不混入tracker reset修复，也不降低并行环境数。模型、训练配置和bundle的SHA256均核对为原值。

三张卡启动前均空闲。GPU gate要求空闲显存至少22000MiB、利用率不超过20%、连续三次采样满足再启动；每卡持有既有评测锁。其他GPU上的任务不停止、不迁移。

原结果目录继续使用 `eval_online_combined_blend_250k_seed100_triplicate_20261028/{seed100_repeat3,seed200,seed300}`，旧GPU6日志保留，新任务分别生成gpu0/gpu1/gpu3日志。启动前已核验并记录8份原有效结果的SHA256，完成时再次核验，禁止覆盖它们。补齐后的seed100第3轮将由原GPU6的8项加GPU0的2项组成，必须标注为跨GPU补齐，不能说整轮都在GPU6完成。

恢复状态单独保存在上述根目录的 `recovery_20261029/`：manifest.json记录输入哈希、原结果哈希和缺项；queue_status.json记录三个GPU gate；summary.json持续统计经过校验的结果；最终生成completion.json。旧supervisor.log的失败结论保留为历史，之后判断进度应读取恢复目录，不再只看旧日志。

恢复脚本：`experiments/momart_boxed_250k_action8_recovery_20261029/run.py`；BHR入口 `/home/jrw/vpwem_jobs/momart_boxed_250k_action8_recovery_20261029.py`。实现提交 `4a45459`，脚本SHA256为 `2e98bc556ad12f491cf4e1eaff5ed2cd8240bedb296f5591f1e824714146dfe9`。四项测试通过，覆盖八步协议验证、原结果保护、部分结果不报最终率、写入中的结果不被提前读取。

此时只确认补跑任务已提交，尚无新的完整成功率；待各任务实际完成后才能更新上方08:00表格。

启动回读：09:14:53–54三卡均已接入。GPU0跳过已完成任务后启动缺失的dishwasher/suboptimal，评测PID3531456；GPU1启动seed200的首项，PID3531340；GPU3启动seed300的首项，PID3531318。三份命令均使用原250k八步入口、各自seed和10并行环境；当前为进程启动/环境加载阶段，不代表完成。

## 后续汇报口径：仅总成功率与 No-History 柱状对照

用户最新要求：MoMaRT只评估总成功率，与之前的普通No-History对照，优先使用柱状图；后续结果汇报按此口径，不再默认展开任务分项和逐episode一致性。09:18原始结果复核及图表见 `docs/momart_total_success_20261029.md`，飞书：https://feishu.cn/docx/RXj4dQ7gHoPAJvxzKB1cTwwMnZd 。旧报告保留历史依据，补跑完成后更新总成功率。训练预算与eval seed在图表中明确区分，未完成轮次不列最终成功率。


## 15:03更新：八步补跑完成，最终四步/八步对照（2026-10-29）

本节取代08:00快照中“第3轮缺项、seed200/300未启动”的当前状态；上文保留为当时的历史记录。恢复协调器于13:49:51+08:00完成，三个任务退出码均为0。

本次重新加载八步5轮与四步3轮共80份result.npy，核验每份30回合、二值成功标志、均值、250k checkpoint、评测seed及执行步长。原第3轮的8份有效结果SHA256全部未变。

### 最终轮次

| 八步轮次 | 成功/回合 | 成功率 | 完成情况 |
|---|---|---|---|
| seed100第1轮 | 177/300 | 59.00% | 10/10完成 |
| seed100第2轮 | 177/300 | 59.00% | 10/10完成 |
| seed100第3轮（补齐） | 177/300 | 59.00% | 10/10完成 |
| seed200 | 176/300 | 58.67% | 10/10完成 |
| seed300 | 177/300 | 59.00% | 10/10完成 |

第3轮由原GPU6完成8项、GPU0补齐2项组成；不能描述为整轮都在GPU6运行。三个seed100轮次全部300个存储序号的成功标志一致，三轮成功率均59.00%，样本标准差和极差均为0。相同成功标志不证明完整模拟器状态或浮点轨迹一致；三轮也不是三个独立seed。

### 同seed执行步数对照

| 评测seed | 八步 | 四步 | 四步−八步 |
|---|---|---|---|
| 100 | 177/300，59.00% | 177/300，59.00% | +0.00个百分点 |
| 200 | 176/300，58.67% | 188/300，62.67% | +4.00个百分点 |
| 300 | 177/300，59.00% | 176/300，58.67% | -0.33个百分点 |
| 三个不同seed合计 | 530/900，58.89% | 541/900，60.11% | +1.22个百分点 |

均值按三个不同评测seed等权计算；八步seed100只取一轮，不将重复轮次重复加权。四步合计多11次成功，收益主要来自seed200；不能据此断言稳定显著更好，训练seed仍只有100。更短执行块同时改变重规划、tracker更新和成功检查频率。

### 完整逐任务对照

| 任务配置 | 八步100 | 八步200 | 八步300 | 四步100 | 四步200 | 四步300 |
|---|---|---|---|---|---|---|
| table_cleanup_to_dishwasher_expert | 13/30 | 11/30 | 14/30 | 17/30 | 16/30 | 14/30 |
| table_cleanup_to_dishwasher_suboptimal | 8/30 | 9/30 | 4/30 | 9/30 | 11/30 | 7/30 |
| table_cleanup_to_sink_expert | 20/30 | 18/30 | 16/30 | 15/30 | 16/30 | 13/30 |
| table_cleanup_to_sink_suboptimal | 14/30 | 17/30 | 15/30 | 18/30 | 18/30 | 16/30 |
| table_setup_from_dishwasher_expert | 27/30 | 26/30 | 27/30 | 25/30 | 22/30 | 22/30 |
| table_setup_from_dishwasher_suboptimal | 25/30 | 25/30 | 27/30 | 26/30 | 26/30 | 25/30 |
| table_setup_from_dresser_expert | 17/30 | 21/30 | 20/30 | 18/30 | 23/30 | 20/30 |
| table_setup_from_dresser_suboptimal | 16/30 | 12/30 | 21/30 | 16/30 | 14/30 | 19/30 |
| unload_dishwasher_to_dresser_expert | 20/30 | 18/30 | 15/30 | 17/30 | 21/30 | 19/30 |
| unload_dishwasher_to_dresser_suboptimal | 17/30 | 19/30 | 18/30 | 16/30 | 21/30 | 21/30 |

上述均使用相同250k权重、预测长度9、最近2帧、DDPM50、旧combined_blend及原计分协议，未混入reset修复。补跑新增660回合，保留原失败日志。统计读取原结果目录及recovery_20261029/的manifest、summary、completion；原始数组不提交Git。
