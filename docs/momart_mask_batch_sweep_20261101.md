# MoMaRT Mask 批量推理测速与四卡恢复

2026-11-01，北京时间。用户先要求BS16，随后要求比较64/128/256。原实现确为逐帧感知，即推理BS1；训练physical_batch_size与本参数无关。

## 单卡实测

BHR GPU3空出原worker后，对同一数据集连续512帧测速，包含原始RGB切片读取、Mask、ROI特征和CPU后处理，不含SQLite编码提交及因果回放。保持FP32及原冻结权重、阈值、输入分辨率。

| batch | 帧/秒 | torch峰值allocated MiB |
|---|---:|---:|
| 16 | 113.16 | 1898.69 |
| 64 | 116.14 | 6895.15 |
| 128 | 108.38 | 12397.65 |
| 256 | OOM | 失败时16804.55 |

失败时allocated并非所需总显存或nvidia-smi占用。实际显存与候选框数量、分配器预留有关，生产进程nvidia-smi占用高于本表。选用64，而非默认越大越快。较早与旧worker争用同一卡的短测36/58帧每秒不能与空卡表直接比较。

## 改动及恢复

新增独立batch_perception执行批量Mask及视觉backbone，复用原select_boxes、检测筛选和ROIAlign函数。prepare_train按可配置inference_batch_size顺序读取、写入，尾批保留；SQLite按跨64帧边界或尾批提交。prepare_multigpu增加--inference-batch-size，默认仍为1。执行适配器单独记录在Git，未修改perception.py/mask_detector.py冻结来源哈希；新增实现不在原bundle源码清单中，不宣称原bundle源码核验覆盖本次变更。

BS16抽样16帧数量类别一致，最大框/score/feature差异分别0.00007391/0.0003823/0.0008483。扩大至64帧、BS64时数量及类别多重集仍全部一致，但逐项按输出顺序比较最大框差异0.045018（归一化坐标）、score差异0.006400、feature差异0.130554。严格小误差断言未通过，不能宣称逐位或严格数值等价；排序/候选框选择在浮点批量执行下可能变化，本次没有重新执行完整感知质量验收。已有BS1提交结果保留，后续BS64结果继续写入同一数学协议缓存，执行批次变化在本报告和运行记录明确留档。

停止旧worker后确认四个worker均为僵尸态、已经退出写入，再结束旧supervisor。四个SQLite续跑起点分别14592、14656、13312、13760帧，保留原缓存，无重复writer。新的supervisor PID765138，GPU3–6 worker分别765139/765140/765141/765142。运行目录沿用/home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101，配置均为inference_batch_size=64。bs64_resume.json保留恢复起点，bs64_supervisor.log保存新supervisor日志。

20:03左右快照：GPU3–6分别已提交17536、17792、16192、16576帧；状态全部perception、BS64，速度115.11/121.12/109.08/110.69帧每秒，合计456.00，约原首批180帧每秒的2.5倍。纯剩余感知按队列估算约25分钟量级，未计后续CPU因果回放、压缩写盘及数据集差异；完整缓存尚未完成。policy_training_started=false。

验证：本地及BHR针对原协议、队列、批量尾帧/续跑的初轮14项检查通过；本地尾帧测试增加覆盖16/64/128/256后共17项通过。git diff --check通过。服务器短暂SSH拒连后通过持久连接恢复。生产代码已部署；权重、缓存、数据库和运行日志均不进入Git。

飞书沿用：https://ccnuoe6sry36.feishu.cn/docx/RfXudCiBOoeG3Bx0D34cbU1rn0g 。
