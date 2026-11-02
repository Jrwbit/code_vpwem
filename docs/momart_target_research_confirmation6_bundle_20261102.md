# 单目标条件：第六组确认与可搬移推理包

2026-11-02。卡 1 的第六组串行流水线已完整结束；预先冻结的模型/阈值未修改。候选为 `transition4_new0.75`，原冻结模型 25% + 切换损失 4 倍模型 75%，最终阈值 0.6、margin 0。控制保持原 `frozen_for_confirm4` 阈值 0、margin 0.2。持续目标 active，完整验收未通过，不升级正式模型。

正文和两张表已追加至 [MoMaRT 飞书报告本节](https://ccnuoe6sry36.feishu.cn/docx/RfXudCiBOoeG3Bx0D34cbU1rn0g#doxcnsXPA58h3OHYY7jSUo1Yixe)，回读确认；未覆盖历史结果。

## 新轨迹结果

16 条未用于选择器训练/开发的新 episode，1265 原始帧中 1228 帧通过对齐，872 个已知弱主体标签（356 帧未知），主体可见 862、匹配候选可用 835。全部来自 source-nonvalid，只证明选择器未见，不构成上游视觉/动作系统完整留出。标签定义、未知处理和全部分母保持冻结协议。

| 第六组固定配置 | 正确/接受 | 精度 | 正确框召回 | 覆盖 | 取碗错类 | 持碗错类 | 即时切换 | 旧类滞留 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始指针 | 725/864 | 83.91% | 84.11% | 99.30% | 28 | 11 | 7/9 | 1 |
| 原冻结模型 | 774/832 | 93.03% | 89.79% | 96.40% | 14 | 2 | 6/9 | 2 |
| 新融合候选 | 795/840 | 94.64% | 92.23% | 97.33% | 10 | 1 | 6/9 | 2 |

相对原冻结模型：正确框 +21，精度 +1.61、召回 +2.44、覆盖 +0.93 个百分点，取碗错类减少 4、持碗错类减少 1；五角色正确框召回均提高。稳定类别抖动 28/751→12/758，稳定正确→拒选 17→16。但即时切换仍只有 6/9、旧类滞留仍为 2，未解决剩余时序错误。

新增的原模型相对门槛总体及源分区均通过；原 V4 相对原始指针的切换门槛失败（7/9→6/9、滞留1→2），因此**双门槛完整验收失败**。不因相对较强模型有收益就撤销先前登记的原始指针门槛，不将该候选宣布为正式升级。

16 episode 成对 bootstrap 10000 次，相对原冻结模型：精度差 95% percentile 区间 [+0.41,+2.83] 个百分点，合并正确框召回差 [+0.62,+4.51]，episode macro 召回差 [-0.033,+4.86]。这是此弱代理语料的有限簇区间，不代表人类意图准确率或动作策略成功率。相对原始指针精度 +10.73、召回 +8.12 个百分点，仍不能掩盖切换门槛失败。

## 全部切换与监督审计

1228 帧支持审计完整复现旧 held 401 帧，0 差异；旧 held 载体冲突 1、前瞻 held 冲突 2、支持且缺少双指抓持 5，集合可重叠。保持所有帧，不按模型错误删除或重标。

`diagnose_intent.py`新增每条轨迹全部已知相邻主体切换的明细，包含成功；未知帧或超过16步间隔打断，和既有序列指标口径一致。三个模型分别生成诊断，保存全部9处切换，新候选失败3处：

| episode / step | 代理切换 | 新候选表现 | 原控制 / 原始指针 | 审计限制 |
|---|---|---|---|---|
| table_setup_from_dishwasher_expert/demo_89 @232 | rack→bowl | 仍选rack | rack / bowl | 有载体共动冲突标记，尚非人工意图错误证明 |
| unload_dishwasher_to_dresser_suboptimal/demo_5 @360 | rack→door handle | 仍选rack，缺少正确实例候选 | rack / bowl，同样均无正确候选 | 排序器不能补出缺失框；不豁免门槛 |
| 同 demo_5 @368 | door handle→bowl | bowl排第一但被拒选 | bowl / bowl | bowl概率0.551864，统一阈值0.6；是真实拒选损失 |

因此“两个模型都6/9”不意味着失败帧完全相同。全部872个已知标签中，新候选正确795、缺少匹配候选37、存在正确候选但错类20、拒选19、同类实例/框错1。oracle类别诊断明确旁路，不当作自主模型。

## 可搬移的研究推理包

新增 `export_runtime_bundle.py`，支持本轮使用的公开 base/geometry/spatial 神经指针、固定可用性和概率融合；不支持的研究分支明确拒绝，不擅自丢弃输入。仅复制两个小模型检查点、必要API源码、任务词表、阈值及逐文件SHA，排除数据集、标签、检测/特征缓存。路径改写为包内相对路径。`runtime.py`允许相对检查点路径相对显式 bundle root 解析，原绝对路径行为保持。

包始终标记 `research_candidate_not_promoted`。它只接受已按原冻结前端生成的公共特征，输出一个归一化 xyxy 框或 valid=false；不是检测器、完整传感器前端、扩散动作策略或闭环部署。仍是四个完整任务名词表，未推广到五任务/十数据源。

BHR 包目录：`/home/jrw/vpwem_outputs/momart_target_research_20261101/runtime_bundle_confirm6`。8个受哈希保护文件加 manifest，manifest SHA256 为 `af366e3f2467f377f6773702f2ec7088d7472474e2508c7cde521ae1dd4e9496`。

`verify_runtime_bundle.py`将包复制到临时新目录，启用 `python -I`、空工作目录，只导入包内API和已安装的NumPy/PyTorch；子进程只接收公共帧特征、task、episode、step。910帧/34episode对比原工程推理，概率最大差 **0**、接受槽差 **0**，逐文件SHA一致。标签仅在外层评分数据读取器中存在，未传入子进程。临时导入路径在验证结束后移除，验证JSON记录可审计来源。

40项相关测试通过，5条既有依赖弃用警告；测试覆盖移动包后的相同预测、逐文件哈希，以及切换诊断不跨未知/大间隔。代码提交`7ce947f`、`b904355`；权重和运行结果不进入Git。

## 下一步与冻结第七组

第六组此后只作已查看诊断/回归，不在其上调阈值。下一轮先在既有train108/dev24研究按预测对象类别校准拒选阈值，保留完整双对照和原/新开发子集门槛；不能仅将第六组0.552这一帧放行。缺框与载体监督冲突分别继续审计，拒选校准不能被描述成已解决这两类错误。

该研究搜索前已冻结第七组16条/1272原始帧，全部source-nonvalid，与全部已查看语料、新train/dev、原保留test/回归均无episode交集。原始manifest SHA256：`fceb36f84e02e6899398c08b12f8b8af1e146707d1b77da90adec5306e21f59d`。尚未生成标签、候选或评分；本轮新增流水线已结束，卡1授权继续有效。

## 证据位置

BHR根目录`/home/jrw/vpwem_outputs/momart_target_research_20261101`：`confirmation6_incumbent_blend/results.json`、`confirmation6_incumbent_blend_pipeline.log`、`confirmation6_vs_baseline_bootstrap.json`、`confirmation6_vs_control_bootstrap.json`、`confirm6_support_audit.json`、`confirm6_{primary,control,baseline}_diagnosis.json`、`runtime_bundle_confirm6_parity.json`和`confirm7_corpus/manifest.json`。累计已完成比较仍为1215条，新增审计和打包不是新算法配置。
