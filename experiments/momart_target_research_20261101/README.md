# MoMaRT 单目标选择实验

持续目标与验收：`protocol.json`、`docs/momart_target_research_goal_20261101.md`。当前不自动替换正式检测器、对象缓存或动作策略；离线选择器分数不代表策略任务成功率。

## 公共输入与单框输出

`runtime.SingleBoxSelector(selection_json, output_root)`读取已冻结的开发集选择结果和可信本地检查点。`select(FrameFeatures(...), episode_id, sensor_step)`只读取公开观测特征，返回`valid / box / class_id / confidence / probability`；`box`为归一化xyxy四坐标，拒选时为`None`。输入类型不包含标签、动作、场景物体GT位置或未来帧。同一sensor_step重复调用不会再次更新状态；跨episode清空状态；倒序或同帧不同输入会报错。

基础输入：候选[8,139]（ROI128、框4、分数1、类别6）、当前/过去图像及proprio上下文[280]、有效槽[8]、任务名；按模型额外需要腕部/深度、机器人几何或预测掩码特征。仅支持当前验证过的四个数据源任务词表；未验证另外六个expert/suboptimal数据源以及table_cleanup_to_sink。

`spatial_frontend.SpatialFrontEnd`为spatial/mask_small/mask_spatial模型提供精简的原始观测入口；设置`include_visual=True`时，额外运行冻结主相机ROI/global编码器，支持base/geometry及其availability组合。输入当前90×120 RGB、归一化深度、当前/过去10维proprio、自身躯干升降关节、episode/step/task；过去观测必须在max(0,t-8)，使用真实机器人测量。相机垂直FOV、深度缩放、机器人缩放从环境配置显式传入。此路径不运行ROI/global视觉编码器、不需要腕部图像，将未使用的视觉通道明确标为不可用；运行时拒绝将这些占位通道送入依赖视觉特征的模型。仍使用已有Mask检测器的完整图像推理。

```python

# 先将 experiments/momart_target_research_20261101 加入 PYTHONPATH
from runtime import SingleBoxSelector
from spatial_frontend import SpatialFrontEnd

selector = SingleBoxSelector(selection_json, output_root)
frontend = SpatialFrontEnd(object_memory_config, robot_urdf, selector.tasks,
                           device="cuda:0", fov=45., depth_high=3., include_visual=True)
try:
    features = frontend.build(current_robot_observation, past_robot_observation)
    target = selector.select(features, current_robot_observation.episode_id,
                             current_robot_observation.step)
finally:
    frontend.close()
```

上述示例要求检查点为相应空间模型。带rich角色头的组件仍需要真实视觉特征，应从既有公共特征通路提供完整`FrameFeatures`。目前返回值尚未接到扩散策略条件层，不能宣称已能提升闭环成功率。短暂失检使用无效标记，不编造新框。

## 重现入口

BHR源码：`/home/jrw/code_vpwem_momart_association_20261030`；Python：`/home/jrw/vpwem_envs/detector_compare/bin/python`；结果根目录：`/home/jrw/vpwem_outputs/momart_target_research_20261101`。树模型依赖安装在结果目录`deps`，使用`PYTHONPATH=<research>/deps:.`，CPU线程使用`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`。不把权重、图片、缓存和结果数组提交Git。

- `benchmark.py`：原始44个对照；`benchmark_geometry.py`：25个几何对照；`refine.py`：50个融合/质量对照；`temporal.py`：96个原始因果角色对照。
- `mask_benchmark.py`：三随机种子、84个掩码配置；`benchmark_dense.py`：原20条train/dev轨迹补密后的91个对照；`mask_benchmark.py --dense`：补密掩码84个对照。配置数含参数、种子和重复对照，不是独立算法数。
- `select_final.py --screens ... --output ...`：只根据同一开发集结果冻结跨屏选择。`confirm.py --prefix confirm2 --purpose confirmation`：随后评价新确认语料。已查看语料必须使用`--purpose regression`，不能继续冒充新确认。
- `verify_runtime.py`：910帧公共逐帧API与批量预测的概率、接受槽和重复步等价性；`verify_frontend.py`：固定均匀抽样的24帧原始传感器与归档空间特征工程对照。
- `densify.py`与`prepare_corpus.sh`：保留episode划分的补密、空闲GPU渲染、CPU特征准备。`freeze_confirmation.py`默认只选源valid；显式`--source-pool valid-first`不足时才从其他未使用源episode补足，并记录source_partition。第三组共有7条valid、1条非valid，不能称全部官方valid或完整视觉/策略系统的留出验证。

## 当前未结束的实验

第二组确认达到V3总体指标，但抓碗错类5→8、切换后旧类1→3，完整目标仍未达成。V4在整体精度/召回及逐角色召回之外，要求开发集抓碗错类严格减少、持碗错类不增、稳定阶段类别抖动及切换即时性不退步。

`expanded.py`公开调整数据用途：第一组每任务按episode字典序第一条train、第二条dev；第二组全部train。与原16条train、4条dev合并为28条train、8条dev（1519/421个代理标签），原test和阶段回归维持独立划分；第三组完全排除。第一/二组结果保留为历史prospective实验记录，此后不再用它们声称新模型留出性能。`benchmark_dense.py --expanded`与`temporal.py --expanded`均只按扩充开发集选择，第三组评分前再次冻结检查点和阈值。

```bash
/path/to/python -m pytest experiments/momart_target_research_20261101/test_research.py -q
```

最新实测及标签审计：`docs/momart_target_research_progress_20261102.md`、`docs/momart_target_research_contact_audit_20261102.md`，同步飞书链接见报告。

## 已确认的接口细节与后续工作

MoMaRT此配置的动作是10维（不启用tuck）：base2、末端IK位姿增量6、夹爪1、reset1；不是12维。实际动作缩放与数据中-1对应闭爪、+1对应张开，不能仅信旧代码注释的符号。原10维proprio是head2、grasped1、EEF position3、EEF quaternion4；末4维不是Euler。原grasped只是平均手指关节开度在(0.002,0.04)范围内的5步滤波标记，不包含对象身份。trunk_joint_pos、gripper_pos、gripper_vel均为已有机器人自身测量，扩展选择器时需明确接口，不可从场景对象GT取代。

第四组16条确认整体精度95.19%、召回92.73%，抓碗错类10→2，但旧标签下仍有1次旧类滞留，V4未达成。全4652帧物理接触/支持关系审计完整复现1472帧旧held标记（0差异），其中20帧出现载体共动且缺少双侧抓持证据，另24帧的前瞻held证据全部来自该类冲突；集合可重叠。这些是需复核的标签冲突，不是已证实的人类意图错误。

下一步先区分“实际操纵主体、尝试接触目标、确认抓持状态”，统一修订监督并保持历史分数，不按模型错误单点删标签。验证目标查询/阶段完成证据、接触前目标保持及腕部/动作意图交叉核验；任何oracle类别/场景状态只作上限对照，不能冒充自主部署。当前未完成五任务/十数据源推广或扩散策略闭环实验。

## 意图查询诊断与自主对照

`diagnose_intent.py --research ROOT --prefix confirm4 --predictions ROOT/confirmation4_v4/primary_predictions.npz --audit ROOT/contact_support_corpus_audit.json --output NEW.json`：对齐冻结概率与原始语料，保留旧标签和总分母，分解缺框、拒选、错类及同类实例/框错误；争议监督只旁路分组。oracle类别只作明确标记的离线诊断。

`hard_query.py --root /home/jrw/vpwem_outputs --output NEW_DIRECTORY`：复用8个train-only角色头，在固定expanded/dev比较32个自主硬类别约束。要求第五组manifest已在搜索前冻结；不会加载其标签/特征。本轮最好配置正确367/418，弱于原可用性模型370/418，因此不升级、不进入新确认。详见`docs/momart_target_research_query_diagnosis_20261102.md`。该方案仅接入批量确认器，未扩展公开流式部署接口。

## 自身历史与监督置信度

`robot_history.py --corpora ... --geometry-metadata ... --urdf ... --output NEW.npz`读取经过验证的机器人自身双指列，生成6维位置/变化及31维严格过去命令统计；以source/demo/step物理键对齐，避开语料复用ID。需要部署端实际自身测量和已执行命令，不能将示范当前/未来动作当输入。默认模型接口未扩展这些新通道。

`benchmark_history.py --root OUTPUT_ROOT --history NEW.npz --output NEW_DIRECTORY`：24次拟合，48个概率对照及1控制；最好370/409，原模型370/410，未增加正确框。

`benchmark_confidence.py --root OUTPUT_ROOT --audit contact_support_corpus_audit.json --output NEW_DIRECTORY`：仅对14个已审计冲突训练帧降权，9次拟合、19条记录；开发标签、权重和全分母不变，原权重三个种子完全复现历史参数，降权未胜出。两个实验合入相同开发集573条V4记录后，既定select_final规则仍保留原可用性模型。新确认集未评分。详见`docs/momart_target_research_history_confidence_20261102.md`。

## 候选对应、冻结策略表示与动作辅助

`benchmark_tracking.py --root OUTPUT_ROOT --output NEW_DIRECTORY`比较12个拟合、25条记录；candidate_history.py仅用相邻公开检测框/外观/观测深度几何对应，无GT身份输入、无跨遮挡追踪保证。

`policy_features.py --corpora ... --checkpoint RAW_POLICY --config SAVED_CONFIG --normalizers SAVED_NPZ --output NEW.npz --device cuda:0 --batch-size 16 --disable-tf32`提取原图策略的冻结EMA观测表示。记录源检查点和归一化SHA、GPU映射及TF32开关；默认数值后端出现过batch1/16差异，后续使用显式严格FP32并保留旧失败输出。benchmark_policy_features.py --root OUTPUT_ROOT --features NEW.npz --output NEW_DIRECTORY比较相同开发集。它不是读取真实未来动作或预测目标标签。

`plan_labels.py --observation-metadata policy_features.json --output NEW_LABELS.npz`仅导出禁止作为推理输入的未来动作标签；benchmark_plan.py --root OUTPUT_ROOT --labels NEW_LABELS.npz --output NEW_DIRECTORY比较9个自预测计划辅助模型。真实命令仅用于损失，目标查询始终使用模型自己预测的计划。上述研究检查点仅支持批量确认器，未加入公开流式部署接口。详见docs/momart_target_research_temporal_plan_20261102.md。

## 训练轨迹扩充与切换训练权重

`freeze_growth.py --reference CORPUS --exclude ALL_EXAMINED_AND_RESERVED --output NEW_CORPUS`固定source-train池的新train/dev，默认每任务20/4条，排除已查看轨迹和保留确认组。`growth_data.py`保留原划分，校验原/新episode以及第三至第五组隔离，合并108train/24dev，仅使用本轮必要公共特征。

`MOMART_ASSIGNED_GPU=1 prepare_corpus.sh PROJECT ROOT PYTHON PREFIX 1 geometry`使用用户明确分配的卡1并跳过未使用腕部特征；不指定分配卡时保留原等待逻辑。`run_growth_when_ready.py --resume-geometry ...`仅在旧进程结束且前置数据存在时显式续跑，原失败状态留档。几何导出已改为候选NPZ一次解压，555帧逐元素等价。

`benchmark_growth.py --root OUTPUT_ROOT --output NEW_DIRECTORY --device cuda:0`做old28/enlarged108的12拟合/25记录，选择同时要求原8/新16dev分别通过V4。两种控制的权重冻结，开发阈值仍重新校准；不得称完整配置冻结。`freeze_growth_confirmation.py`在确认组未标注/未提取候选前冻结胜出结果、原控制、训练隔离审计及确认manifest SHA。`confirm.py --public-views geometry`显式读取无需腕部的公共特征；默认rich保持原要求。`bootstrap_confirmation.py --reference baseline|control`按episode成对抽样。

第五组新候选P92.87%、R90.75%，原冻结控制P93.04%、R88.68%；取碗错类8→12，切换正确8/10→6/10，故未升级。详见`docs/momart_target_research_growth_confirmation_20261102.md`。

`freeze_confirmation.py --exclude ... --exclude-audit FITTED_DATA_AUDIT --output NEW_CORPUS ...`同时排除完整拟合/评分审计里的episode，不能仅传补密train/dev而遗漏保留test/回归。下一轮`benchmark_growth.py --weighting transition`保持108/24划分与输入，15拟合/32记录，比较仅训练损失的切换后3帧2/4倍、取碗2倍及组合，并先校验第六组隔离。标签不进入推理，未知帧打断训练窗口。

重加权单模型完成但相对原冻结模型的抖动门槛未通过。`benchmark_growth_blend.py --root OUTPUT_ROOT --output NEW_DIRECTORY`按五种权重家族的开发胜出成员，比较新模型占25/50/75%的固定概率融合及两个完整固定配置控制，17记录、0拟合；阈值选择同时要求相对原始指针和原冻结模型的combined24/old8/new16门槛。

`select_growth_final.py --root OUTPUT_ROOT --control FROZEN_CONTROL --holdout UNTOUCHED_CORPUS --screens COMPLETE_COMPARABLE_SCREENS --output NEW_SELECTION`检查完整搜索和episode身份，按原/新开发子集与原冻结控制门槛选择。新增`incumbent_evaluation_gate`比较accepted recall，保留旧指针raw recall逻辑；选中配置将`incumbent_gate_required=true`传给confirm.py，确认时相对原控制的总体及每个源分区均须通过。第六组此前未标注，在胜出模型/阈值冻结后才准备。开发胜出transition4_new0.75提高正确框和切换，但正确→拒选及碗架召回有退步，详见`docs/momart_target_research_transition_emphasis_20261102.md`。
