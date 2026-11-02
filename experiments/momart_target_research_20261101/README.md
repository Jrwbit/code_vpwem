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
