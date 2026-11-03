# 十源统一目标选择器与新框缓存，2026-11-03

十源654460帧新目标框缓存已全部完成，逐源完整性和两种策略输入检查通过。用户要求补齐原No-History十个数据源的新框缓存，并明确不接受水槽专用模型/路由。最终方案是一个联合训练的五任务MLP64指针网络，十源共用一套权重、同一全局校准和同一推理代码。expert/suboptimal只表示示范质量来源，映射到同一任务ID；没有按任务切换权重或旧模型回退。原四任务模型仅作离线对照，不参与新缓存推理。主相机选框；动作策略原有腕部等全部观测保留。

## 任务与训练

五个任务是cleanup dishwasher、cleanup sink、setup dishwasher、setup dresser、unload dishwasher to dresser，各有expert/suboptimal两个数据源。公共视觉/机器人上下文从280维扩为281维，几何候选仍155维，含池化的最终上下文450维；只增加第五个任务one-hot位置，视觉编码器/检测器保持冻结。四任务旧runtime仍按原280维合同运行，五任务从冻结词表确定维度。

补充水槽24条episode：每个水槽源8 train、2 dev来自原策略train，2 test来自原策略valid，seed20261103、每16step均匀取帧，先划分后处理。共1061候选帧；因仿真重渲染与录制RGB的MAE>12，沿用已有弱标签准入规则剔除329帧的参考监督，保留732帧。此过滤只影响弱标签补训/评分，全量缓存仍包含原654460训练帧，不删难帧。标签是active_subject_v1的运动/接近/持碗代理，不是人工意图真值；参考对象状态与未来画面只生成训练标签，不进入模型输入。已有确认10完全未打开。24条水槽episode的完整state轨迹SHA未发现跨split完全重复。选择器监督来自原四源和新增两水槽源的抽样episode，其余同任务质量来源使用同一网络泛化；不把它描述成十源全量监督拟合。选择器自己的任务编码不改动作策略原task_id。

联合拟合6187已知训练查询（原5794＋水槽393），开发1338（原1243＋水槽95）。新水槽训练行损失权重4，原训练行及标签保持。一个seed100、MLP64、AdamW lr.001/wd.001，沿用原训练器150轮上限/30轮开发停止；检查点按五任务等权dev macro F1选择。全局availability偏置0/4及置信阈值网格只在开发选择，最终offset0、threshold.4、margin0。测试不参与模型/阈值选择。

## 冻结结果及限制

| 指标 | 统一模型 |
|---|---:|
| 全开发正确/已选 | 1220/1279 |
| 全开发选框Precision / Recall | 95.39% / 92.15% |
| 五任务开发macro F1 | 94.23% |
| 原四任务开发正确 | 1125（原冻结模型1134） |
| 原四任务开发抓碗错类 / 持碗错类 | 4 / 1（原2 / 0） |
| 新水槽测试正确/已选 | 90/90 |

不隐瞒旧四任务正确数减少9和核心错类增加。水槽测试仅4条新episode、90个已知查询，标签均为抓碗/携碗代理；100%只是这组弱标签记录，不是总体可靠性或任务成功率。按用户接受短暂错框并推进训练的偏好，生成此研究版缓存，不将旧门槛当成已通过。没有新动作策略闭环成功率。

导出包只有一个权重文件。732条水槽公共特征逐帧重放，对原checkpoint概率max差0、接受槽0差。bundle manifest SHA：c68b54d8a3c3b544b76f0d47ad14feb015fd360834657b6a64c13eb1619f2f74。初始导出README仍写四任务的包装错误已修，另导出bundle_five_task，旧包原样保留。checkpoint和选择未因包装修正而重训或重新调阈值。

## 全量缓存流程

已逐源核对原Zarr训练frame_map、demo、episode边界、感知缓存连续帧数及冻结检测器/ROI权重和全部metadata一致，十源654460帧均可复用已有Mask候选/ROI，避免重跑检测。冻结主RGB层2全局特征由GPU1 batch128提取；后续四个CPU进程各管独立数据源，计算原始主depth与自身proprio/torso的当前/t−8几何、同一个五任务指针选框及valid。t−8严格夹在同一episode内；预存在磁盘的未来全局特征不被访问。未处理腕部像素或读取目标参考标签。

每个新sidecar独立保存boxes、valid、class_id、confidence；绑定训练源/映射/episode/冻结模型/前端协议SHA。每源独占写锁，512帧或episode尾提交完整前缀；中断可恢复，实际支持的指针为前馈网络，拒绝用同一断点协议跳过GRU隐藏状态。旧数据、旧红碗缓存和旧策略输出不改写。最终四个数组逐源SHA256及两种策略Dataset加载、保留原其他模态/动作、原始主传感器重算将单独核验。

十源训练帧数：

| 数据源 | 帧数 |
|---|---:|
| unload_dishwasher_to_dresser_suboptimal | 83175 |
| unload_dishwasher_to_dresser_expert | 65586 |
| table_cleanup_to_sink_suboptimal | 87120 |
| table_cleanup_to_sink_expert | 52016 |
| table_cleanup_to_dishwasher_suboptimal | 110614 |
| table_cleanup_to_dishwasher_expert | 69813 |
| table_setup_from_dishwasher_suboptimal | 44003 |
| table_setup_from_dishwasher_expert | 41340 |
| table_setup_from_dresser_suboptimal | 53540 |
| table_setup_from_dresser_expert | 47253 |

前端协议SHA：81d06fd88293c717e101cf8eb272a9d2297032b793cd19d775f6336947c51aad。

代码b89cdd4。58项相关测试通过，6条既有依赖弃用警告；包括一套runtime覆盖十源、五任务编码、四/五任务schema兼容、模拟中断恢复、episode过去帧边界以及旧多模态两结构前反向。实际十源自身机器人schema预检均通过，只从state中提取自己的torso列，不把场景状态送入推理。

缓存目录：/home/jrw/vpwem_data/momart_target_unified_five_task_20261103。模型目录：/home/jrw/vpwem_outputs/momart_target_research_20261101/unified_five_task_20261103。十源新框缓存已全部完成并通过完整性和两种策略输入检查，原始传感器30帧抽样重算通过。正式动作策略训练尚未启动。

## 全部完成后的验收

十源均complete=true，完整提交帧数合计654460；boxes/valid/class_id/confidence四数组逐源SHA256全部复核通过。新框有效564371帧（有效比例约86.24%，只是供框覆盖率，不是准确率）。两种策略输入各每源抽4个索引，共80个实际样本，包含初帧、首episode末帧、中间及全源末帧；腕部RGB/depth、主depth、scan、proprio和动作与原No-History逐数组相同。最新帧坐标/valid对应缓存自己的物理帧，缺框保持原RGB；两种模式均已实际消费十源缓存。

| 数据源 | 完整帧数 | 有效选框帧数 | 两种输入/四数组校验 |
|---|---:|---:|---|
| unload_dishwasher_to_dresser_suboptimal | 83175 | 71398 | 通过 |
| unload_dishwasher_to_dresser_expert | 65586 | 54324 | 通过 |
| table_cleanup_to_sink_suboptimal | 87120 | 82483 | 通过 |
| table_cleanup_to_sink_expert | 52016 | 48144 | 通过 |
| table_cleanup_to_dishwasher_suboptimal | 110614 | 97363 | 通过 |
| table_cleanup_to_dishwasher_expert | 69813 | 62110 | 通过 |
| table_setup_from_dishwasher_suboptimal | 44003 | 36055 | 通过 |
| table_setup_from_dishwasher_expert | 41340 | 35096 | 通过 |
| table_setup_from_dresser_suboptimal | 53540 | 43400 | 通过 |
| table_setup_from_dresser_expert | 47253 | 33998 | 通过 |

独立从原始主RGB、主depth、自身proprio和torso重新跑batch1检测/视觉特征/几何/统一选框，每源首/中/尾各1帧，共30帧：valid分歧0、类别分歧0、有效框最小IoU 0.99964682340576、归一化坐标max差 5.733966827392578e-05（最多约0.007像素）。检测缓存原批量推理和在线batch1存在微小数值差，不能宣称逐bit相同或全654460帧在线重算均已验证。

最终产物：缓存根目录target_metadata.json（两种策略共同读取）、十个SOURCE.bbox.zarr、summary.json、每源verified.json及global特征/进度/机器人schema；全量完整性报告ten_target_full_integrity_20261103.json、原始传感器重算ten_target_full_raw_replay_20261103.json在research输出目录。生成器独占GPU1特征提取，CPU四源并行，同一冻结模型无分支/回退。旧源/检测缓存只读。

代码b89cdd4（统一模型及生成器）、804b3d9（源完整性与原始传感器复算工具）。没有启动动作策略训练；在线模拟器与动作策略的最终接线仍待完成。长期研究目标仍active，本次十源缓存任务已完成。
