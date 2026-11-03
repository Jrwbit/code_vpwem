# 当前选框器的复现与在线训练准备核验，2026-11-03

用户认为视频中的短暂错误可接受，希望推进训练，并询问是否可复现以及在线评测能否达到相同选框表现。结论：现有固定选框器可以复现，并已有因果逐帧API；两条视频的原始观测现场重算也得到相同最终框，可以进入研究训练。尚未证明整套动作策略在线任务成功率，不能把示范回放当闭环。

## 当前版本

transition4_new0.75固定配置，两geometry MLP：原模型25%与切换4倍训练模型75%，各使用原固定可用性偏置4，最终threshold.6/margin0。全部权重、归一化、任务词表、前处理和源码可固化；本轮未重训或改模型/阈值，不把新核验当新算法。

输入仅主相机RGB、主深度、当前及t−8观测、任务和机器人自身proprio/torso；机器人几何由URDF和自身测量生成。无侧面/腕部增强，无未来帧、未来动作、人工目标框、目标类别真值或物体GT位姿。绿色框只供外部评分/审阅，在线选框接口不接收它。原始重放器在HDF中按已核验schema只取自身躯干关节列，在线应由环境/机器人自身trunk joint测量提供，不读整场景状态。

额外检查iGibson源码fetch_gripper_robot.get_proprio_obs/grasped：proprio中的grasped是夹爪关节开度阈值及5步滤波产生的正负标记，不是被抓物体类别或实例ID；不能把它写成已知目标碗身份。语义组织环境get_state可提供该proprio字典，含head/trunk/eef自身状态。

## 本轮直接证据

| 核验 | 范围 | 结果 |
|---|---|---|
| 缓存公共特征逐帧API vs 离线批量 | 146条轨迹/10592查询，含未知标签观测 | 接受槽0差，概率max2.54e−7；每步重复调用幂等 |
| 搬移新导出包、python -I、空工作目录 | 同上10592查询，仅包内API+公共输入 | 接受槽0差，概率0差；代码/权重/配置逐文件SHA一致 |
| 两条完整视频，从原始传感器重做检测/ROI/global/几何/选框 | demo109全部49查询（0..384）、demo42全部169查询（0..1344），合218查询 | 最终接受槽0差；归一化框坐标max差0.0003053，折合原相机任一坐标小于0.04像素 |

原始重算与缓存不是逐bit一致：218查询中2次候选有效槽集合不同，最大概率差0.02551，但所有最终框选择一致。具体为demo109 step104、demo42 step160，原始单帧检测与归档批处理存在数值/提案边界差异；本轮没有进一步证明是哪一个过滤环节触发，不归因未核验原因。不得把这些特征/概率说成完全一致。

物理卡1、包括感知/特征/选择器的原始重算中位40.14ms、P95 57.34ms；首次648.58ms包含冷启动/模型加载影响，不能拿预热后中位当首次耗时。这不含传感器采集和动作策略推理，也不是整个控制环频率保证。缓存选框器单独中位1.12ms/P95 1.22ms，不能用这个代替完整感知延迟。

## 可搬移包与一次版本问题

重新导出包runtime_bundle_current_20261103，共9个文件受哈希保护，保留原模型权重，含runtime/models/availability/calibration源码、相对权重路径、任务词表、阈值和manifest。包manifest SHA57b69f3279fcbf0b97ef34a6177c471930b91a15a549b5a00190cda0b404b029，原选择配置SHA8d84bb4e0eb649daec882481c9b60c874b7a03818907cebe6d2c62c5960a0a5a。

先检查历史runtime_bundle_confirm6时，当前验证脚本新增无条件import calibration，而旧冻结包没有该文件，发生ModuleNotFoundError；旧包/旧910帧历史验证保留。未掩盖失败，使用现有导出器重新导出一致版本，再按同样10592帧搬移隔离检查全部通过。新包不是检测器或动作策略；raw主视角前端及其原检测器/ROI权重仍必须另外按配置加载。

## 如何接训练与在线评测

1. 先冻结上述选择器和原感知权重，以同一代码路径给训练样本与在线观测产生框、valid和可选置信度。训练时不把绿色真值框当模型预测，也不在在线阶段读取训练缓存。
2. 在线维护恰好t−8的自身观测缓冲，保持原相机/深度预处理、任务词表、归一化和查询时钟；每条新episode清空历史。实时主视角检测生成候选，再调用SingleBoxSelector，接受一个框或明确valid=false。
3. 将框/valid真正接入动作策略输入，再训练带条件策略；与相同训练预算的无框策略做闭环对照，记录任务成功、错误后的恢复及延迟。现有A1多对象bank还不是本单框adapter，当前尚无已经接入单框的策略checkpoint或在线任务成功率。

离线旧完整门槛未通过不等于禁止研究训练；按用户当前偏好，允许带着这些短暂选框错误推进训练，不要求框错误先归零。研究候选身份和历史失败仍如实保留，不把“可开始训练”改写为“正式部署可靠”。新轨迹或动作偏离示范时不能保证继续与视频逐帧相同，需实际闭环检验。

## 复现命令与证据

- verify_runtime.py --root OUTPUT_ROOT --selection CURRENT_SELECTED --growth --output NEW_JSON
- export_runtime_bundle.py --root OUTPUT_ROOT --selection CURRENT_SELECTED --output NEW_BUNDLE
- verify_runtime_bundle.py --root OUTPUT_ROOT --selection CURRENT_SELECTED --bundle NEW_BUNDLE --growth --output NEW_JSON
- verify_current_raw_replay.py --root OUTPUT_ROOT --config ORIGINAL_GPU_CONFIG --urdf FETCH_URDF --output NEW_DIRECTORY --device cuda:0

原始重算入口默认固定核验刚交付的confirm7/demo109与confirm9/demo42，写入配置/URDF SHA和因果输入协议，逐查询保存实际/归档结果及耗时；检查原始RGB字节SHA与manifest一致，读取原归档特征仅供外部比较。没有运行新仿真或修改示范动作，没有使用确认10。

77项相关测试通过，5条既有依赖弃用警告；新原始重放CLI导入检查通过，218原始查询为直接集成验证。运行JSON、原图、权重和缓存不入Git。目标仍active，下一阶段重点是单框动作策略训练/闭环，而不是继续以短暂错框为由无限离线搜索。
