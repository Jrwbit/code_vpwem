# 电机检测项目近两周图文工作汇报

统计范围：2026年10月16日—10月29日（北京时间）；本机最新现场证据截至10月28日。覆盖金龙／台州160、132、100电机检测。

已完成多次实机机械往返、检测流程修复、右臂23组手眼标定切换和三型号真实RGB-D采集；最新三型号实测目标已通过共同腰姿下的离线非接触完整往返。插接、压入、电测及三型号实机全流程仍待验收。

本报告以原始现场图片、界面截图和反馈曲线为主；仿真图片均单独标注。历史图片用于说明当时工作，不代表当前实时状态。

## 01｜10月17—21日：定位、运动学与实机往返

统一控制器有效运动学参数和定位链路，阶段性切换15组标定；修复路线各段起点不一致、复位与中断返回。10月20日完成三轮共78步机械往返，21日又完成20目标及新视觉22目标循环。

![图1：10月20日·反馈分析｜模型与控制器运动学口径对照，减少规划与执行坐标差异。](../outputs/diagnostics/kinematic-alignment-20261020/feedback_semantics.png)

图1｜10月20日·反馈分析｜模型与控制器运动学口径对照，减少规划与执行坐标差异。

![图2：10月20日·现场视觉｜15组标定阶段的新接线盒识别及坐标轴；该标定已于28日被23组右臂标定替换。](../outputs/diagnostics/calibration15-new-inspection-20261020/new15_detection.jpg)

图2｜10月20日·现场视觉｜15组标定阶段的新接线盒识别及坐标轴；该标定已于28日被23组右臂标定替换。

![图3：10月20日·实机反馈｜连续三轮完整往返共78步；同一视觉结果下的反馈重复性通过，不能代替独立实物精度验收。](../outputs/diagnostics/repeat3-20261020/analysis/three_rounds_trace.png)

图3｜10月20日·实机反馈｜连续三轮完整往返共78步；同一视觉结果下的反馈重复性通过，不能代替独立实物精度验收。

![图4：10月21日·实机反馈｜统一实际反馈起点后完成到位、约20秒停留及原路返回；停留不等于电测完成。](../outputs/diagnostics/shared-planning-anchor-20261021/inspection-feedback-trace.png)

图4｜10月21日·实机反馈｜统一实际反馈起点后完成到位、约20秒停留及原路返回；停留不等于电测完成。

## 02｜10月19—23日：操作流程和现场入口修复

完善检测位撤离、重复复位与重启后的返回来源；新增示教保存入口，取消历史示教记录的启动门槛，扫码型号自动匹配工装；修复金龙和台州桌面入口。

![图5：10月20日·页面实测｜恢复原四段返回路线生成；截图中的旧执行锁是当时状态。](../outputs/diagnostics/taizhou-reset-20261020/return_ready.png)

图5｜10月20日·页面实测｜恢复原四段返回路线生成；截图中的旧执行锁是当时状态。

![图6：10月21日·页面实测｜示教参考和保存入口部署到现场，支持操作留档。](../outputs/diagnostics/teaching-record-ui-20261021/teaching-record-ui.png)

图6｜10月21日·页面实测｜示教参考和保存入口部署到现场，支持操作留档。

![图7：10月21日·页面实测｜历史示教不匹配时仍可进入路线运行确认。](../outputs/diagnostics/optional-teaching-gate-20261021/route-ready.png)

图7｜10月21日·页面实测｜历史示教不匹配时仍可进入路线运行确认。

![图8：10月23日·页面实测｜台州检测桌面入口恢复，浏览器正常进入任务选择页。](../outputs/diagnostics/desktop-entry-20261023/taizhou.png)

图8｜10月23日·页面实测｜台州检测桌面入口恢复，浏览器正常进入任务选择页。

## 03｜10月22—26日：扫码、共用工位与非接触路径研究

验证160扫码定位，核对工装方向、盒体几何和相机禁入区，探索三型号共同工位及固定腰姿。下列23日工位／抬高方案属于历史离线候选，28日已根据新标定和实测目标重求。

![图9：10月22日·现场视觉｜160二维码与接线盒识别成功；其他失败帧另有深度支撑问题。](../outputs/diagnostics/qr160-20261022/success.jpg)

图9｜10月22日·现场视觉｜160二维码与接线盒识别成功；其他失败帧另有深度支撑问题。

![图10：10月23日·实物资料核对｜100连续三柱、工装长边与法兰方向对照，纠正配合方向理解。](../outputs/diagnostics/actual-terminal-alignment-20261023/archive/jl100-edge-audit/100长边与法兰坐标核对.png)

图10｜10月23日·实物资料核对｜100连续三柱、工装长边与法兰方向对照，纠正配合方向理解。

![图11：10月23日·离线投影｜仅抬高相机80mm的历史候选，三型号非接触往返模型检查通过；未据此移动现场相机。](../outputs/diagnostics/camera-lift-20261023/final-reproduce/camera/shared_station_camera.png)

图11｜10月23日·离线投影｜仅抬高相机80mm的历史候选，三型号非接触往返模型检查通过；未据此移动现场相机。

## 04｜10月26—28日：上电复核、相机恢复和右臂标定

26日左臂／腰部各约20秒原位上电保持及退出通过；28日恢复网络反馈和相机新帧，将彩图曝光500000调为100000μs。修复手动采样错误记录左臂的问题，完成23组右臂标定并切换活动外参。J4失电不保持已由用户确认为正常特性，后续自动流程需持续受监控持腰，不能沿用早期“必为抱闸故障”的推断。

![图12：10月28日·现场原图｜调整前彩图曝光500000μs。](../outputs/diagnostics/right-handeye-20261028/exposure-before.jpg)

图12｜10月28日·现场原图｜调整前彩图曝光500000μs。

![图13：10月28日·现场原图｜曝光100000μs后稳定检测40角点，改善标定板可观测性。](../outputs/diagnostics/right-handeye-20261028/exposure-final.jpg)

图13｜10月28日·现场原图｜曝光100000μs后稳定检测40角点，改善标定板可观测性。

![图14：10月28日·标定样本｜选臂修复后新右臂有效配对样本，72角点；随后完成23组右臂解算和切换。](../outputs/diagnostics/right-handeye-20261028/right-first-valid.jpg)

图14｜10月28日·标定样本｜选臂修复后新右臂有效配对样本，72角点；随后完成23组右臂解算和切换。

## 05｜10月28日：三型号真实RGB-D采齐，修正深度过滤范围

160盒口约388mm，原400mm下限会过滤盒沿。对照查明原因后，用户批准改为350—800mm，正式定位复核通过。三型号原始彩图均能解码实际二维码，下列每型彩图和深度预览来自对应完整原始帧。

![图15：10月28日·160真实彩图｜二维码JM160-001解码成功，盒口中心光轴距离约387.783mm。](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JM160/depth350-diagnostic/fusion/trigger_000001/01_rgb_original.png)

图15｜10月28日·160真实彩图｜二维码JM160-001解码成功，盒口中心光轴距离约387.783mm。

![图16：10月28日·160同帧深度预览｜用于盒口平面及位姿估计；伪彩是深度显示，不能直接当绝对精度验收。](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JM160/depth350-diagnostic/fusion/trigger_000001/04_depth_pseudo_auto.png)

图16｜10月28日·160同帧深度预览｜用于盒口平面及位姿估计；伪彩是深度显示，不能直接当绝对精度验收。

![图17：10月28日·132真实彩图｜二维码JL132-001解码成功，盒口中心光轴距离约473.734mm。](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JL132/full-evidence/fusion/trigger_000001/01_rgb_original.png)

图17｜10月28日·132真实彩图｜二维码JL132-001解码成功，盒口中心光轴距离约473.734mm。

![图18：10月28日·132同帧深度预览｜用于盒口平面及位姿估计；伪彩是深度显示，不能直接当绝对精度验收。](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JL132/full-evidence/fusion/trigger_000001/04_depth_pseudo_auto.png)

图18｜10月28日·132同帧深度预览｜用于盒口平面及位姿估计；伪彩是深度显示，不能直接当绝对精度验收。

![图19：10月28日·100真实彩图｜二维码JL100-010解码成功，盒口中心光轴距离约550.934mm。](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JL100/full-evidence/fusion/trigger_000001/01_rgb_original.png)

图19｜10月28日·100真实彩图｜二维码JL100-010解码成功，盒口中心光轴距离约550.934mm。

![图20：10月28日·100同帧深度预览｜用于盒口平面及位姿估计；伪彩是深度显示，不能直接当绝对精度验收。](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JL100/full-evidence/fusion/trigger_000001/04_depth_pseudo_auto.png)

图20｜10月28日·100同帧深度预览｜用于盒口平面及位姿估计；伪彩是深度显示，不能直接当绝对精度验收。

## 06｜最新结果：三型号实测目标下的共同腰姿候选

28日使用最新23组右臂标定、三型号实测盒位及当次右臂／头部姿态，求得模型腰角[-35,30,20,-90]°。从共同等待位非接触接近、撤离和返回均通过离线复核：160为1250点、132为3862点、100为3304点，共8416点，腰变化0°。目标停在名义配合位前40mm；本轮未下发机器人运动。

| 项目 | 截至10月28日的结论 |

|---|---|

| 132普通工况机械往返 | 已有多次实机成功记录 |

| 三型号扫码与RGB-D | 同一新标定下真实采集完成，三型二维码解码通过 |

| 三型号共腰完整非接触循环 | 实测目标下离线模型通过，尚未完成三型实机复测 |

| 进入／退出新工作腰姿 | 待验证，模型角不可直接当控制器指令 |

| 真实插接、轻触补偿压入、电测 | 未完成验收 |

## 07｜下一阶段工作

先核实实物几何、足底定位和台面，清理100画面右侧备用132或将其建模；完成受监控持腰和工作腰姿进入／退出，再做低速非接触实机复核。之后以独立量具验收定位和端子配合，推进轻触、补偿压入、电测握手与结果记录。

## 证据索引与历史归档

既有全流程报告：https://feishu.cn/docx/I2GAdun3nofskUxAOjWcxVwjnJO 。本次独立图文汇报不替换历史正文；最新判断以10月28日实测报告为准。

- 本机证据报告：docs/taizhou_controller_kinematics_alignment_20261020.md

- 本机证据报告：docs/jinlong_calibration15_switch_and_inspection_attempt_20261020.md

- 本机证据报告：docs/jinlong_three_round_repeatability_20261020.md

- 本机证据报告：docs/jinlong_shared_planning_anchor_execution_20261021.md

- 本机证据报告：docs/taizhou_original_return_restore_fix_20261020.md

- 本机证据报告：docs/jinlong_teaching_record_ui_20261021.md

- 本机证据报告：docs/jinlong_optional_teaching_gate_20261021.md

- 本机证据报告：docs/jinlong_taizhou_desktop_entry_fix_20261023.md

- 本机证据报告：docs/jinlong_qr160_diagnosis_20261022.md

- 本机证据报告：docs/jinlong_actual_terminal_alignment_20261023.md

- 本机证据报告：docs/jinlong_camera_vertical_lift_20261023.md

- 本机证据报告：docs/jinlong_right_handeye_preparation_20261028.md

- 本机证据报告：docs/jinlong_right_handeye_manual_capture_fix_20261028.md

- 本机证据报告：docs/jinlong_calibration_rgbd_git_archive_20261028.md

- 本机证据报告：docs/jinlong_measured_common_waist_20261028.md

- 本机证据报告：docs/jinlong_stationary_power_check_20261026.md

- 本机证据报告：docs/jinlong_new_calibration_waist_assessment_20261028.md

飞书图文文档：https://feishu.cn/docx/FNjddpBU8oZHrHx6hpOcLRwunjg

同步核验：正文关键结论回读通过；20张嵌入图片逐张下载与本机源图SHA-256一致。既有全流程报告已追加专报入口，原正文块顺序保留。
