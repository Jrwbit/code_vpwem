# 中断位置原路返回成功与新检测路线准备（2026-10-21）

## 结果与更正

用户反映当前既不能复位也不能检测。本轮已实际将左臂从中断位置退回保存home，最终最大单关节误差约0.03808°，原生completed、safe_exit_confirmed=true，三部件下电手动空闲、busy=false、latched=false。随后重新采集RGB-D并完成型号132三维识别，生成并装载新的完整检测路线，状态ready、collision_checked=true。**没有执行新路线的检测对位或电测，不宣称检测工作全部完成。**

本轮开始时腰部第4轴已经回到接近工作目标的位置，约相差0.0365°，与上一轮−514失败后相差1.55°的状态不同。左臂仍在原中断停点。本轮没有发腰部运动或清警，不能将这次腰姿变化归因于上一轮失败的复位命令，也不能据此宣称身体控制器上电故障或被动位移根因已修复。

最新普通复位任务 `963ccaed44394fa5b225278c51d0f77c` 失败于 `detection_initial_segments → fewest_connect`，报“连不上（0个中间点从起点可达，2个可到终点）”，没有retreat_source。它是在从当前点直接搜索收臂路线；不能把此失败等同于所有返回路线都不存在。

## 路线来源与只读校验

来源为本次原完整路线 `6ef0a844c8e341c29b2c5af82683390e`，计划摘要 `d173b6b2a21d93d4f5c53e1130f9df078a8b7c3e190eacb1f5d0a7a22c4907da`。逐点对照原生日志 `20261021_103353_8244bd4a0fdf.jsonl`，确认前6个点完成，第7点未完成；没有通过最近位置或扫描历史记录猜选计划。

新增只读工具 `tools/jinlong_hmi/check_interrupted_return.py`，校验完成事件顺序、目标与原计划一致，核对当前运动学、静止状态、腰姿、头部/右臂、控制器限位及原几何来源。先从实际停点接回最后完成点，再逆序经过已走过的6条关节线。对7条线使用当前实测腰姿重新做碰撞检查，保留原阶段adjust/corridor规则，不使用全局忽略碰撞或新的插接豁免。模型最小间隙约11.006995 mm；模型间隙不等于实物测量。

当前复位输入与原检测输入的site、calibration.sha256、camera_T_box一致，作为本次反向返回的场景来源。本轮最终新检测另行重新采图，没有将旧视觉当成新采集。

## 两次返回执行及必要修正

| 执行 | 结果 | 原生记录 |
| --- | --- | --- |
| 原路7点返回，10 mm/s | 前5点完成；第6条大转角线触发route waypoint timeout；安全退出确认成功，未回到home | `20261021_105903_a2c53f315aea.jsonl` |
| 从上述实际停点继续剩余原线，每点不超过10°，10 mm/s | 16个短路点全部完成，回到保存home，safe_exit_confirmed=true | `20261021_110208_d5c844bea543.jsonl` |

第一次降低速度后，大转角单点超过既有120秒时限。第二次没有调大超时或保护门限，而是严格在线性关节插值上细分剩余路线，重新检查全部16条短线（模型最小间隙约11.008765 mm），并再次核对实际起点、工位、标定及完成记录，再通过HMI唯一owner执行。执行中保持所属计划心跳，配置变化或异常时停止。

`tools/jinlong_hmi/finish_interrupted_return_20261021.py` 保留第二次执行源记录，绑定本次故障、原生完成进度、工位和标定；不是任意故障自动重试入口。执行完成后重新运行会因现场失败身份/状态已改变而拒绝。本轮没有将此现场恢复扩展成生产页面的通用“故障续检”功能。

## 新视觉与新计划

- 新视觉任务：`20261021_110338_e7f934`，沿用原方案 `rokae_depth_fast_recovery` 和现场原参数，重新采集RGB-D；state=completed、success=true、型号132、quality.ready=true。
- 新视觉接线盒中心（左臂SDK基座，mm）：`[243.782104, −450.449026, 405.647298]`；平面RMSE约0.20215 mm。
- 新完整路线：`f5d3340e18dc4c42b4087038ad38e30b`，ready、collision_checked=true，8阶段20个关节点，GPU搜索结果经CPU检查；模型最小间隙约10.0 mm（含原插接区域规则），没有靠偏移常数或放宽限位放行。
- 页面当前装载 `inspection_f5d3340e18dc4c42b4087038ad38e30b.json`；新路线仍使用现场原设置：中转30 mm/s、接近/对位20 mm/s，不把本轮返回的10 mm/s谎称为已持久保存的检测速度修改。
- 最终三部件均off/manual/idle，反馈新鲜，最后一次实际运动completed、安全退出确认成功。新路线尚未启动。

三维结果仍保留 `site_motion_accepted=false`、`robot_ready=false` 和独立实测验收未通过的元数据；规划入口生成ready不等于将这些验收记录改为通过。没有电测接口，本轮不报告电测结果。

## 验证、证据与限制

原生完成事件绑定的3项本地测试通过，覆盖有序完成前缀、错序/异源/非有限目标拒绝、空/全完成记录拒绝；Python语法编译与git diff --check通过。另有现场两次全路径几何复核和真实运动完成回执，分别记录首次超时和第二次成功，未隐去失败。

现场证据目录：`/home/bit-bhr/helios_hmi_sessions/diagnostics/fault-resume-20261021/`，含current_return_snapshot.json、reverse_validation.json、return_execution_plan.json、return_execution_final.json、return_split_plan.json、return_split_validation.json、return_split_final.json、fresh_vision.json、fresh_trajectory.json、delivery_state.json。运行数据、模型、图像、程序输出及私有状态不入Git，不将本地路径冒充飞书附件。

本轮只增加辅助校验工具、单次恢复执行记录和文档，没有修改生产源码、保护参数、工位配置或验收标志，也没有部署通用断点续检入口。腰部原先上电失败、身体第4轴异常的根因尚未关闭；返回完成及新路线ready不能作为其已修复的证据。

本文追加到飞书既有全流程报告，保留原正文，回读核对新结果与历史正文；报告和源工具均提交Git。
