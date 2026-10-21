# 2026-10-21 当前检测路线失败：阶段接缝不连续，并非目标逆解无解

用户询问当前检测状态机械臂是否也无解。读取现场状态及实际规划日志后，结论：最新视觉检测成功，GPU已经给出路线并通过CPU复核，随后在整条路线拼接的连续性校验处失败，不能归类为目标逆解无解。

- 最新视觉任务：`20261021_170537_bf642b`，全流程 completed，success=true，geometry_pose_ready、center_pose_ready、base_pose_ready及vision_robot_ready均为true。
- 最新路线：`59d35730737d4602bcb12c6ed560d99a`，inspection，state=failed，next_segment=0。使用15组ANDREFF标定，SHA仍为 `e8a7ee509c0764c6214df5d424f40d1f93f28e0d02232443a322ffdb8d2d2c9b`。
- 实际日志：GPU检查5365个位姿，回执cpu_review_passed=true，随后记录“GPU corridor passed CPU review”。“Corridor candidates: 0”是GPU已有可用结果时跳过另一条候选枚举分支的结果，不代表全部逆解失败。
- 最终错误：“轨迹阶段接缝不连续，拒绝导出：检测腰姿保持，左臂收回home → home → GPU左侧中转避障，最大关节差0.036765°”。
- `vision/app/hk_pointcloud_registration/robot/corridor_worker.py:139` 的 `require_continuous_segments()` 对相邻11轴路点端点使用最大绝对差大于1e-8 rad的条件拒绝导出。当前是两个阶段使用的边界关节值不完全一致；0.036765°为关节角差，不是末端毫米误差。

本次证据说明当前路线的拼接失败；尚未追溯到具体哪个关节或哪一处生成边界值的代码产生差异，也未据此修改阈值或程序。合适的修复方向是统一相邻阶段的实际边界状态，并检查连接段及整条路线，而不是将现象解释为目标不可达。

现场读取时机器人connected=true、busy=false、latched=false，各部件下电／手动／空闲，未启动本次失败路线。历史示教记录required_for_start=false，不是本次失败来源。此轮仅排查并记录结论，没有修改机器人代码或发运动指令。
