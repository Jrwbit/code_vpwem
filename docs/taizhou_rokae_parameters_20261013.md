# 台州检测参数映射（2026-10-13）

本映射按本轮实际字段目录、参数验证、命令构造、设置消费代码和路径规划整理。检测根目录见同日审计记录；下述路径以 `app/hk_pointcloud_registration/` 为相对根。保存成功与现场验收是不同条件。

## 工位参数：36 个表单项

入口 `web/rokae_parameters.mjs`（毫米／角度显示与转换）及高级 JSON → POST `/api/robot/config`（expected_revision）→ `robot/rokae_site.py::save_site/validate_site` → `config/rokae_site.json`。默认来源 template；未知扩展保留，前后历史在 outputs/rokae_site_history。

| 界面字段 | 配置键 | 实际消费代码／用途 | 单位／坐标系 | 验证 |
|---|---|---|---|---|
| 法兰 → 实际TCP：XYZ毫米，RPY角度 | `flange_T_tcp` | rokae_planner.build_stages；rokae_site.workspace_check：中心／探针补偿 | 存储米制刚体矩阵；表单 XYZ mm、RPY °；法兰→TCP | 有限4×4、无布尔／字符串、末行、正交旋转 det=1；几何改动清验收；补偿和全路径测试 |
| TCP → 探针尖端：XYZ毫米，RPY角度；Z可填写沿TCP正Z的探针长度 | `tcp_T_probe` | rokae_planner.build_stages；workspace_check：探针尖端 | 存储米制刚体矩阵；表单 XYZ mm、RPY °；TCP→探针 | 有限4×4、无布尔／字符串、末行、正交旋转 det=1；几何改动清验收；补偿和全路径测试 |
| 探针直径（mm；端盖三孔间隙检查） | `probe_diameter_mm` | rokae_planner：只用于端盖三孔孔径／单边余量 | mm | 正有限；端盖间隙不足拒绝；四孔未使用该直径门 |
| 端盖孔插入深度（mm） | `probe_insert_depth_mm` | rokae_planner：端盖 EMPTY 孔的尖端插入目标 | mm，沿孔局部轴 | 0..200，明确EMPTY与独立接触工艺确认；未知螺钉状态拒绝 |
| TCP → JL100 工装目标：XYZ毫米，RPY角度 | `tcp_T_objects.JL100` | rokae_planner.build_stages：仅对应型号中心工装 | 存储米制刚体矩阵；表单 XYZ mm、RPY °；TCP→该型号目标 | 有限4×4、无布尔／字符串、末行、正交旋转 det=1；几何改动清验收；补偿和全路径测试 |
| 接线盒中心修正（检测盒局部XYZ，mm；仅中心流程） | `box_center_adjustments_mm.JL100` | rokae_planner：右乘 box_T_center_adjustment，仅中心 | mm，检测盒局部 XYZ | 三个有限数值±200；不移动孔位回归 |
| 工装高度修正（TCP局部Z，mm；不改变原工装矩阵） | `fixture_z_adjustments_mm.JL100` | rokae_planner：对应 tcp_T_object 的 Z 加一次，仅中心 | mm，TCP 局部 Z | 有限数值±200；不同工装旋转的补偿回归 |
| TCP → JL132 工装目标：XYZ毫米，RPY角度 | `tcp_T_objects.JL132` | rokae_planner.build_stages：仅对应型号中心工装 | 存储米制刚体矩阵；表单 XYZ mm、RPY °；TCP→该型号目标 | 有限4×4、无布尔／字符串、末行、正交旋转 det=1；几何改动清验收；补偿和全路径测试 |
| 接线盒中心修正（检测盒局部XYZ，mm；仅中心流程） | `box_center_adjustments_mm.JL132` | rokae_planner：右乘 box_T_center_adjustment，仅中心 | mm，检测盒局部 XYZ | 三个有限数值±200；不移动孔位回归 |
| 工装高度修正（TCP局部Z，mm；不改变原工装矩阵） | `fixture_z_adjustments_mm.JL132` | rokae_planner：对应 tcp_T_object 的 Z 加一次，仅中心 | mm，TCP 局部 Z | 有限数值±200；不同工装旋转的补偿回归 |
| TCP → JM160 工装目标：XYZ毫米，RPY角度 | `tcp_T_objects.JM160` | rokae_planner.build_stages：仅对应型号中心工装 | 存储米制刚体矩阵；表单 XYZ mm、RPY °；TCP→该型号目标 | 有限4×4、无布尔／字符串、末行、正交旋转 det=1；几何改动清验收；补偿和全路径测试 |
| 接线盒中心修正（检测盒局部XYZ，mm；仅中心流程） | `box_center_adjustments_mm.JM160` | rokae_planner：右乘 box_T_center_adjustment，仅中心 | mm，检测盒局部 XYZ | 三个有限数值±200；不移动孔位回归 |
| 工装高度修正（TCP局部Z，mm；不改变原工装矩阵） | `fixture_z_adjustments_mm.JM160` | rokae_planner：对应 tcp_T_object 的 Z 加一次，仅中心 | mm，TCP 局部 Z | 有限数值±200；不同工装旋转的补偿回归 |
| X范围（左臂SDK基座，mm） | `workspace_m.x` | rokae_site.workspace_check：每个法兰/TCP/探针采样 | 存储 m，显示 mm，左臂 SDK 基座 | 三轴上下界、有限数值、全路径越界拒绝；现场净空仍需核对 |
| Y范围（左臂SDK基座，mm） | `workspace_m.y` | rokae_site.workspace_check：每个法兰/TCP/探针采样 | 存储 m，显示 mm，左臂 SDK 基座 | 三轴上下界、有限数值、全路径越界拒绝；现场净空仍需核对 |
| Z范围（左臂SDK基座，mm） | `workspace_m.z` | rokae_site.workspace_check：每个法兰/TCP/探针采样 | 存储 m，显示 mm，左臂 SDK 基座 | 三轴上下界、有限数值、全路径越界拒绝；现场净空仍需核对 |
| 7轴初始参考（°）；当前返回本次起点，不自动回此关节参考 | `initial_joints_rad` | motion_failures：需完整配置；返回用本次真实起点，不自动回此参考 | 7轴 rad；显示 ° | 7个有限非布尔数值；没有回参考关节功能 |
| 关节速度（%，当前固定1） | `speed_percent` | 运动门：固定1%；HMI/native原有速度限制 | % | 只读；只允许数值1 |
| 线速度（mm/s，最大10） | `linear_speed_mm_s` | rokae_controller → HMI linear preview/send → MoveL | mm/s | (0,10]，有限非布尔；SDK下发再验证 |
| 沿检测物体轴的接近距离（mm） | `approach_mm` | rokae_planner：中心轴／孔轴沿法向接近 | mm，物体／孔的局部轴 | 保存0..200，执行必须>0；0空载接近拒绝 |
| 过渡抬升距离（mm） | `transit_clearance_mm` | rokae_planner：安全过渡和抬升 | mm，按基座方向向量 | 0..200；各段全路径工作空间检查；现场路径确认 |
| 抬升／过渡方向（当前左臂基座单位向量） | `transit_direction_base` | rokae_planner：方向×clearance | 基座单位向量，3项 | 有限3项、长度1；不继承旧UR方向 |
| 每个执行阶段停留（s；包含接近、撤退和返回） | `dwell_s` | rokae_controller：每个执行阶段后有guard的停留，含接近／返回 | s | 0..200、有限；失去租约即停止 |
| 实际TCP已核对 | `tcp_verified` | rokae_calibration.motion_failures：现场声明放行门（不是自动实测证据） | 布尔值，工位全局声明 | 几何／路径／指标改动使相关声明失效；重新读取 revision 后核对；fixture声明需覆盖各启用型号 |
| 当前工装已核对 | `fixture_verified` | rokae_calibration.motion_failures：现场声明放行门（不是自动实测证据） | 布尔值，工位全局声明 | 几何／路径／指标改动使相关声明失效；重新读取 revision 后核对；fixture声明需覆盖各启用型号 |
| 工作空间已核对 | `workspace_verified` | rokae_calibration.motion_failures：现场声明放行门（不是自动实测证据） | 布尔值，工位全局声明 | 几何／路径／指标改动使相关声明失效；重新读取 revision 后核对；fixture声明需覆盖各启用型号 |
| 接近／检测／撤退路径已核对 | `paths_verified` | rokae_calibration.motion_failures：现场声明放行门（不是自动实测证据） | 布尔值，工位全局声明 | 几何／路径／指标改动使相关声明失效；重新读取 revision 后核对；fixture声明需覆盖各启用型号 |
| 珞石上下电／拖动／停止已验收 | `control_acceptance` | rokae_calibration.motion_failures：现场声明放行门（不是自动实测证据） | 布尔值，工位全局声明 | 几何／路径／指标改动使相关声明失效；重新读取 revision 后核对；fixture声明需覆盖各启用型号 |
| 接触／插入工艺已确认 | `contact_verified` | rokae_calibration.motion_failures：现场声明放行门（不是自动实测证据） | 布尔值，工位全局声明 | 几何／路径／指标改动使相关声明失效；重新读取 revision 后核对；fixture声明需覆盖各启用型号 |
| 姿态方向已核对 | `orientation_verified` | rokae_calibration.motion_failures：现场声明放行门（不是自动实测证据） | 布尔值，工位全局声明 | 几何／路径／指标改动使相关声明失效；重新读取 revision 后核对；fixture声明需覆盖各启用型号 |
| 独立定位最大误差（mm） | `acceptance_metrics.position_error_mm` | rokae_calibration：独立点误差门／重算验收 | mm | 正有限；未填写不放行 |
| 实际到位最大位置误差（mm） | `acceptance_metrics.arrival_error_mm` | rokae_controller：实际起点和每小步到位位置误差 | mm | 正有限；未填写不放行 |
| 实际到位最大姿态误差（°） | `acceptance_metrics.arrival_rotation_deg` | rokae_controller：实际起点和每小步到位姿态误差 | ° | 正有限；未填写不放行 |
| 最低成功率（0～1；仅保存现场约定，尚无自动统计／放行） | `acceptance_metrics.success_rate_min` | 仅保存现场约定；暂无自动统计／验收／连续运行消费 | 比率 | 类型与范围验证；不是自动执行功能 |
| 单次最大耗时（s；仅保存现场约定，尚无自动验收） | `acceptance_metrics.cycle_time_s_max` | 仅保存现场约定；暂无自动统计／验收／连续运行消费 | s | 类型与范围验证；不是自动执行功能 |
| 连续运行次数（正整数；仅保存约定，不会自动连跑） | `acceptance_metrics.continuous_runs_min` | 仅保存现场约定；暂无自动统计／验收／连续运行消费 | 次数 | 类型与范围验证；不是自动执行功能 |

`candidate_probe_lengths_mm`、candidate_source、note：资料／候选或备注，不能代替 tcp_T_probe 实测；未作为有效参数表单功能。设备 ID/base_frame/translation_unit 由高级 JSON 可见，但不允许改为其它设备或旧 UR 坐标。

## 视觉及相机字段：113 个公开 schema 项

入口 `web/app.js` → 本浏览器 localStorage `r435PipelineParamsV2` → 本次请求 params → `web_runtime/task_schema._validated_params` → 本次 web_task.json。不是跨浏览器共享工位参数文件。单节点／全流程命令由 task_manager 构造；常驻签名不符拒绝，需要重新应用相机模式。

下表“4/5”表示该项在四／五的消费范围；其它算法专用项保留原值并在四／五禁用。选择不是数值的字段用枚举验证；数值要求有限且满足 schema 范围，整数不得截断，小数／空值／布尔／巨大溢出拒绝；相关上下界及形态学奇数还做联合校验。相机 SDK 节点实际支持与回读需要真实设备验证，原生计数单位未说明时不猜测为毫米。

| 界面字段／配置键 | 实际消费代码（非仅转发） | 单位／坐标系 | 范围或验证／4/5范围 |
|---|---|---|---|
| 电机型号<br>`product_type` | motor_detection/roi_geometry.py:479；motor_detection/roi_geometry.py:606 | 枚举／标签；stride为像素采样间隔 | 枚举 100／132／160；4/5消费（部分字段按方向／型号／条件分支） |
| YOLO 输入尺寸<br>`yolo_imgsz` | task_manager viewer／YOLO任务 → MotorBoxDetector 参数 | px | 320..1920；整数；4/5不使用；前端禁用 |
| 置信度阈值<br>`yolo_conf` | task_manager viewer／YOLO任务 → MotorBoxDetector 参数 | 无量纲或算法原生量，见消费代码 | 0.01..0.95；有限数值；4/5不使用；前端禁用 |
| NMS IoU 阈值<br>`yolo_iou` | task_manager viewer／YOLO任务 → MotorBoxDetector 参数 | 无量纲或算法原生量，见消费代码 | 0.05..0.95；有限数值；4/5不使用；前端禁用 |
| 推理设备<br>`yolo_device` | task_manager viewer／YOLO任务 → MotorBoxDetector 参数 | 枚举／标签；stride为像素采样间隔 | 枚举 auto／cpu／0；4/5不使用；前端禁用 |
| 位姿来源策略<br>`pose_source_policy` | motor_detection/fusion.py:150；motor_detection/fusion.py:672 | 枚举／标签；stride为像素采样间隔 | 枚举 yolo-preferred／yolo-required；4/5不使用；前端禁用 |
| 深度模式曝光<br>`exposure_us` | task_manager._viewer_base → camera/viewer_components.apply_acquisition_settings：相机节点写入／回读 | μs | 24.0..99840.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 深度模式增益<br>`gain` | task_manager._viewer_base → camera/viewer_components.apply_acquisition_settings：相机节点写入／回读 | 相机 Gain 原生单位，现场回读核对 | 0.0..39.99；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 相机采集帧率<br>`camera_fps` | task_manager._viewer_base → camera/viewer_components.apply_acquisition_settings：相机节点写入／回读 | 帧/s | 1.0..60.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 启用相机端距离截取<br>`camera_height_extract_enable` | task_manager._viewer_base → camera/viewer_components.apply_acquisition_settings：相机节点写入／回读 | 布尔 | 严格布尔；4/5消费（部分字段按方向／型号／条件分支） |
| 相机端最小距离（mm）<br>`camera_height_extract_min_mm` | task_manager._viewer_base → camera/viewer_components.apply_acquisition_settings：相机节点写入／回读 | mm；视觉深度／物体几何（非SDK基座位置） | 0.0..2000.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 相机端最大距离（mm）<br>`camera_height_extract_max_mm` | task_manager._viewer_base → camera/viewer_components.apply_acquisition_settings：相机节点写入／回读 | mm；视觉深度／物体几何（非SDK基座位置） | 10.0..4000.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 最小连通域面积<br>`box_min_area` | motor_detection/roi_geometry.py:150；motor_detection/roi_geometry.py:162 | px² | 100..2000000；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 最小矩形度<br>`box_min_rectangularity` | motor_detection/roi_geometry.py:329 | 无量纲 | 0.05..1.0；有限数值；4/5不使用；前端禁用 |
| 最小实心度<br>`box_min_solidity` | motor_detection/roi_geometry.py:332 | 无量纲 | 0.05..1.0；有限数值；4/5不使用；前端禁用 |
| 闭运算核<br>`box_close_px` | motor_detection/roi_geometry.py:334 | px | 1..51；整数；4/5不使用；前端禁用 |
| 开运算核<br>`box_open_px` | motor_detection/roi_geometry.py:335 | px | 1..51；整数；4/5不使用；前端禁用 |
| YOLO ROI 扩边比例<br>`roi_padding` | motor_detection/fusion.py:177；motor_detection/fusion.py:687 | 无量纲或算法原生量，见消费代码 | 0.0..0.5；有限数值；4/5不使用；前端禁用 |
| 深度选取策略<br>`depth_selection` | motor_detection/fusion.py:183 | 枚举／标签；stride为像素采样间隔 | 枚举 peak-band／all-roi；4/5不使用；前端禁用 |
| 峰值带半宽<br>`depth_band_half_width_mm` | motor_detection/fusion.py:180 | mm；视觉深度／物体几何（非SDK基座位置） | 0.25..50.0；有限数值；4/5不使用；前端禁用 |
| 深度直方图间隔<br>`depth_histogram_bin_mm` | motor_detection/fusion.py:180 | mm；视觉深度／物体几何（非SDK基座位置） | 0.05..10.0；有限数值；4/5不使用；前端禁用 |
| 孔邻域半径比例<br>`height_reference_hole_radius_ratio` | motor_detection/fusion.py:505 | 无量纲 | 0.01..0.3；有限数值；4/5不使用；前端禁用 |
| 基准面最大倾角<br>`height_reference_max_tilt_deg` | motor_detection/fusion.py:531 | °；视觉方向／倾角 | 1.0..80.0；有限数值；4/5不使用；前端禁用 |
| 高度聚类间隔<br>`height_cluster_bin_mm` | motor_detection/fusion.py:548 | mm；视觉深度／物体几何（非SDK基座位置） | 0.1..20.0；有限数值；4/5不使用；前端禁用 |
| 高度层半宽<br>`height_cluster_half_width_mm` | motor_detection/fusion.py:555 | mm；视觉深度／物体几何（非SDK基座位置） | 0.25..50.0；有限数值；4/5不使用；前端禁用 |
| 高度层最少样本<br>`height_min_selected_samples` | motor_detection/fusion.py:599；motor_detection/fusion.py:714 | 像素、点或迭代计数（按字段名） | 20..100000；整数；4/5不使用；前端禁用 |
| 高度层最小跨度<br>`height_min_span_ratio` | motor_detection/fusion.py:716；motor_detection/fusion.py:718 | 无量纲 | 0.01..1.0；有限数值；4/5不使用；前端禁用 |
| 平面 RMSE 上限<br>`max_plane_rmse_mm` | motor_detection/roi_geometry.py:541；motor_detection/roi_geometry.py:542 | mm；视觉深度／物体几何（非SDK基座位置） | 0.01..20.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 平面内点率下限<br>`min_plane_inlier_ratio` | motor_detection/roi_geometry.py:543；motor_detection/roi_geometry.py:544 | 无量纲 | 0.01..1.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 边缘中心离散上限<br>`max_edge_center_spread_mm` | motor_detection/fusion.py:373；motor_detection/fusion.py:375 | mm；视觉深度／物体几何（非SDK基座位置） | 0.01..50.0；有限数值；4/5不使用；前端禁用 |
| PnP 平均误差上限<br>`max_pnp_mean_error_px` | motor_detection/fusion.py:381；motor_detection/fusion.py:390 | px | 0.01..100.0；有限数值；4/5不使用；前端禁用 |
| PnP/深度中心差上限<br>`max_pnp_depth_gap_mm` | motor_detection/fusion.py:382；motor_detection/fusion.py:394 | mm；视觉深度／物体几何（非SDK基座位置） | 0.1..200.0；有限数值；4/5不使用；前端禁用 |
| 法向夹角上限<br>`max_normal_gap_deg` | motor_detection/fusion.py:383；motor_detection/fusion.py:395 | °；视觉方向／倾角 | 0.1..90.0；有限数值；4/5不使用；前端禁用 |
| 应用手眼基座变换<br>`use_base_transform` | task_manager._fusion_args 添加当前标定供视觉内部参考；最终 adapt_result 始终重算珞石基座 | 布尔 | 严格布尔；仅视觉内部参考；最终珞石变换固定当前标定 |
| 基座参考航向<br>`geometry_reference_yaw_deg` | motor_detection/roi_geometry.py:442 | °；视觉方向／倾角 | -180.0..180.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 航向最大误差<br>`geometry_reference_max_error_deg` | motor_detection/roi_geometry.py:443 | °；视觉方向／倾角 | 1.0..89.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 方向规定方式<br>`geometry_direction_mode` | motor_detection/roi_geometry.py:417 | 枚举／标签；stride为像素采样间隔 | 枚举 fixed-branch／base-heading；4/5消费（部分字段按方向／型号／条件分支） |
| 固定方向分支<br>`geometry_fixed_branch_deg` | motor_detection/roi_geometry.py:419 | °；视觉方向／倾角 | 枚举 0／90／180／270；4/5消费（部分字段按方向／型号／条件分支） |
| 单边最小支持长度<br>`geometry_min_edge_support_px` | motor_detection/roi_geometry.py:547；motor_detection/roi_geometry.py:548 | px | 0.0..5000.0；有限数值；4/5不使用；前端禁用 |
| geometry_hole_inset_x_ratio<br>`geometry_hole_inset_x_ratio` | motor_detection/roi_geometry.py:449 | 无量纲 | 0.5..1.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| geometry_hole_inset_y_ratio<br>`geometry_hole_inset_y_ratio` | motor_detection/roi_geometry.py:450 | 无量纲 | 0.5..1.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| geometry_four_point_min_span_mm<br>`geometry_four_point_min_span_mm` | motor_detection/roi_geometry.py:558；motor_detection/roi_geometry.py:562 | mm；视觉深度／物体几何（非SDK基座位置） | 10.0..300.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| geometry_four_point_max_span_mm<br>`geometry_four_point_max_span_mm` | motor_detection/roi_geometry.py:559；motor_detection/roi_geometry.py:563 | mm；视觉深度／物体几何（非SDK基座位置） | 20.0..500.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 外框中心离散上限<br>`geometry_max_edge_center_spread_mm` | motor_detection/roi_geometry.py:538 | mm；视觉深度／物体几何（非SDK基座位置） | 0.1..20.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 深度 ROI 最小矩形度<br>`geometry_min_rectangularity` | motor_detection/roi_geometry.py:329；motor_detection/depth_geometry.py:699 | 无量纲 | 0.05..1.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 深度 ROI 最小实心度<br>`geometry_min_solidity` | motor_detection/roi_geometry.py:332；motor_detection/depth_geometry.py:701 | 无量纲 | 0.05..1.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 外框平面深度带半宽<br>`geometry_plane_band_half_width_mm` | motor_detection/roi_geometry.py:393 | mm；视觉深度／物体几何（非SDK基座位置） | 0.25..20.0；有限数值；4/5不使用；前端禁用 |
| 外框深度直方图间隔<br>`geometry_plane_histogram_bin_mm` | motor_detection/roi_geometry.py:394 | mm；视觉深度／物体几何（非SDK基座位置） | 0.05..5.0；有限数值；4/5不使用；前端禁用 |
| 宽工作区最小深度<br>`geometry_workspace_depth_min_mm` | motor_detection/roi_geometry.py:145；motor_detection/depth_geometry.py:564 | mm；视觉深度／物体几何（非SDK基座位置） | 50.0..2000.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 宽工作区最大深度<br>`geometry_workspace_depth_max_mm` | motor_detection/roi_geometry.py:146；motor_detection/depth_geometry.py:565 | mm；视觉深度／物体几何（非SDK基座位置） | 100.0..4000.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 相对平面距离阈值<br>`geometry_relative_plane_threshold_mm` | motor_detection/roi_geometry.py:403；motor_detection/roi_geometry.py:176 | mm；视觉深度／物体几何（非SDK基座位置） | 0.25..10.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 候选平面最小边长<br>`geometry_candidate_min_size_mm` | motor_detection/roi_geometry.py:200；motor_detection/depth_geometry.py:692 | mm；视觉深度／物体几何（非SDK基座位置） | 10.0..500.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 候选平面最大边长<br>`geometry_candidate_max_size_mm` | motor_detection/roi_geometry.py:201；motor_detection/depth_geometry.py:693 | mm；视觉深度／物体几何（非SDK基座位置） | 20.0..1000.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 候选矩形最大长宽比<br>`geometry_candidate_max_aspect` | motor_detection/roi_geometry.py:152；motor_detection/depth_geometry.py:694 | 无量纲 | 1.0..5.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 候选平面最少点数<br>`geometry_plane_min_points` | motor_detection/depth_geometry.py:570 | 像素、点或迭代计数（按字段名） | 80..100000；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 多平面 RANSAC 次数<br>`geometry_plane_ransac_trials` | motor_detection/depth_geometry.py:590 | 像素、点或迭代计数（按字段名） | 50..2000；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 最多提取平面数<br>`geometry_max_candidate_planes` | motor_detection/depth_geometry.py:591 | 像素、点或迭代计数（按字段名） | 1..20；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 机械臂允许最大平面倾角<br>`geometry_max_plane_tilt_deg` | motor_detection/roi_geometry.py:549；motor_detection/roi_geometry.py:550 | °；视觉方向／倾角 | 1.0..85.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 粗检测像素步长<br>`geometry_coarse_pixel_stride` | motor_detection/depth_geometry.py:574 | 枚举／标签；stride为像素采样间隔 | 枚举 1／2／3／4；仅5；4禁用 |
| RANSAC 批量假设数<br>`geometry_ransac_batch_size` | motor_detection/depth_geometry.py:611 | 枚举／标签；stride为像素采样间隔 | 枚举 16／32／64／128；仅5；4禁用 |
| RANSAC 目标置信度<br>`geometry_ransac_confidence` | motor_detection/depth_geometry.py:612 | 无量纲 | 0.5..0.999999；有限数值；仅5；4禁用 |
| 最终 PCA 残差阈值<br>`geometry_final_refine_threshold_mm` | motor_detection/depth_geometry.py:747 | mm；视觉深度／物体几何（非SDK基座位置） | 0.25..10.0；有限数值；仅5；4禁用 |
| C 形单边覆盖门限<br>`geometry_cshape_min_side_coverage` | motor_detection/roi_geometry.py:281 | 无量纲 | 0.1..1.0；有限数值；仅5；4禁用 |
| C 形矩形度门限<br>`geometry_cshape_min_rectangularity` | motor_detection/roi_geometry.py:286 | 无量纲 | 0.1..1.0；有限数值；仅5；4禁用 |
| C 形实心度门限<br>`geometry_cshape_min_solidity` | motor_detection/roi_geometry.py:288 | 无量纲 | 0.1..1.0；有限数值；仅5；4禁用 |
| 回扫平面距离阈值<br>`geometry_recovery_plane_threshold_mm` | motor_detection/depth_geometry.py:390 | mm；视觉深度／物体几何（非SDK基座位置） | 0.25..10.0；有限数值；仅5；4禁用 |
| 恢复后单边覆盖门限<br>`geometry_recovery_min_side_coverage` | motor_detection/depth_geometry.py:401 | 无量纲 | 0.1..1.0；有限数值；仅5；4禁用 |
| 型号推断最小分差<br>`geometry_recovery_model_margin_mm` | motor_detection/depth_geometry.py:343 | mm；视觉深度／物体几何（非SDK基座位置） | 0.0..30.0；有限数值；仅5；4禁用 |
| 模板搜索外扩像素<br>`geometry_recovery_search_margin_px` | motor_detection/depth_geometry.py:383 | px | 0..100；整数；仅5；4禁用 |
| 任务超时时间<br>`task_timeout_s` | task_manager：子进程截止、常驻触发超时 | s | 5.0..180.0；有限数值；4/5消费（部分字段按方向／型号／条件分支） |
| 最小孔直径（px）<br>`front_cover_min_hole_diameter_px` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | px | 4.0..100.0；有限数值；4/5不使用；前端禁用 |
| 三孔弦长变异上限<br>`front_cover_max_triplet_chord_cv` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | 无量纲 | 0.01..0.5；有限数值；4/5不使用；前端禁用 |
| H01 期望图像角度<br>`front_cover_h01_expected_angle_deg` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | °；视觉方向／倾角 | -180.0..180.0；有限数值；4/5不使用；前端禁用 |
| H01 方向误差上限<br>`front_cover_max_anchor_error_deg` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | °；视觉方向／倾角 | 1.0..180.0；有限数值；4/5不使用；前端禁用 |
| 平面最少内点数<br>`front_cover_min_plane_points` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | 像素、点或迭代计数（按字段名） | 20..200000；整数；4/5不使用；前端禁用 |
| 平面内点率下限<br>`front_cover_min_plane_inlier_ratio` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | 无量纲 | 0.01..1.0；有限数值；4/5不使用；前端禁用 |
| 平面 RMSE 上限（mm）<br>`front_cover_max_plane_rmse_mm` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | mm；视觉深度／物体几何（非SDK基座位置） | 0.01..20.0；有限数值；4/5不使用；前端禁用 |
| 安装耳平面 RMSE 上限（mm）<br>`front_cover_max_mounting_support_plane_rmse_mm` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | mm；视觉深度／物体几何（非SDK基座位置） | 0.1..10.0；有限数值；4/5不使用；前端禁用 |
| 端盖绝对倾角上限<br>`front_cover_max_absolute_tilt_deg` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | °；视觉方向／倾角 | 0.1..80.0；有限数值；4/5不使用；前端禁用 |
| 孔组节圆误差上限（mm）<br>`front_cover_max_pattern_error_mm` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | mm；视觉深度／物体几何（非SDK基座位置） | 0.1..50.0；有限数值；4/5不使用；前端禁用 |
| 锁付接近距离（mm）<br>`front_cover_approach_distance_mm` | task_manager._front_cover_resident_command／overrides → camera/pipeline_setup → front_cover_fastening 配方与 detector | mm；视觉深度／物体几何（非SDK基座位置） | 1.0..200.0；有限数值；仅端盖视觉参考接近点；执行用工位 approach_mm |
| 端盖彩图曝光（SDK值）<br>`front_cover_color_exposure_time` | task_manager → --color-exposure-us → 相机彩色曝光 SDK 写入／回读 | μs | 50.0..500000.0；有限数值；4/5不使用；前端禁用 |
| 网页显示 Gamma（不进入检测）<br>`front_cover_rgb_gamma` | camera/live_viewer 增强显示图；不修改检测原始输入 | 无量纲 | 0.1..3.0；有限数值；4/5不使用；前端禁用 |
| 网页显示 CLAHE（0=关闭）<br>`front_cover_rgb_clahe_clip` | camera/live_viewer 增强显示图；不修改检测原始输入 | 无量纲或算法原生量，见消费代码 | 0.0..20.0；有限数值；4/5不使用；前端禁用 |
| 常驻触发等待上限（秒）<br>`front_cover_trigger_timeout_s` | task_manager：常驻请求截止／单次进程超时／独占模式选择 | s | 10.0..180.0；有限数值；4/5不使用；前端禁用 |
| 端盖执行模式<br>`front_cover_execution_mode` | task_manager：常驻请求截止／单次进程超时／独占模式选择 | 枚举／标签；stride为像素采样间隔 | 枚举 resident／single_shot；4/5不使用；前端禁用 |
| 单次任务保护上限（秒，0=不限）<br>`front_cover_single_shot_timeout_s` | task_manager：常驻请求截止／单次进程超时／独占模式选择 | s | 0.0..600.0；有限数值；4/5不使用；前端禁用 |
| 灰度阈值<br>`camera_vmsl_gray_threshold` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `VMSLFpgaGrayThreshold`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 0..255；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 梯度阈值<br>`camera_vmsl_gradient_threshold` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `VMSLFpgaGradientThreshold`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 0..255；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 最小线宽<br>`camera_vmsl_min_line_width` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `VMSLFpgaMinLineRange`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 0..4095；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 最大线宽<br>`camera_vmsl_max_line_width` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `VMSLFpgaMaxLineRange`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 0..4095；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 照度补偿<br>`camera_vmsl_illumination_compensation` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `VMSLRelillumCompFlag`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 严格布尔；4/5消费（部分字段按方向／型号／条件分支） |
| 杂点滤除强度档位<br>`camera_noise_filter_level` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NoiseFilterClass`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 枚举 关闭／弱／中／强；4/5消费（部分字段按方向／型号／条件分支） |
| 杂点滤除使能<br>`camera_noise_filter_enable` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NoiseFilterEnable`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 严格布尔；4/5消费（部分字段按方向／型号／条件分支） |
| 杂点滤除类型<br>`camera_noise_filter_type` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NoiseFilterType`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 枚举 连通域滤除／斑点滤除；4/5消费（部分字段按方向／型号／条件分支） |
| 斑点连通最大深度差<br>`camera_noise_speckle_depth_diff` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NoiseFilterSpeckleDepthDiff`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 1..65535；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 斑点面积<br>`camera_noise_speckle_area` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NoiseFilterSpeckleArea`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 1..10000000；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 法向量滤除强度档位<br>`camera_normal_filter_level` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NormalFilterClass`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 枚举 不使用／弱／中／强；4/5消费（部分字段按方向／型号／条件分支） |
| 法向量计算窗口宽度<br>`camera_normal_filter_window_width` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NormalFilterWindowWidth`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 3..15；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 形态学窗口宽度<br>`camera_normal_filter_morph_size` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NormalFilterMorphSize`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 0..17；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 夹角计算方式<br>`camera_normal_filter_angle_type` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NormalFilterZangleType`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 枚举 XY方向／X方向／Y方向；4/5消费（部分字段按方向／型号／条件分支） |
| 滤除夹角阈值<br>`camera_normal_filter_angle_threshold` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NormalFilterZangleTher`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 0..90；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 法向量滤除类型<br>`camera_normal_filter_compare_type` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `NormalFilteCompType`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 枚举 大于模式／小于模式；4/5消费（部分字段按方向／型号／条件分支） |
| 延拓强度档位<br>`camera_extension_level` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `ExtenClass`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 枚举 关闭／弱／中／强；4/5消费（部分字段按方向／型号／条件分支） |
| 延拓使能<br>`camera_extension_enable` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `ExtenEnable`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 严格布尔；4/5消费（部分字段按方向／型号／条件分支） |
| 间隙填充强度档位<br>`camera_gap_fill_level` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `GapFillClass`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 枚举 关闭／弱／中／强；4/5消费（部分字段按方向／型号／条件分支） |
| 间隙填充使能<br>`camera_gap_fill_enable` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `GapFillEnable`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 严格布尔；4/5消费（部分字段按方向／型号／条件分支） |
| 间隙填充模式<br>`camera_gap_fill_mode` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `GapFillMode`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 枚举 X方向／Y方向／XY方向；4/5消费（部分字段按方向／型号／条件分支） |
| X方向间隙填充数量<br>`camera_gap_fill_x_count` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `GapFillXNum`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 0..100；整数；4/5消费（部分字段按方向／型号／条件分支） |
| Y方向间隙填充数量<br>`camera_gap_fill_y_count` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `GapFillYNum`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 0..100；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 间隙填充高度阈值<br>`camera_gap_fill_height_threshold` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `GapFillHeightTh`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 1..10000；整数；4/5消费（部分字段按方向／型号／条件分支） |
| 深度图滤波强度档位<br>`camera_range_filter_level` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `RangeFilterClass`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 枚举 关闭／弱／中／强；4/5消费（部分字段按方向／型号／条件分支） |
| 深度图滤波使能<br>`camera_range_filter_enable` | camera 参数JSON → viewer_components.apply_camera_algorithm_params → SDK `RangeFilterEnable`；枚举按 hik_camera_algorithm_params 转换／回读 | 设备原生枚举／开关／计数；物理单位未明确 | 严格布尔；4/5消费（部分字段按方向／型号／条件分支） |

隐藏内部项：front_cover_hole_detector 由端盖方案选择；camera_normal_filter_enable 默认关闭，由滤波等级联动，不作为独立可编辑功能。SDK相机表包含27个节点，其隐藏项也走内部严格验证。

## 标定与现场操作参数

| 字段／入口 | 保存位置／实际用途 | 单位与验证 |
|---|---|---|
| 检测手眼变换、waist_model_rad、设备／相机／来源 | calibration/rokae_current.json；任务开始固定版本，所有视觉分支适配基座；导入脚本不自动接受 | 变换m、腰部模型rad；身份、刚体、来源签名、腰部一致；现场独立点验收 |
| 独立点 point_camera_m／measured_point_base_m、used_for_calibration、measurement_evidence | 标定独立验收及 outputs/rokae_calibration_validation；从点重算误差，不信摘要 | 3个以上不同且非共线、ID唯一的独立点，测量证据需非空文字，各3个有限非布尔数值m；位置门取现场配置；需真实现场测量声明；旧验收失效只供读取哈希，不放行，重新录入时归档旧标定 |
| 新解算 | HMI handeye session 的 results；检测页 proposal，不覆盖活动检测标定 | 板型12×9格、15mm格／11.25mm标记、DICT_5X5_100；内部残差与姿态覆盖只作解算质量证据 |
| 拖动／暂停／采样／恢复确认，当前厂家报警已核对 | HMI motion owner 与本轮 operation/generation、浏览器 lease；不是 site 的自动验收 | 当前所有者、真实反馈、Ready事件、停稳／新帧／腰部一致；停止后不可续活 |
| 腰部回正预览 | HMI evidence 的 waist_hold_reference.json，保留示教原始控制器角，不把零位当参考 | 4腰部轴rad，头部保持；第二轴模型+π/2、回发减π/2；预览30s，1%与5°限制；现场重复精度待验 |
| 初始／模拟关节目标和线性预览 | HMI当前提案，token绑定generation与当时状态；不直接应用URDF世界位姿作SDK基座目标 | 左臂7轴rad；软限位与漂移／到位门、IK/FK、零工具、实基座世界变换检查；只读预览不代表实际运动 |
| 成功率／周期／连续次数约定 | site.acceptance_metrics存储值；尚无自动聚合统计 | 不填不编默认指标，不以离线测试数或内部解算残差冒充生产验收 |

字段表覆盖现有公开 schema 和珞石工位表单；相机原生节点单位、现场指标、真实零工具／TCP等不能从表单默认值推定。旧UR表单在生产页面隐藏；其速度、UR工作空间和旧tool_frames不作为珞石现场配置来源。
