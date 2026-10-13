# 台州电机检测软件检查与部署交付记录（2026-10-13）

> 后续状态：本文件记录首次离线审计阶段。随后通过本机已有 SSH 别名 `jinlong-ipc`（192.168.1.67）连接到同一 bit-bhr，已备份、部署并实测相机／前端。最新版本与逐按钮结论见 [相机修复与现场按钮检查](taizhou_camera_buttons_20261013.md)。以下“尚未部署”等描述保留为阶段记录，不代表当前状态。

## 本轮结论和边界

已检查本地迁移工程、珞石控制工程、启动器、配置模板、相关说明和离线测试，并完成下述修复、Git 提交与离线验证。没有执行真机上电、拖动、腰部运动、电机检测、清警或重试上电。

**工控机尚未部署。** `bit-bhr@192.168.10.66:22` 多次连接超时；`bit-bhr` 主机名不能解析。用户确认 IP 未更换。另尝试资料中的工控机相机接口 `192.168.1.68`，也返回 No route to host；未改变任何地址。本机当前有 `192.168.1.20/24` 和 `192.168.0.126/24`，默认路由经 `192.168.0.1`，未擅自改网络。无法读取今天的现场 Git HEAD、工作树、配置、运行 PID、依赖、相机或机器人反馈；无法进行现场当前版本备份、重启及部署生效验证。

10 月 12 日记录中的版本、设备状态、标定哈希和现场故障仅用于查找入口。本轮不以这些历史记录证明今天的工控机状态，更不声称现场验收通过。

本地项目：

- 检测：`/home/wjr/code/r435-rokae-migration`；目标 `/home/bit-bhr/r435-release/r435-vision-1080ti-v1`。
- 控制：`/home/wjr/code/jinlong-motor-arm-control-codex-send-stop`；目标 `/home/bit-bhr/code/jinlong-motor-arm-control-codex-send-stop`。
- 交付记录和部署工具：本仓库 `docs/`、`tools/taizhou/`。

## 问题、原因、修复和证据

以下严重程度表示软件缺陷的潜在后果，并非声称现场已经发生事故。证据来自实际代码、模拟 HTTP、真实浏览器、实际子进程与持久化文件回归；真机效果另列。

| 级别 | 问题与原因 | 已完成修复 | 验证证据 |
|---|---|---|---|
| P0 | 方案四／五按外框和图纸推导的四点被命名为孔位，可能生成接触路径；这些点并非实测孔 | 四孔规划只接收明确 YOLO 实测或 RGB 暗孔证据；`calibrated_outer_frame` 拒绝；检查完整唯一 LT/RT/RB/LB，三孔 ID 也必须唯一 | `test_rokae_hole_evidence.py`：理论孔、重复孔拒绝；中心补偿不移动实测孔 |
| P0 | 常驻进程可能沿用启动时的标定基座结果，端盖等非四／五分支也可能留下旧基座孔位 | 所有方案完成时用本次固定的检测标定从相机矩阵重算基座中心／孔位／参考接近点；结束时核对标定哈希 | `test_rokae_observation.py`、`test_rokae_scheme45.py`：旧基座缓存被当前标定替换，缺失相机位姿不编造目标 |
| P0 | 检测前后腰部相同，不能证明检测中间未漂移；原结果无法绑定完整状态时段 | 只读观测器贯穿视觉任务，约 0.25 s 采样，采样缺失、长间隔、上电或漂移锁存失败；状态曾偏离再回到原位也更新 epoch | `test_rokae_observation.py`：漂移后返回、缺少观察器、传输超时、任务替换均失效 |
| P0 | 旧成功任务、取消任务、参数编辑或腰部变化后仍可能使用历史结果 | 执行结果必须为当前完成的 full 任务，任务 ID、结果 ID、状态、标定哈希、观测 epoch、完成时间均绑定；失败重写未放行 result，原算法结果另存诊断 | 上述观测测试覆盖完成后取消、算法成功但后续失败、型号冲突、任务替换；规划和阶段执行再次检查身份与起始实际位姿 |
| P0 | 页面隐藏、状态轮询中断、启动回执迟到时，浏览器可能继续心跳或采用旧动作 | 隐藏／关闭、状态超过 1.5 s、请求超时即停止租约；停止后确认和计划失效；服务器拒绝续活已过期租约；退出关闭控制器 | `test_rokae_lifecycle.py`、`tests/js/rokae_lifecycle.cjs`：运行真实前端脚本，覆盖隐藏、旧状态、迟到回执；服务 finally 统一退出 |
| P0 | 拖动开始请求持有控制器锁阻塞 Stop；延迟到达的开始请求可能落在 Stop 之后 | 相机和操作请求使用短锁预留；设备请求在锁外； pending 动作排他；HMI 以启动前 native generation 拒绝旧代请求，迟到回执不能收养为当前动作 | `test_rokae_concurrency.py`：阻塞的真实线程请求期间 Stop 可达、迟到成功不收养；`test_button_drag.py`：旧 generation 在 SDK inspect 前拒绝 |
| P1 | 手眼独立验收仅信摘要，修改矩阵、腰部或来源可留下旧 accepted 记录 | 接受记录绑定矩阵、单位、设备、相机、腰部与解算来源签名；重新计算独立点误差；拒绝布尔证据、重复点和共线点；导入／验收写入互斥；独立验收须腰部一致且三部件下电停稳；旧验收失效时保留只读界面／当前哈希用于重新测点，运动仍严格拒绝；重新录入前归档旧标定 | `test_rokae_calibration_binding.py`：变换、腰部、来源、原始点、签名篡改拒绝；新解算不自动应用检测标定 |
| P1 | 两个编辑器覆盖配置、未知扩展字段丢失、修改几何后仍保留已验收标记 | 内容 revision + 文件锁 + 原子保存；拒绝旧 revision；递归合并保留未知字段；记录前后版本；按改动清除相关现场确认；旧计划及 epoch 失效 | `test_rokae_config_revision.py`：两个并发保存仅一个成功，保存历史、未知字段保留及重新确认覆盖 |
| P1 | HMI 一个接口两次读状态可能混合帧；缺少速度反馈被当成 0，漏掉保留轴／头部 | 同一个 live snapshot 同时生成模型和 raw；速度需要完整、有限、非布尔反馈，优先完整 qd_raw，并包含保留轴；无反馈不判停稳 | `test_inspection_snapshot.py`：HTTP 一次采样、缺项／NaN／头部速度；跨模块桥接将传输耗时计入左臂、腰部、右臂各自反馈年龄，拒绝无效年龄并检查七轴类型 |
| P1 | 常驻触发迟到回执或旧 latest.json 可冒充本次结果 | 请求、回执、latest live.trigger_index 必须一致；超时／错号使 owner 失效并结束进程，不复用迟到结果；保存帧索引、采集时间、内参和流信息 | `test_hik_depth_resident_runtime.py`：真实子进程模拟迟到、错号、陈旧 latest |
| P1 | 常驻成功后失败或异步诊断写图时，任务拷贝整个共享目录，会夹入旧帧图或半写图；端盖旧 pose_overlay 也会残留 | 四／五独立触发目录；按 live 触发身份拷贝；异步证据未完成时只发布同步结果／预览，明确 pending；端盖仅拷贝当前 bundle 和当前 latest | `test_rokae_artifact_identity.py`：旧根目录、旧触发和部分写入均不混入；错号目录拒绝；同帧扫码完整调度回归 |
| P1 | 共享预览反复包装同一 HMI 帧成新网页 ID；静止缓存 RGB 可用于扫码 | HMI 返回原始 captured_monotonic；桥接核对真实年龄、RGB 类型、有限内参和相机身份；重复／更旧采集时间不推进网页帧游标；扫码拒绝过期 RGB | `test_shared_camera_identity.py`、`test_motor_qr_http.py`：重复／倒退帧、陈旧扫码拒绝 |
| P2 | 返回路径包含重复当前位置，会触发 HMI no_change；0 mm 空载接近实为到接触点 | 路径剔除静止航点，接近距离必须 >0；保留其它路径和质量门 | `test_rokae_path_runtime.py`：实际返回轨迹没有重复目标；0 mm 空载接近拒绝 |
| P1 | 长计划的12秒冷却已可能超过现有600秒计划有效期，造成后续撤退阶段不能启动 | 计算步间冷却＋直线距离/设定速度＋停留下限；下限已超期的计划预览即阻塞，并显示下限与原因；不延长有效期、不减少冷却 | `test_rokae_plan_preconditions.py`：真实短路径可用，120步内但最低耗时已超期的长路径在任何HMI请求前拒绝；非目标部件上电／缺速度／旧状态拒绝执行 |
| P2 | subprocess readline 等待半行日志可卡住取消；脚本与子进程退出未回收就尝试恢复相机 | 非阻塞按块读管道；新进程组；TERM 有界等待后 KILL、wait 回收；相机脚本 exec Python；日志缓冲有界 | `test_rokae_path_runtime.py`：真实半行输出且忽略 TERM 的子进程超时被回收，避免无限等待 |
| P2 | 空值、布尔数值、小数整数、巨大数值被转换；未消费参数看似生效，旧脚本可能缓存 | 空白保持 null；整数不截断；矩阵／独立点只接收真实数值；非有限和溢出拒绝；四／五不用的旧参数禁用并说明，实际深度形状门仍可编辑；聚合验收指标标注仅保存；脚本版本更新 | 参数持久化、算法参数恶意类型回归；真实 Chrome 检查形状门可编辑、旧深度层参数禁用、七轴／空白 TCP／型号限制 |
| P2 | HMI revision 常量、磁盘 Git HEAD 不能证明运行进程已更新 | 进程启动固定 Git revision、源码摘要、启动时间；检测 GET `/api/version`，HMI state 的 runtime_version；部署后须与包和磁盘三方比较 | `test_runtime_source_version.py`：磁盘产生新提交后，老服务仍报告旧启动版本，不能伪装已重启 |

## 方案四／五完整调用链

路径前缀 `V` 为检测项目，`H` 为控制项目。

1. `V/app/hk_pointcloud_registration/web/app.js` 选择型号、任务和方案，使用 `/api/pipeline/schema` 的支持列表。四：`rokae_depth_multiplane`，100／132；五：`rokae_depth_fast_recovery`，100／132／160。切换型号、方案、算法参数或工具表单使旧目标失效。
2. POST `/api/pipeline/run` → `service/http_api.py` 的 `vision_action` 排他预留 → `web_runtime/task_manager.py::start/_run`。珞石身份经 `robot/rokae_schemes.py::vision_scheme` 映射为各自原算法 `depth_multiplane`／`depth_fast_recovery`，保存结果仍保留所选珞石方案身份。
3. 共享 RGB：HMI `camera.py` → GET `/inspection/frame` 的无损 RGB、内参、畸变、相机身份、采集时间 → `rokae_bridge.py::frame` → `service/shared_camera.py` → FrameStore → 预览／RGB 扫码。检测独占时停止共享读取、由 HMI release 交出相机，结束并回收检测 owner 后 acquire、恢复共享预览。
4. 四的独占命令：`run_hik_live_viewer.sh` → `camera/live_viewer.py`，WorkingMode=4、对齐 RGB-D，`rgbd-geometry-pose`，bootstrap=`depth-plane`。五为 bootstrap=`depth-plane-fast-recovery`、fast mode；五 full 可经 `hik_depth_resident_runtime.py` 常驻软触发，跳过重复打开和配置。四不复用五的常驻 owner，单节点也不能假称兼容。
5. `camera/pipeline_setup.py` → `motor_detection/roi_geometry.py::process_geometry_roi_pose_frame` → `depth_geometry.py` 的多平面发现、物理尺寸／形状候选、平面拟合。五采用降采样＋批量自适应 RANSAC；C 形条件触发 `_recover_c_shape_from_original_depth`，回到原始深度搜索并最终全分辨率精化，恢复也须通过质量门。
6. 图纸／外框四点、方向消歧、局部坐标和质量门 → 相机 `camera_T_box`、平面 RMSE、内点率、候选尺寸、方向／四点跨度、回扫支持证据。由外框推导的四点属于视觉几何参考，**不直接放行实测四孔探测**。
7. QR 常驻“同帧全流程”：同一次软触发 RGB 解码 → `motor_qr_registry.py` 识别型号 → 当帧切换五的视觉配方 → 在同包 RGB-D 上计算。保存 QR identity／解码时序，结果型号与 QR／选择型号交叉核对；成功 QR 不能把失败视觉改成成功。普通预览扫码后再检测不宣称两次采集为同一帧。
8. 本次检测开始固定 `calibration/rokae_current.json` 及观测 epoch；观测器贯穿采集计算。结束用 `adapt_result` 重算当前左臂 SDK 基座位姿，核对标定未变、腰部观测、任务身份和型号。四／五与端盖等分支均执行当前标定适配。任何视觉 ready 均不自动给控制验收、TCP 或接触工艺放行。
9. 任务目录 `outputs/hik_web_pipeline/<task_id>/` 保存 `result.json`、`web_task.json`、产物索引；常驻 bundle 记录触发编号。失败覆盖放行结果，原算法成功数据可保存在 `algorithm_result_unreleased.json` 供诊断；未完成异步证据明确 pending，不夹用旧图。
10. POST `/api/robot/prepare` → `robot/rokae_controller.py::prepare` → `rokae_calibration.motion_failures`／独立点、腰部、设备／上下电预检 → `robot/rokae_planner.py::build_stages` 补偿并规划。中心按“安全过渡→接近→检测→撤退→返回本次起点”；空载中心不包含检测接触阶段。四孔／端盖三孔分别按孔序接近、检测／插入、撤退和安全过渡。最终规划检查法兰、TCP 和探针全路径工作空间，每步 ≤5 mm／3°、全计划 ≤120 步。
11. 浏览器逐阶段确认 → `/api/robot/move` 检查当前阶段、确认、观测 epoch、当前任务、实际起始法兰位姿、配置和标定 → `_execute_stage`。每小步等待冷却，POST HMI `/inspection/linear/preview`，再次 guard，然后 `/inspection/linear/send`。
12. `H/helios_hmi/server.py` → `motion.py::MotionManager` 和原生 solve／prepare／send → `native/xcore_motion.cpp`。目标为 161 左臂 SDK 基座中的法兰矩阵；检查物理 ID、零工具／零参考、控制器基座到世界变换和七轴冗余 elbow，做 IK/FK 预检后走 MoveL。目标部件外的右臂和保留轴进入反馈约束，不由检测项目直接建立第二个 SDK owner。
13. 原生回报与实际 state 必须同一 operation_id、completed、safe_exit_confirmed；检测再算实际法兰位置／姿态到位误差，记录 `rokae_execution.jsonl` 与 `rokae_target_preview.json`；停留同样受 guard。每阶段结束后等待下一次确认，不自动续跑。停止、断线、租约失败或异常调用统一退出；退出未确认保持阻塞。

生产 `hik_fast_rgb_service.py` 仅实例化 `RokaeInspectionController`，只允许物理左臂 161／7 轴。旧 UR 类、旧手眼文件、旧工作空间仍存在于兼容模块／离线测试和原包，但不作为生产运动入口；当前检测标定必须带珞石控制器 ID、相机序列号、`rokae_left_sdk_base` 和米制声明。

## 坐标和补偿

`A_T_B` 表示把 B 坐标转换到 A。矩阵平移为米；表单 XYZ 为毫米、RPY 为度；关节配置为弧度，表单为度。RPY 按 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`，对应 SciPy 外旋 `xyz`，不能当成旧 UR 轴角。腰部第二轴 `q_model[1] = q_controller[1] + π/2`；回发必须减偏置。

```text
base_T_box = base_T_camera @ camera_T_box
tcp_T_object_adjusted = tcp_T_objects[model]（其 TCP 局部 Z 平移加 height_mm / 1000 一次）
base_T_flange(center) = base_T_box @ box_T_center_adjustment
                       @ inverse(tcp_T_object_adjusted) @ inverse(flange_T_tcp)
base_T_flange(probe)  = base_T_hole_or_tip @ inverse(flange_T_tcp @ tcp_T_probe)
```

中心 XYZ 修正位于检测盒局部轴，仅进入中心路径；工装高度是 TCP 局部 Z 修正，仅进入中心工装矩阵。两者不会改写实测孔位。探针长度通过 `tcp_T_probe` 真正参与三孔／四孔目标与工作空间计算。SDK 工具设置须为零，防止外部 TCP 补偿又在控制器重复计算。

整体 URDF 的世界、腰部／头部与 SDK 左臂基座属于不同链；显示 URDF 不能证明真实 TCP、探针方向或法兰到工装几何已验收。现场标定、腰部回正、零工具状态和安装关系仍需实测。

完整逐字段参数映射见 [参数映射](taizhou_rokae_parameters_20261013.md)。

## 现场控制、相机与稳定性边界

- 拖动复用 HMI 腰部当前位置／保存基准保持逻辑，左臂需实体按键。Ready 要求当前原生 operation、控制器身份和原生 ready 事件，不能仅凭已发请求／旧历史日志判断。暂停左臂下电并停稳后才采样；腰部保持会话中需持续监督。恢复检查阶段、冷却、所有者和租约；停止后重新开始确认。
- HMI 为唯一设备控制入口；运动、拖动、采样、腰部回正、自动回放和相机借用有排他规则。状态重连受 SDK 共享锁保护；已有 owner 时不新建竞争连接。右臂、非目标部件和保留轴不能被“目标七轴”过滤而忽略。
- 报警历史明确为 historical_only。当前报警、清警资格和上下电修复不能由历史 51015／31015 推断；本轮未清警、未尝试恢复腰部上电。
- 端盖要求三孔几何、平面、倾角、图纸残差和状态质量门。探针插入要求明确 EMPTY；螺钉已在、未知或检测失败均拒绝。视觉 pose/detection_ready 与接触／插入完成是不同证据。
- 常驻仅兼容当前签名、正确方案与 full 入口；参数变化后必须重新应用相机模式，模型同帧切换走受支持路径。内参与图像来自同一 HMI 帧包；深度原始单位按真实 intrinsics 使用，RGB-D 必须对齐。
- 内存队列已有边界：单个视觉任务，无无限任务队列；网页任务日志尾部有界，NativeJob 最近事件 deque 长度 100。相机和原生任务产物、配置历史与磁盘日志持续累积，未自动删除现场证据；网络未通，无法确认当前磁盘余量、增长率或商定保留周期。该项列为持续运行待测，不能编造资源指标。
- 保留逐小步安全退出、1% 关节速度、≤10 mm/s 线速度、12 s 冷却。N 个连续小步，仅步间冷却理论成本约 `12 × (N−1)` 秒；若 N=120，则约 1428 s，另加运动、上电反馈、退出、HTTP、人工确认和停留。这是代码约束推算，不是实机总周期。计划预览还加入直线运动／停留下限，下限已达600 s的计划拒绝启动。下限未超期仍不能证明实机会在有效期内完成，现场须验证实际上下电和人工确认耗时。没有实施持续上电或缩短冷却。
- 有证据的性能修复限于提前拒绝必然超期计划、去掉重复静止目标、避免半行日志无限阻塞、让常驻兼容路径复用 owner、隔离异步证据、防止重复共享帧。没有依据离线耗时修改视觉门限或重构算法。0.25 s 软件状态采样也不代表曝光硬同步或连续机械测量。

## 验证证据与缺失条件

| 检查 | 本轮结果 | 实际覆盖／限制 |
|---|---|---|
| 检测完整隔离回归 | 246 项通过 | `scripts/run_offline_tests.py --isolated`；复制源码到临时目录、使用明确未验收的合成标定；不改现场／本地活动标定；覆盖持久化冲突、失效结果、观察区间、协议、实际子进程和前端故障 |
| HMI Python 与浏览器租约回归 | 163 项通过，无 Node 跳过 | 运行 `helios_hmi/tests`；排除本机 HTTP(S) 代理，仅模拟设备。最初 502 为代理拦截 Host 伪造测试，正确环境下 403 预期不变 |
| 原生 C++ 请求／清理／协议离线检查 | 70 项通过 | `tests/test_native_motion.py`，request-only 模式；不执行 SDK 上电或运动 |
| 真实 Chrome 界面检查 | 通过 | 本地隔离服务，HMI 指向 loopback 1 且 defer camera；原始脚本和参数表单实际加载。四禁 160、五支持、七轴、空白 TCP、模型偏置、形状门／旧参数状态、按钮禁止执行、无 JS 异常；previewWidth=0，明确无真实相机验收 |
| 脚本语法、Git diff | 通过 | Node 检查修改的 JS/mjs，Python AST 检查部署工具，提交前 diff --check |
| 实际工控机、完整运行包依赖／manifest | 未检查 | SSH 超时；本地无活动 rokae_current.json、现场 SDK／模型完整资产。默认现场依赖模式未通过不能用合成文件伪装；不得把隔离回归当成现场 verify.sh 已通过 |

测试日志在本地部署包的 `evidence/`，其 SHA256 在包 `SHA256SUMS`；不进入 Git。原始本地日志：`/tmp/taizhou-isolated-suite-20261013.log`、`/tmp/taizhou-hmi-20261013.log`、`/tmp/taizhou-native-20261013.log`、`/tmp/taizhou-ui-review-20261013.log`、连接日志。浏览器截图是离线 UI 证据，不是设备图像。

## 版本、部署包及回滚

最终软件提交和包摘要以 `outputs/deployment/taizhou-20261013/manifest.json`、`SHA256SUMS` 为准；本文件下方记录最终 HEAD。包只有已提交源码、Git bundle 和软件检查日志，不带现场标定／参数、凭据、模型、SDK、图像、数据集、缓存或运行产物。控制和检测必须一起更新：generation 回执和 captured_monotonic 属于本轮新增协议，旧 HMI 不满足时会明确拒绝。

执行与回滚细则见 [部署工具与步骤](../tools/taizhou/README.md)。恢复网络后先运行只读 probe，按当天实际状态核对 HEAD、工作树、组件、PID、运行版本和现场文件哈希。只有空闲、退出已确认且三部件真实下电／手动／停稳时才进行软件服务切换；保留现场配置和未知未跟踪文件。备份当前源码、Git 状态、活动配置／标定、原生二进制、启动器和运行摘要后，以 fast-forward 更新两个项目，重启原软件入口，再对包／磁盘／新进程的版本和摘要进行三方核对。没有任何自动清警／重试上电步骤。

本轮未创建或核实远程回滚目录。准备位置为 `/home/bit-bhr/r435-release/backups/audit-20261013-<实际时间>/`，必须以真实备份成功后记录的路径和哈希为准。10 月 12 日完整备份 `/home/bit-bhr/r435-release/backups/r435-before-rokae-20261012.tar.gz` 及历史腰部备份仅为历史资料，今天未重新核验；原包恢复脚本可能恢复旧 UR 工程，不得自动启动。新部署回滚以本次真实部署前的两个 Git HEAD 为准，保留现场配置，不将旧 UR 运动入口作为自动恢复动作。

## 软件已验证、现场待验证与暂不支持

软件已验证：上表故障回归、四／五调度和保存身份、当前标定重算、工具补偿数学、路径／质量／参数门、软件排他及停止租约、进程回收、源码版本固定。所有这些均有软件验证范围。

现场待验证：今天的运行版本与配置、依赖及模型／SDK哈希、真实 RGB／RGB-D 交接与恢复；TCP／零工具及探针安装、中心／高度补偿符号；腰部 162 上电故障与拖动 Ready；松键／隐藏页面／断线／进程退出的真实安全停止；腰部回正重复精度；独立点实际精度；真实电机视觉成功率、接触与插入工艺、周期和连续运行／磁盘资源。

现场精度、到位位置／姿态阈值、成功率、周期和连续次数以现场活动配置及约定为准；模板这些验收指标为 null，本轮无法取得真实现场约定，**缺失而非默认通过**。内部手眼解算残差不替代独立定位精度，模型输出视觉成功不替代插入成功。

暂不支持：基于四／五图纸外框参考点直接作真实四孔接触；方案四检测 160；未完备标定／工具／工作空间／接触工艺验收时真机执行；自动选择新标定覆盖活动检测标定；自动清警、上电重试、掉线后自动续跑；自动聚合成功率／周期／连续次数验收；按阶段持续上电优化；由检测入口控制右臂。`candidate_probe_lengths_mm` 是候选资料，实际探针变换必须独立填入／核对。

## 最终本地提交与交付包校验

最终本地工作树已提交；包状态为 `prepared_only`。以下为待部署软件身份，不代表工控机已运行这些版本。

| 项目 | Git HEAD | 已提交源码 SHA256 |
|---|---|---|
| vision | `a8dc45c56e466dd8e737e23b0a635c850075e4c2` | `7793012c3c8ae1cc6c7f047cc91ea547b7ef97ff0268bc7c6a950f8e75a71732` |
| hmi | `7443cb6372b2ca34a1dfa109003b89ab0878e588` | `8fbc1b32f4c781b202d808fa1125d7bc6af8e9779a3522dff1c353e0e61c9855` |

包路径：`/home/wjr/code/code_vpwem/outputs/deployment/taizhou-20261013`。`manifest.json` SHA256：`cd74efb7d41544e8a92b5e03d598a51f7d6870cf7f93e9e67e6a7692c3b0de53`。11 个软件／日志文件的 `SHA256SUMS` 全部通过；两个 Git bundle 校验通过；tar 逐文件摘要与 manifest 完全一致，源码摘要与实际 runtime source_version 算法一致。

部署工具提交：`51a131625bd9b7e2a9726c368a924219179599c6`；只读探针字段修正提交：`2c91127e533d19f0d31881fe204815921a133fb9`。本文和参数映射另作同一文档变更提交，其 hash 以本仓库 Git 和交付回复为准。

### vision 本轮清晰变更集

```text
511f7d4 Reject stale site edits and revoke acceptance after parameter changes
d9033de Expire browser control leases on hidden pages and reject delayed starts
8ee2548 Require measured hole evidence before generating contact probe paths
68c5f80 Correlate resident camera results and invalidate timed-out trigger owners
c2d566b Bind executable vision results to observed waist state and current task identity
243597f Reserve shared device actions and keep Stop available during start requests
b63f4a6 Skip stationary Cartesian waypoints and reap timed-out camera processes
512958b Bind independent acceptance to exact calibration and recompute recorded point errors
99c5444 Isolate trigger evidence and reject ambiguous or inactive operator parameters
284146e Run isolated behavior regressions and update read-only browser review
9ac00c4 Report pinned process source identity for deployment verification
4df0d54 Preserve shared camera capture identity and refuse repeated cached frames
05ee0e5 Keep consumed depth shape gates editable and clarify reference pose fields
4945f01 Reject nonnumeric transforms and measurements before configuration persistence
6b9ea57 Reject plans that exceed existing lifetime and enforce all-part power preconditions
9456972 Keep read-only revalidation available while rejecting invalid old acceptance
5d6aca5 Reject degenerate independent validation points and nontext evidence
a8dc45c Include transport age in every component safety snapshot
```

### hmi 本轮清晰变更集

```text
288a5f2 Publish one inspection state snapshot and reject incomplete velocity feedback
5d1de42 Reject delayed drag starts from an earlier stop generation
f79425d Expose source revision and digest pinned at HMI startup
7443cb6 Publish original shared camera capture timestamp for frame identity
```
