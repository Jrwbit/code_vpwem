# 相机切换修复与实际前端按钮检查（2026-10-13）

首次未发现本机已有 SSH 别名 `jinlong-ipc`；本次已通过该别名的 192.168.1.67 网口连接到同一 bit-bhr。192.168.10.66 地址未改动。首次复核曾出现相机拒绝访问；现场相机重启后，又修复了超时强制退出及扫码等待预算。最终 RGB、常驻软触发同帧 RGB／深度、超时释放和 RGB 恢复均已实测，当前状态见末尾“重启后的最终复测”。这仍不是机器人运动或电机检测验收。

## 实际修复

| 问题 | 原因与修复 | 证据 |
|---|---|---|
| RGB 无法切至常驻深度 | HeightExtract 关闭时上下限节点不可读写，原代码先写上下限。改为先启用、按当前范围决定上下限顺序，并逐项回读；单位仍按 ZUnit 转换 | 现场启用前读写返回 0x80060007，启用后可读；真实常驻初始化成功。test_height_extract_window.py 覆盖启用、上移／下移窗口、忽略写入及超界 |
| 切换失败未恢复预览／原因不清 | stop 与命令生成移入恢复事务；确认深度 owner 退出后才恢复 RGB；同时报告原失败和恢复失败；优先取 Python 异常，避免 XOpenDisplay 尾行掩盖原因 | test_camera_mode_handoff.py 覆盖拒绝交接、缺启动器、退出未确认及恢复失败 |
| 常驻参数按钮走错接口 | 接线盒常驻模式仍写无人消费的 RGB 参数队列。改为重启当前常驻进程，使用本次参数并核对 SDK 回读 | 现场曝光 100→101 us，requested=actual=101，applied=true |
| 常驻预览返回旧 RGB | FrameStore 有错误仍返回旧缓存；常驻没有首帧时接口回落到交接前缓存。改为拒绝错误缓存／跨模式旧帧，页面明确等待首次软触发；常驻拒绝 RGB 下载保存 | 现场 frame.jpg 503；只读按钮禁用；旧缓存 HTTP 回归 |
| 未编辑参数保存重算矩阵 | RPY 往返产生浮点变动并补入缺省键，可能误判几何改动。保留未编辑字段的精确源值、关节精度和缺省键 | 实际两种保存 HTTP 200；test_rokae_form_roundtrip.py；语义一致后恢复原现场字节，保留保存历史 |
| 端盖任务启动找不到配方 | 命令显式指向旧 app/front_cover_fastening/config，而发布包在 config。改用 runtime_paths.FRONT_COVER_RECIPES，并在安装检查校验配置 | 实际端盖常驻初始化、三套方案和六类节点按钮；既有命令测试现在真实读取配方 |
| 共享 RGB 超期，重开失败 | 3200×1944 图压缩约 2.78 s，超过既有 2 s 门限。改为无损无压缩 NPZ，保留所有原始像素、内参、畸变与采集时间，门限不变 | 三次实际请求从 2.755–2.788 s 降至 0.053–0.062 s；全尺寸高熵 HTTP 回归在原门限内无损返回 |
| 状态与未应用参数反馈不完整 | 共享 stream 缺宽高与字节数导致 undefined／NaN；CameraAlgorithmParams 仅在日志。补齐元数据，将 SDK 未应用回执送入 camera_settings；普通 JSON 日志不破坏 reader | 实际状态、参数回执与 resident/shared 单元回归 |

## 验证范围

检测 259 项和 HMI 164 项在本机及工控机通过，工控机 Node 检查未跳过。verify.sh installed 校验真实运行包、SDK／模型资产及依赖通过。原生源码和二进制未改动；此前 70 项 request-only 检查不是本轮真机运动证据。

实际 Chrome CDP 访问工控机服务，共记录 154 场景和 48 个初始按钮，零浏览器异常。154 包含不同状态、动态方案、节点和页签，不能解释成 154 个生产操作全部验收通过。没有执行机器人上电、拖动、腰部运动、真实电机检测、清警或上电重试。

相机／配置／界面请求到达真实后端。真实检测 POST 在独立审核浏览器中截断，记录请求节点和方案，验证前端调用与失败提示；它们不证明算法现场成功。机器人动作按钮在当前只读入口下验证禁用和无请求。停止只检查空闲回执，无 executor 的 safe_exit_confirmed=false，不能证明运动中的实际安全退出。

当前场景没有可识别二维码，扫码验证了失败回执，未声称已注册码成功。没有真实检测结果，因此目标导出和任务产物缩略图的现场功能待验。共享 RGB 模式尚无在线写参接口；常驻模式的曝光、增益、帧率可以应用，设备不接受的可选算法节点会明确报告未应用。

## 逐按钮与动态场景

| 标识 | 界面／证据 | 实测范围 |
|---|---|---|
| `junction_box` | 01电机接线盒检测           接线盒识别、中心姿态、方向与基座坐标 | live task selection |
| `taskSwitchButton` | 选择任务 | 真实界面状态 |
| `refreshButton` | 刷新 | live status |
| `rokaeRefresh` | 刷新连接与状态 | live feedback |
| `startCameraButton` | 打开相机 | live RGB owner |
| `streamButton` | 暂停预览 | 真实界面状态 |
| `fitButton` | 切换画面适应方式 | 真实界面状态 |
| `fullscreenButton` | 全屏 | live browser |
| `captureButton` | 采集一帧等待并下载最新画面 | 实际采图／保存 |
| `saveButton` | 保存到工控机 | 实际采图／保存 |
| `copyLogsButton` | 复制 | 实际浏览器复制 |
| `copyResultJsonButton` | 复制 JSON | 实际浏览器复制 |
| `resetParamsButton` | 恢复默认 | 真实界面状态 |
| `openQrTestButton` | 二维码识别测试 | 真实界面状态 |
| `qrTestScanButton` | 识别当前画面中的二维码 | live camera QR response |
| `closeQrTestButton` | × | 真实界面状态 |
| `rokaeParameterReload` | 读取全部机械臂参数 | 实际配置读取 |
| `rokaeLoadConfig` | 读取参数 | 实际配置读取 |
| `rokaeParameterSave` | 保存表单参数与历史版本 | 实际同值保存、回读及历史 |
| `rokaeSaveConfig` | 保存工位参数并记录版本 | 实际同值保存、回读及历史 |
| `rokaeValidationTemplate` | 生成当前标定录入模板 | 真实界面状态 |
| `rokaeValidate` | 计算并保存独立点验收 | live backend guard |
| `rokaeWaistPreview` | 预检腰部回到当前标定姿态 | live read-only waist preflight |
| `rokaePreview` | 预览最新成功检测的左臂基座目标 | live missing-result guard |
| `rokaeDragStart` | 腰部上电保持＋开启左臂按键拖动 | 现场联锁已检查；实际动作待验 |
| `rokaeDragPause` | 暂停左臂并下电（腰部保持） | 现场联锁已检查；实际动作待验 |
| `rokaeDragResume` | 继续左臂拖动／进入 Ready | 现场联锁已检查；实际动作待验 |
| `rokaeWaistStart` | 确认回到当前标定腰部姿态 | 现场联锁已检查；实际动作待验 |
| `rokaeCapture` | 采集当前手眼样本 | 现场联锁已检查；实际动作待验 |
| `rokaeSolve` | 解算当前手眼样本组 | 现场联锁已检查；实际动作待验 |
| `rokaeExecute` | 执行下一阶段：等待有效计划 | 现场联锁已检查；实际动作待验 |
| `cancelTaskButton` | 取消任务 | 现场联锁已检查；实际动作待验 |
| `rokaeStop` | 停止机器人与检测 | 实际空闲停止回执；动态停止待验 |
| `rokaeOperatorStop` | 结束现场控制并停止／下电 | 实际空闲停止回执；动态停止待验 |
| `cameraModeSelect:working_mode_4_resident` | {"mode": "working_mode_4_resident", "ready": true, "pid": 2798753, "error": null} | 实际模式初始化 |
| `resident initial preview` | {"status": 503} | live stale-frame guard |
| `applyParametersButton` | 应用到相机 | live resident restart and SDK readback |
| `scanDetectButton` | 扫码并自动检测二维码 → 型号 → 第五套流程 → Base 位姿 | 前端调用已验证；真实检测未执行 |
| `cameraModeSelect:rgb_preview` | {"mode": "rgb_preview", "ready": false, "pid": null, "error": null} | 实际模式初始化 |
| `model:JM160` | Switched to supported 160 before testing its scheme | live UI model selection |
| `scheme:junction:0` | {"label": "方案四 · 珞石左臂 · 纯深度多平面 ROI珞石·方案4JL100/JL132 使用；JM160 不再允许纯外框中心输出；输出当前手眼标定的左臂SDK基座位姿，经珞石TCP／工装补偿后分步执行。（仅支持 100/132）", "disabled": true, "active": false} | live model restriction |
| `scheme:junction:1` | {"label": "方案五 · 珞石左臂 · 高速深度＋C形回扫珞石·方案5JL100/JL132/JM160 统一使用 WorkingMode=4 常驻软触发、扫码型号切换和第五套全流程；JM160 保留偏位风险警告；输出当前手眼标定的左臂SDK基座位姿，经珞石TCP／工装补偿后分步执行。", "disabled": false, "active": true} | live UI scheme selection |
| `node:junction:1:capture_rgb` | {"path": "/api/pipeline/run", "node": "capture_rgb", "scheme": "rokae_depth_fast_recovery", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:1:depth` | {"path": "/api/pipeline/run", "node": "depth", "scheme": "rokae_depth_fast_recovery", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:1:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "rokae_depth_fast_recovery", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `scheme:junction:2` | {"label": "YOLO + 高度层融合推荐100/132 用 YOLO；160 用固定外框四点标定，再做高度层与 PnP 交叉验证", "disabled": false, "active": true} | live UI scheme selection |
| `node:junction:2:capture_rgb` | {"path": "/api/pipeline/run", "node": "capture_rgb", "scheme": "yolo_height", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:2:yolo` | {"path": "/api/pipeline/run", "node": "yolo", "scheme": "yolo_height", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:2:depth` | {"path": "/api/pipeline/run", "node": "depth", "scheme": "yolo_height", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:2:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "yolo_height", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `scheme:junction:3` | {"label": "YOLO + RGB-D 平面对照100/132 用 YOLO；160 用固定外框四点标定，再结合自适应 RGB-D 平面", "disabled": false, "active": true} | live UI scheme selection |
| `node:junction:3:capture_rgb` | {"path": "/api/pipeline/run", "node": "capture_rgb", "scheme": "yolo_rgbd", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:3:yolo` | {"path": "/api/pipeline/run", "node": "yolo", "scheme": "yolo_rgbd", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:3:depth` | {"path": "/api/pipeline/run", "node": "depth", "scheme": "yolo_rgbd", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:3:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "yolo_rgbd", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `scheme:junction:4` | {"label": "RGB 四边形 + 相对平面100/132·无 YOLOJL100/JL132 使用；JM160 外框定位已停用，需改用专用内部结构 RGB-D 定位", "disabled": false, "active": true} | live UI scheme selection |
| `node:junction:4:capture_rgb` | {"path": "/api/pipeline/run", "node": "capture_rgb", "scheme": "geometry_only", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:4:depth` | {"path": "/api/pipeline/run", "node": "depth", "scheme": "geometry_only", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:4:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "geometry_only", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `scheme:junction:5` | {"label": "纯深度多平面 ROI100/132·纯深度JL100/JL132 使用；JM160 不再允许纯外框中心输出", "disabled": false, "active": true} | live UI scheme selection |
| `node:junction:5:capture_rgb` | {"path": "/api/pipeline/run", "node": "capture_rgb", "scheme": "depth_multiplane", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:5:depth` | {"path": "/api/pipeline/run", "node": "depth", "scheme": "depth_multiplane", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:5:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "depth_multiplane", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `scheme:junction:6` | {"label": "高速纯深度 + C 形回扫方案五·100/132/160JL100/JL132/JM160 统一使用 WorkingMode=4 常驻软触发、扫码型号切换和第五套全流程；JM160 保留偏位风险警告", "disabled": false, "active": true} | live UI scheme selection |
| `node:junction:6:capture_rgb` | {"path": "/api/pipeline/run", "node": "capture_rgb", "scheme": "depth_fast_recovery", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:6:depth` | {"path": "/api/pipeline/run", "node": "depth", "scheme": "depth_fast_recovery", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:6:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "depth_fast_recovery", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `scheme:junction:7` | {"label": "JM160 内部结构 RGB-D160·实验定位160 专用实验方案：内部金属平行结构定位 XY/方向，深度平面求 Z/法向；结果保留旧外框位姿对照，验收前禁止运动", "disabled": false, "active": true} | live UI scheme selection |
| `node:junction:7:capture_rgb` | {"path": "/api/pipeline/run", "node": "capture_rgb", "scheme": "jm160_internal_rgbd", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:7:depth` | {"path": "/api/pipeline/run", "node": "depth", "scheme": "jm160_internal_rgbd", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:7:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "jm160_internal_rgbd", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `scheme:junction:8` | {"label": "珞石 RGB 四孔 PnPJL100/JL132 · 只读联调共享真实海康原图与内参，识别四孔及接地螺丝，使用本次左臂标定输出位姿；不打开深度SDK", "disabled": false, "active": true} | live UI scheme selection |
| `node:junction:8:capture_rgb` | {"path": "/api/pipeline/run", "node": "capture_rgb", "scheme": "rokae_rgb_pnp", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:8:yolo` | {"path": "/api/pipeline/run", "node": "yolo", "scheme": "rokae_rgb_pnp", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:junction:8:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "rokae_rgb_pnp", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `tab:junction:基础` | Parameter group rendered | 真实界面状态 |
| `tab:junction:YOLO` | Parameter group rendered | 真实界面状态 |
| `tab:junction:相机` | Parameter group rendered | 真实界面状态 |
| `tab:junction:相机深度范围` | Parameter group rendered | 真实界面状态 |
| `tab:junction:深度` | Parameter group rendered | 真实界面状态 |
| `tab:junction:融合` | Parameter group rendered | 真实界面状态 |
| `tab:junction:高度层` | Parameter group rendered | 真实界面状态 |
| `tab:junction:质量门限` | Parameter group rendered | 真实界面状态 |
| `tab:junction:位姿` | Parameter group rendered | 真实界面状态 |
| `tab:junction:方向判定` | Parameter group rendered | 真实界面状态 |
| `tab:junction:纯几何` | Parameter group rendered | 真实界面状态 |
| `tab:junction:四点标定` | Parameter group rendered | 真实界面状态 |
| `tab:junction:相对平面` | Parameter group rendered | 真实界面状态 |
| `tab:junction:高速回扫` | Parameter group rendered | 真实界面状态 |
| `tab:junction:C形恢复` | Parameter group rendered | 真实界面状态 |
| `tab:junction:端盖孔检测` | Parameter group rendered | 真实界面状态 |
| `tab:junction:端盖平面` | Parameter group rendered | 真实界面状态 |
| `tab:junction:端盖质量门` | Parameter group rendered | 真实界面状态 |
| `tab:junction:端盖位姿` | Parameter group rendered | 真实界面状态 |
| `tab:junction:端盖相机` | Parameter group rendered | 真实界面状态 |
| `tab:junction:网页显示增强` | Parameter group rendered | 真实界面状态 |
| `tab:junction:振镜参数` | Parameter group rendered | 真实界面状态 |
| `tab:junction:深度后处理` | Parameter group rendered | 真实界面状态 |
| `front_cover` | 02前端盖检测与锁付定位           JL100/JL132 端盖倾斜、三孔识别与逐孔 6D 位姿 | live task and camera owner selection |
| `scheme:front:0` | Front-cover scheme selected | 真实界面状态 |
| `node:front:0:capture_rgbd` | {"path": "/api/pipeline/run", "node": "capture_rgbd", "scheme": "front_cover_fastening", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:0:hole_detection` | {"path": "/api/pipeline/run", "node": "hole_detection", "scheme": "front_cover_fastening", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:0:plane_fit` | {"path": "/api/pipeline/run", "node": "plane_fit", "scheme": "front_cover_fastening", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:0:pose_output` | {"path": "/api/pipeline/run", "node": "pose_output", "scheme": "front_cover_fastening", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:0:quality_gate` | {"path": "/api/pipeline/run", "node": "quality_gate", "scheme": "front_cover_fastening", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:0:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "front_cover_fastening", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `scheme:front:1` | Front-cover scheme selected | 真实界面状态 |
| `node:front:1:capture_rgbd` | {"path": "/api/pipeline/run", "node": "capture_rgbd", "scheme": "front_cover_depth_roi_dark", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:1:hole_detection` | {"path": "/api/pipeline/run", "node": "hole_detection", "scheme": "front_cover_depth_roi_dark", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:1:plane_fit` | {"path": "/api/pipeline/run", "node": "plane_fit", "scheme": "front_cover_depth_roi_dark", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:1:pose_output` | {"path": "/api/pipeline/run", "node": "pose_output", "scheme": "front_cover_depth_roi_dark", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:1:quality_gate` | {"path": "/api/pipeline/run", "node": "quality_gate", "scheme": "front_cover_depth_roi_dark", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:1:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "front_cover_depth_roi_dark", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `scheme:front:2` | Front-cover scheme selected | 真实界面状态 |
| `node:front:2:capture_rgbd` | {"path": "/api/pipeline/run", "node": "capture_rgbd", "scheme": "front_cover_model_ring_ransac", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:2:hole_detection` | {"path": "/api/pipeline/run", "node": "hole_detection", "scheme": "front_cover_model_ring_ransac", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:2:plane_fit` | {"path": "/api/pipeline/run", "node": "plane_fit", "scheme": "front_cover_model_ring_ransac", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:2:pose_output` | {"path": "/api/pipeline/run", "node": "pose_output", "scheme": "front_cover_model_ring_ransac", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:2:quality_gate` | {"path": "/api/pipeline/run", "node": "quality_gate", "scheme": "front_cover_model_ring_ransac", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:2:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "front_cover_model_ring_ransac", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `scheme:front:3` | Front-cover scheme selected | 真实界面状态 |
| `node:front:3:capture_rgbd` | {"path": "/api/pipeline/run", "node": "capture_rgbd", "scheme": "front_cover_plane_spatial_circle", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:3:hole_detection` | {"path": "/api/pipeline/run", "node": "hole_detection", "scheme": "front_cover_plane_spatial_circle", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:3:plane_fit` | {"path": "/api/pipeline/run", "node": "plane_fit", "scheme": "front_cover_plane_spatial_circle", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:3:pose_output` | {"path": "/api/pipeline/run", "node": "pose_output", "scheme": "front_cover_plane_spatial_circle", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:3:quality_gate` | {"path": "/api/pipeline/run", "node": "quality_gate", "scheme": "front_cover_plane_spatial_circle", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `node:front:3:full` | {"path": "/api/pipeline/run", "node": "full", "scheme": "front_cover_plane_spatial_circle", "blocked": true, "status": 409} | 前端调用已验证；真实检测未执行 |
| `tab:front:基础` | Parameter group rendered | 真实界面状态 |
| `tab:front:相机` | Parameter group rendered | 真实界面状态 |
| `tab:front:相机深度范围` | Parameter group rendered | 真实界面状态 |
| `tab:front:位姿` | Parameter group rendered | 真实界面状态 |
| `tab:front:端盖孔检测` | Parameter group rendered | 真实界面状态 |
| `tab:front:端盖平面` | Parameter group rendered | 真实界面状态 |
| `tab:front:端盖质量门` | Parameter group rendered | 真实界面状态 |
| `tab:front:端盖位姿` | Parameter group rendered | 真实界面状态 |
| `tab:front:端盖相机` | Parameter group rendered | 真实界面状态 |
| `tab:front:网页显示增强` | Parameter group rendered | 真实界面状态 |
| `tab:front:振镜参数` | Parameter group rendered | 真实界面状态 |
| `tab:front:深度后处理` | Parameter group rendered | 真实界面状态 |
| `cameraModeSelect:rgb_preview` | {"mode": "rgb_preview", "ready": false, "pid": null, "error": null} | 实际模式初始化 |
| `shutdownButton` | 关闭相机 | live camera release |
| `camera reopen after shutdown` | Fresh RGB preview restored | live owner recovery |
| `saveFixtureZButton` | 覆盖保存当前工装 Z 补偿 | 旧兼容按钮隐藏，不作为生产功能 |
| `saveFixtureXYButton` | 覆盖保存 JM160 中心 XY 偏移 | 旧兼容按钮隐藏，不作为生产功能 |
| `prepareRobotButton` | 根据最新全流程结果生成末端 TCP 目标 | 旧兼容按钮隐藏，不作为生产功能 |
| `prepareProbeButton` | 生成 120 mm 探针验证计划 | 旧兼容按钮隐藏，不作为生产功能 |
| `anonymous-43` | 步骤 1快速到接线盒上方 | 旧兼容按钮隐藏，不作为生产功能 |
| `anonymous-44` | 步骤 2插入接线柱并保持 | 旧兼容按钮隐藏，不作为生产功能 |
| `anonymous-45` | 步骤 3退回上方并回初始位 | 旧兼容按钮隐藏，不作为生产功能 |
| `robotPrimaryAction` | 执行当前阶段 | 旧兼容按钮隐藏，不作为生产功能 |
| `cancelRobotPlan` | 取消计划 | 旧兼容按钮隐藏，不作为生产功能 |
| `anonymous-30` | 清除控制器报警（使用 RobotAssist） | 明确禁用／暂不支持 |
| `rokaeExport` | Export link stays hidden without a current plan | no valid motion plan; export pending |
| `artifact thumbnails` | No real motor task was started | no real detection result; artifact review pending |

## 部署与回滚

真实备份目录：`/home/bit-bhr/r435-release/backups/camera-buttons-20261013-143855/`。before.json、两个 Git bundle／source tar、现场配置与标定、原生执行文件、启动器和桌面入口已保存；初始 SHA256SUMS 校验通过。远程原有未跟踪 open_real.sh/open_simulation.sh 保留。仅 fast-forward 更新已提交源码，不覆盖 SDK、模型、数据或现场配置。

- vision：`ef0ceae4bc158ba8e939de3acf3033a4bc35291a`；源码 SHA256 `1bd5435e88723253bc9dfb7ae2251b74dfd8809dd9ea653fc47dacf4f78f97c9`。
- hmi：`d7598c1b10725fbe6a05f407411a9df1a7c2181b`；源码 SHA256 `be1dc0d0f3a7bbaac7bcfd3a526a6e08691ea572d42bf51398588f629a22f431`。

本地交付包：`outputs/deployment/taizhou-camera-buttons-20261013-handoff/`；包清单的 prepared_only 只表示打包，不冒充运行验证。部署结果、实时 PID／版本、实际按钮日志、检查日志与原件保留证据另存部署记录和 evidence。

运动入口保持只读，原 HMI 写入开关及 left_arm/waist 限制保留。回滚应先确认下电空闲、停止当前软件及 owner，再切换两个备份记录中的部署前 HEAD（vision e8bfa046aa52565e424125371f43991947ef1c85，HMI 3163665b95df52c45a4e30a1859f761dc9c760ac）；保留现场配置并核对原件／二进制哈希。不能自动启动旧 UR 包。详见 [部署步骤](../tools/taizhou/README.md)。

TCP、腰部上电、拖动 Ready、回正重复精度、独立点实际精度、真实电机成功率、接触／插入、总周期和连续运行仍未验收。12 秒冷却、600 秒计划有效期、质量门和设备联锁未放宽。

## 第一次复核：相机尚未恢复（历史状态）

严重程度最高的当前阻塞是相机设备访问失败。完成上述 154 场景后，额外进行不调用检测算法、不连接机器人执行器的 SDK 相机采样：选中 00DA8664418、local IP 192.168.1.68，设置 WorkingMode=4 与软件触发。OpenDevice、Start、SoftTrigger 返回成功，但 FetchFrame 在约 5 秒后返回 0x80060006（无数据），图像数为 0。它只证明初始化成功，不证明深度采集可用。

停止取流、关闭 SDK 后，恢复 RGB 仍无新帧；重启 HMI 后 OpenDevice 返回 0x8006000e（无设备访问权限）。相机 192.168.1.12 经 enp2s0 可以 ping 通，只读 GVCP 寄存器请求返回 access denied。已释放软件相机 owner、停止 HMI／SDK 进程并等待超过 XML 中 600000 ms 的最大心跳时间，再启动正常 HMI，故障仍存在。现有证据无法确定是设备残留独占状态还是其他设备所有者；不将猜测当作根因。未执行强制抢占、寄存器解锁或 DeviceReset；XML 明确说明 DeviceReset 会恢复初始／默认设置。

2026-10-13 10:07:52 UTC（北京时间 18:07:52）最后只读复核：vision PID 2831744，使用 --defer-camera-start；HMI PID 2870606，保留原 --enable-write 和 left_arm/waist 限制。两服务运行版本、磁盘 Git HEAD、源码摘要与交付包一致，camera.ok=false、error=OpenDevice 0x8006000e。左臂、右臂、腰部反馈新鲜且均 off/manual/idle，motion busy=false、latched=false、capturing=false。未发出运动命令。

已请求现场仅对相机断电重启，机器人保持下电；完成后仍需重测 RGB、常驻初始化、同帧 RGB／深度软触发及异常恢复。尚未收到完成回执，因此没有将现场恢复或深度软触发列为通过。机器人按钮只验证当前禁用状态及离线故障路径，不能替代实际上电／运动验收。

首轮部署时，原现场配置、标定、原生二进制及启动器共 11 个备份原件已逐一核对相同。曝光实测修改后已恢复最初读到的 2000 us，增益 0、帧率 30；之后设备访问失败，当时不能声称设备读回再次通过。参数保存历史留存，现场参数文件已恢复原始字节。实时故障、历史成功记录、测试日志、原件哈希及部署记录位于交付包 evidence 和 deployment_record.json；运行日志仍在 outputs/hik_rgb_fast/service/service.log 与 ~/.local/state/jinlong-hmi/real.log。

## 重启后的最终复测（北京时间 19:15:47）

现场第一次相机重启后，独立枚举再次找到 00DA8664418 on 192.168.1.68；重启 HMI 相机 owner 后 RGB 和曝光 2000 us 恢复。网页扫码复测暴露另一条真实故障路径：常驻扫码原等待 3 s，而完整取帧＋无标签扫码实际耗时约 4.4 s。超时随后只给 resident 0.2 s 退出，可能在 SDK FetchFrame 期间 terminate，跳过 Stop／CloseDevice；HTTP 未捕获 TimeoutError，浏览器显示 Failed to fetch。相机随后再次拒绝访问，不能将第一次重启视为闭环修复。

本次追加修改与证据：

| 问题 | 修复 | 验证 |
|---|---|---|
| QR 无标签时缺少深度取帧证据 | 将既有 frame_diagnostics 移至 QR 提前返回之前；仅记录尺寸、帧号、深度统计，不改变算法或质量门 | 6ea8dcd；真实软触发 RGB／深度帧号一致 |
| 超时立即强制结束 SDK owner | 先使结果失效，保留正常 12 s 退出等待；SIGTERM 转为异常退出，使 finally 的 Stop／CloseDevice 有机会执行；保留最终强制退出后备 | cefec28；真实子进程回归验证在途取帧结束后关闭、迟到结果不能复用，以及 SIGTERM 进入 finally |
| HTTP 超时丢失回执 | 捕获 TimeoutError，返回 JSON HTTP 504 | cefec28；实际 HTTP 回归及工控机 3 s 故障注入均收到明确回执 |
| 网页 3 s 扫码等待短于实测耗时 | 仅常驻模式独立扫码改为 10 s，RGB 仍为 3 s；更新脚本缓存版本 | b2b240e；真实网页请求 timeout_ms=10000，4.664 s 返回“未识别到二维码”，常驻 owner 继续就绪 |

第二次现场相机重启后，使用 10 s 诊断等待进行不启动检测算法的真实相机采样：RGB 和深度均 3200×1944、同一 frame_num=1，深度 501318 个非零像素；无二维码回执用时 4.438 s，正常切回 RGB 成功。再人为使用原 3 s 等待，HTTP 504 在包含关闭等待的 6.029 s 后返回；迟到结果被作废，随后打开 RGB HTTP 200、新帧有效、无 access denied。这是实际故障恢复证据，未延长图像新鲜度或放宽质量门。

最终真实 Chrome 界面重新选择接线盒任务并执行 RGB→常驻→RGB，拒绝首帧前旧 RGB（503）。独立扫码取得同帧 RGB／深度，深度 500004 个非零像素；界面明确显示无二维码，常驻仍 ready=true。通过现有参数按钮恢复曝光 2000 us，requested=actual=2000、applied=true，然后恢复 RGB。没有执行真实电机检测、机器人上电、拖动、腰部运动、清警或重试上电。

追加后的本机隔离软件回归和工控机回归均 262 项通过。本机正常目录缺现场 rokae_current.json，直接运行检查该文件存在的用例不通过；使用已有隔离运行器的明确未验收合成标定后全量通过，未复制合成标定到现场。工控机使用实际现场标定，未跳过 Node 检查。最初 154 个按钮／状态场景和 HMI 164 项是此前版本证据；最终追加的真实界面检查仅针对相机路径，不声称重新验收全部按钮。

最终 vision HEAD／运行版本 b2b240eb7f878c126132132e184a7c847bef54a4；HMI 在本轮期间另有变更推进并运行到 b93a31a1a05f198bdf39101d5c617022bb02c451。该 HMI 变更不由本轮编写或回滚，原 164 项检查仅对应 d7598c1；本轮只核对它的运行版本／磁盘摘要以及相机恢复、机器人只读反馈，不宣称其新腰部功能已由本轮验收。

2026-10-13 11:15:47 UTC（北京时间 19:15:47）复核 camera.ok=true、曝光 2000 us、新帧 age_s=0.06；磁盘、运行版本与最终交付包源码摘要一致。三部分反馈均 off/manual/idle，未执行运动。当前采用正常 RGB 启动入口；常驻模式按需软触发，首触发前显示等待状态而不返回旧帧。实际检测精度、孔位质量、TCP、接触插入和连续运行仍需现场验收。

最新包与证据：outputs/deployment/taizhou-camera-power-cycle-final-20261013/。其 deployment_record.json 区分本轮 vision 部署与另行观察到的 HMI 版本，保留历史失败及后续成功记录。远程归档位于原备份目录下 power-cycle-final/；原始备份与回滚位置保留。

## 有二维码后的全流程异常与修复（北京时间 19:33:43）

用户提供现场二维码后，自行启动的任务 20261013_192754_2555c3（19:27:54，方案五）实际识别出 JL132-001，qr_same_frame=true。同次触发 RGB／深度均为 frame_num=1，采集耗时 3448.902 ms、二维码解码 43.106 ms。任务 result.success=false、robot_ready=false，错误为 name 'write_geometry_roi_pose_summary_artifacts' is not defined。state=completed 仅表示处理结束，此任务没有获得成功位姿或运动许可。

根因是模块拆分后，summary 和 background 两个 writer 仅在 camera/pipeline_setup.py 的 prepare_pipelines 局部作用域导入，而 camera/live_viewer.py 的 graded 分支直接使用这两个名字。实际融合处理完成后，第一处摘要写入产生 NameError；恢复分支也存在同样的缺失引用。7420299 将该实际分支收拢为 write_graded_geometry_artifacts，并在运行时延迟导入两个 writer，避免 fusion→comparison→viewer 兼容门面的循环导入；移除初始化模块内无人消费的两条导入。

新增回归直接调用 viewer 使用的 graded 写入入口，核对摘要文件、内参、触发身份、失败结果和 robot_ready=false 不被改写，并覆盖 recovery_attempted=true 后后台 writer 的绑定和调度。本机隔离与工控机实际配置下均 263 项通过，包括独立冷启动的 SIGTERM 清理回归；未放宽门限。

另使用该失败任务已保存的 01_rgb_original.png、03_depth_raw.npy、camera_intrinsics.json 及原 resident 参数进行工控机离线重放，未打开 SDK 设备或启动现场检测：再次识别 JL132-001，使用原算法与当前检测标定计算，摘要写入成功，无 NameError，耗时约 0.705 s。离线结果 success=true、视觉几何 quality.robot_ready=true；这是已有图像的算法结果，当前独立标定 accepted=false，不能据此宣称珞石运动、接触或实际检测已验收。离线文件单独位于 outputs/offline_qr_replay/20261013_writer_fix，没有替换生产任务或标定。

最终部署 vision 742029966f051a8bfc2253cf8826732412350a55，源码摘要 f834533af83f740913e7e24a9dc3ea2a5d4fc7b725f6d91ca48e5dfb0c54374a；磁盘、Git、运行版本与交付包一致。19:33:43 相机新帧 age_s=0.09、camera.ok=true、曝光为用户此次检测使用的 100 us；三部分机器人均 off/manual/idle。HMI 保留另行部署的 b93a31a，未修改或重启。本轮仅读取用户实际失败记录、离线修复／重放并部署，未自动重新触发现场全流程。

交付包 outputs/deployment/taizhou-qr-full-writer-fix-20261013/ 保存原失败 JSON、263 项检查日志、已存帧重放结果及运行版本复核；远程原备份下 qr-full-writer-fix/ 存放同一归档，before-qr-writer-fix-service.log 保留重启前服务日志。原备份与回滚入口不变。最新源码修复后现场全流程再次执行的结果仍需用户现场操作确认。

## 扫码／手动入口、亮度与二维码候选排除

用户继续操作后的实际记录中，方案五 full 可以完成；19:26:45 capture_rgb、19:26:52 depth 被当前常驻入口拒绝。19:36:07、19:38:32 等失败请求的 scheme 是 rokae_rgb_pnp，并非方案五，却仍有第五套 resident owner，故报“当前常驻相机与所选检测方案不匹配”。扫码会强制选择第五套，但原入口未同步可见方案按钮；扫码结果也未更新型号表单，容易使显示选择与后续手动请求不一致。

0b3dfc5 同步扫码后的可见方案、节点和实际二维码型号。常驻方案五的 capture_rgb/depth 单步尚未实现，明确禁用并说明仅支持 full；不将它们伪装成有效功能。其他接线盒方案的执行入口先释放 resident 并恢复 RGB，再提交该方案的请求。真实独立浏览器检查确认 full 使用 ready resident、其他 RGB PnP 请求前相机切为 rgb_preview。检测 POST 全部在审核浏览器中拦截；型号回填使用明确标记的已注册二维码结果重放验证，未生成新的真实电机检测任务。

亮度有独立实证：此前 resident SDK 的深度 ExposureTime 为 50 us，但 ColorExposureTime 为 500000 us（设备最大值），Gray2DExpGainMode=0。设备 XML 明确 ColorExposureTime 单位为 us，即 0.5 s；这两个节点独立，调深度曝光不能解决彩图曝光过长。dfb9565 新增 camera_color_exposure_time，沿现有彩图参数默认值采用 10000 us，启动前后设置并回读 ColorExposureTime；快捷曝光输入明确改为彩图曝光，深度曝光仍独立在参数表中保留。现场相机回读 requested=actual=10000、applied=true，深度曝光保持用户值 50。共享 RGB 模式仍没有在线写参接口，不声称可在该模式直接修改。

二维码候选误识别的机制：深度平面与矩形筛选不理解标签语义，现有尺寸范围内的二维码牌也可能进入候选。0165140 在每个当前帧检测 QR 区域，排除一半以上面积由 QR 覆盖的候选，且在候选排名／提前结束之前排除，允许继续选择真实接线盒；不抹除原始 RGB、深度或孔位。4781197 在快速解码器未找到标签时继续用 OpenCV 定位，包含未注册的 QR 标签。不存在可读／可定位 QR 区域时，这项防护不能保证排除所有标签或其他矩形；ROI 限制、型号内部特征及真实孔位验收仍有必要。小标签贴在较大接线盒上不会删除整盒深度。

回归覆盖较大二维码候选与真实较小接线盒同场、只有二维码时必须拒绝、接线盒内小标签保留、原深度数组不变、未注册标签和快速解码器失败后的定位回退。已保存的失败任务原图／原深度离线重放保持正确接线盒候选；记录 QR 区域数和被拒绝候选数，未改变现有几何质量门或标定。最终本机隔离与工控机均 266 项检查通过。

实际网页在 10000 us 彩图曝光下独立扫码成功读取 JL132-001（OpenCV 回退，约 1148.7 ms），仅扫码及相机采样，不执行电机全流程。独立审核浏览器核对了方案五分步禁用、常驻／RGB 交接、扫码后的可见方案五与型号 132；完整电机视觉 POST 被拦截，机器人操作 POST 不转发。更新网页脚本缓存版本，现场已打开页面仍需刷新一次以加载新脚本。

本轮源码变更集：0165140（候选排除）、0b3dfc5（入口同步及只读彩图状态）、dfb9565（彩图曝光实际消费）、4781197（定位回退）。完整包与最新运行复核为 outputs/deployment/taizhou-qr-entry-exclusion-final-20261013/，原备份下 qr-entry-exclusion-final/ 同步保存。HMI、现场标定与工位参数不由本次修改；相机彩色曝光有意从实测 500000 改为 10000 us，后续可在常驻模式表单调整。真实运动、接触和连续检测不在此次验证范围内。
