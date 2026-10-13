# 台州机械臂简化操作与部署记录（2026-10-13）

本轮按用户澄清实施：先轻触，再按补偿量继续下压；“清除错误姿态”指清除控制器报警；检测沿用旧动作。软件部署和隔离按钮验证完成后记录实际版本。没有执行真机上电、清警、腰部回正、拖动、接触、下压或电机检测。

## 问题、原因与修复

| 严重度 | 问题／原因 | 修复与证据 |
|---|---|---|
| 高 | 取消确认勾选后，停止／心跳失败若仅靠状态刷新，会有旧计划重新变得可点击的风险 | 显式记录作废计划 ID；隐藏／关闭、迟到启动回执、续租失败和停止后均不能自动恢复旧计划。实际前端 VM 和隔离 Chrome 验证 |
| 高 | 反馈仍为 off 时，本页待上电任务不能被当成已经结束 | 下电先取消所属任务；不接管其他页面的任务。待上电与外部任务的回归测试 |
| 高 | TCP、工装和下压混为同一步，难以对应“先轻触再压入” | 六阶段后台计划；末端 Z 补偿仅加入一次 TCP 变换，下压独立沿实测接线盒局部 Z。非零 TCP 旋转及 2 mm 下压的跨模块数值测试 |
| 高 | 常用字段藏在多层展开项；参数修改可能使旧位置继续可用 | 常用工装、末端补偿、工装转换、下压补偿、速度和空间限位直接显示；CAS 修订保存、保留未知字段、保存历史、立即作废旧计划。几何／路径改变仍撤销相应验收记录 |
| 中 | 缺独立左臂上电与按部件清警通路 | vision → 唯一 HMI owner → 官方 SDK。左臂上电仅保持手动空闲，不开启拖动或下发位置；清警只调用选定左臂／腰部的 `clearServoAlarm`。CSRF、角色身份、启动代次、租约、重复请求、退出和非目标部位检查仍有效 |
| 中 | 原生请求校验未涵盖新的维护组合 | 无 SDK／无网络的请求检查拒绝右臂维护、腰部误走左臂上电、不同位置目标、无启动门、冲突参数；官方 SDK 编译通过 |
| 中 | 正常下电后前端续租未结束，可能把成功提示覆盖为续租失败 | 下电回执成功立即结束本页租约；隔离浏览器验证上下电按钮状态和所属操作 ID |
| 中 | 相机共享预览失败使普通启动直接退出，机器人状态页也不可访问 | 捕获预览初始化 RuntimeError，保留错误状态和恢复入口，不伪造帧、不自动重试。启动失败回归覆盖；现场当前相机仍拒绝访问，未宣称恢复 |
| 低 | 新旧状态区重复 DOM ID | 常用提示与工程故障明细使用不同 ID；完整 HTML 唯一性测试和实际部署页面检查 |

## 实际操作与参数映射

常用操作不需要勾选确认，也不需要输入运动密码。按钮使用仅本机可读的进程会话凭据；重启后旧凭据失效，非本机 Host 被拒绝。按钮点击并不自动确认验收参数：工位数据、实际反馈、质量门、接触验收、故障联锁及原有冷却继续生效。

| 界面 | 配置键／接口 | 消费处、单位与验证 |
|---|---|---|
| 工装 JL100／JL132／JM160 | `selected_fixture`、`tcp_T_objects[fixture]` | `prepare` 校验所选工装与本次视觉 100／132／160 一致；不会仅按页面工装替换视觉型号 |
| 末端 Z 补偿，一个输入＋保存 | `tcp_z_adjustment_mm` | `build_stages`: `F_T_TCP = flange_T_tcp × Tz(mm/1000)`，沿 TCP 局部 Z，±200 mm，加入一次；须已有实测完整 TCP |
| 工装转换角度，一个输入＋保存 | `tcp_T_objects[fixture]` | 前端 `fixtureWithYaw` 用 TCP Z 旋转左乘完整工装矩阵，连同 XY 安装偏移旋转，界面显示 XYZ 偏移；输入度，矩阵平移为米。完整 6D 变换仍可在工程设置编辑 |
| 工装下压补偿，一个输入＋保存 | `fixture_press_mm[fixture]` | 轻触位置后沿实测 `base_T_box` Z 继续下压，0～50 mm；分型号保存。不是旧几何高度 `fixture_z_adjustments_mm`，不重复叠加 |
| 速度 | `linear_speed_mm_s` | 珞石 MoveL mm/s，(0,10]；不把它重复换算成百分比 |
| 空间限位 X/Y/Z 上下界 | `workspace_m` | 页面 mm→配置米，左臂 SDK 基座坐标；整段插值路径检查。SDK 七轴软限位另外实时读取，工程区可查看，页面空间限位不覆盖控制器软限位 |
| 保存当前位置为初始位置 | `/api/robot/initial/save` → `initial_T_base_flange`、`initial_joints_rad` | 只读获取真实法兰矩阵和 7 轴位置，在新鲜、下电、手动、空闲、停稳状态保存；版本冲突拒绝。未保存时不得借用 UR 初始位／当前位／零位 |
| 腰部回正 | `/api/robot/waist/preview` → operator `waist_start` → `/waist/reset/start` | 同一按钮自动预检，已与标定一致时明确无需运动；目标仍来自保存标定姿态并核对腰部基准，不归零、不重定义第二轴偏置 |
| 腰部上下电 | operator `waist_power_on/off` → `/waist/power/on`／取消所属会话 | 沿用原腰部保持通路；关闭只结束本页所属操作，保持基准不重采样 |
| 左臂上下电 | operator `arm_power_on/off` → `/arm/power/on`／取消所属会话 | `NativeBackend.launch_maintenance(power_hold)`，原生 `--power-only`；心跳失效、父进程退出、反馈异常或最长 120 s 到期下电退出 |
| 腰部／左臂清除报警 | operator `waist_clear_alarm`／`arm_clear_alarm` → `/controller/alarm/clear` | 选定组 `--clear-servo-alarm`；不调用急停 `recoverState`，不解除外部安全回路、不自动重试上电。返回清警命令回执，不将历史日志当作实时报警清零证明 |
| 工程参数与验收记录 | 完整配置表／JSON、独立定位录入接口 | 验收布尔值显示现有记录而非勾选功能；没有把缺失数据填成零、没有自动把验收状态设为通过 |

保存写入 IPC 现有 `config/rokae_site.json`，一直保留直到覆盖，包括软件重启；提交时锁住常用输入，防止编辑或重复点击覆盖。修改机器人几何参数会撤销旧计划和对应验收，不必单纯因 TCP 编辑重复视觉计算；相机、标定、腰部及现场控制变化仍作废旧视觉观测。旧结果和新计划分别检查任务 ID、标定摘要、观测代次及 600 s 有效期。

## 方案四／五到六阶段的调用链

方案四 `rokae_depth_fast_recovery`、方案五 `rokae_depth_icp_adaptive` 保持原对应视觉算法，扫码与手动入口、常驻模式、同帧 RGB／深度和型号门沿用已修复的调用链，详情见 [前一轮现场相机与按钮记录](taizhou_camera_buttons_20261013.md)。本轮不再更换视觉算法或质量预期。

`web/app.js` 选择方案／本次任务 → `/api/pipeline/run(full)` → `PipelineTaskManager` 原方案四／五算法 → 珞石标定转换 `base_T_box` 和带 task_id／观测上下文的产物 → `rokae.js` 生成位置 `/api/robot/preview`（`production_center`＋工装） → `RokaeInspectionController.prepare` 核对成功产物、型号、质量门、标定、腰部和配置 → `rokae_planner.build_stages` → 下一步 `/api/robot/move` → `_execute_stage` → HMI `/inspection/linear/preview` 与 `/inspection/linear/send` → 唯一 SDK owner 执行并核对真实到位与下电退出 → `rokae_execution.jsonl`／`rokae_target_preview.json`。

六步为：

1. 到保存的实际初始法兰位置。
2. 在安全过渡路径中旋转工装并到接线盒正上方；XYZ／旋转同时插值。
3. 到名义轻触位置。
4. 按该型号下压补偿量继续沿接线盒局部 Z 运动。
5. 检测／保持：保留旧版位置到达后的保持逻辑与 `dwell_s`，本轮未找到旧代码中的外部电测接口。
6. 撤离并回到保存的初始位置。

“轻触／下压”目前是位置规划，软件到位不代表有力传感接触证据。“检测／保持”不产生虚构的电测通过结果。原每小步下电、12 s 冷却和统一停止机制保留；没有未经现场验证改成连续上电。

## 验证范围与当前现场条件

本地及 IPC 检测端完整检查 276 项、控制端 180 项；原生无 SDK 请求／退出保护检查 76 项。真实 SDK 0.7.1 静态库与官方头文件编译通过，SDK 源码包、库包和 ELF 摘要在部署包 build_receipt。没有把请求校验当作真机执行测试。

`tools/taizhou/review_production.mjs` 对实际部署 HTML／JS 与实时 GET 检查：无确认勾选、唯一 DOM ID、默认收起工程数据、私有会话、各常用保存接口、清警／回正／上电请求、上下电双态及操作 ID、生成／下一步／停止。所有 POST 在独立 Chrome 内截断或以隔离配置响应；不写现场参数，不转发机器人请求。截图和报告不含会话凭据。此前其他视觉按钮的现场验证记录仍在上文链接；本轮没有重复真实电机检测。

当前三部件实时为 off/manual/idle、无 active/busy/capturing/latched。实际定位独立验收仍为 false，现场缺 `flange_T_tcp`、`workspace_m`、初始位置、过渡方向／安全高度和定位／到位验收指标，且几何、路径、接触、控制验收未完成。因此常用维护接口可接受显式按钮请求，但真实检测计划不会因取消勾选自动放行。这些值必须来自现场测量及约定，不能编造。

相机在本轮服务重启后的 HMI RGB 打开返回 `OpenDevice 0x8006000e`。只读进程核对未发现遗留视觉采集进程或第二个机器人 owner；没有发送相机 DeviceReset、没有恢复默认设置。软件恢复入口可用，相机真实恢复、RGB→常驻再交接仍需恢复设备访问后验证。此次状态不等于前一轮已通过的取帧／QR／视觉结果无效，但不能宣称当前相机在线。

## 版本、日志与回滚

最终包也已归档在 IPC 备份的 `final-package/`，SHA256 校验通过。最终部署版本及摘要见工作区 `outputs/deployment/production-ui-final-20261013-r1/manifest.json` 与 `deployment_record.json`；源代码与运行服务的 Git revision／source_sha256 已核对，dirty=false。HMI 两个原有 untracked 启动脚本保留。

- 检测端：`9e0e167`（完整摘要见部署 manifest）；前序独立变更 `792ab81`、`46c421e`、`3acdbf2`、`bc07619`、`1f4c4e9`。
- 控制端：`49bfe03cb57bf31741fe5e9725854f6a12abfc66`。
- 新运动二进制 SHA256：`8b247e46389d2bfc335a19ef1dc2aa2be5d8b4be9d8990bd6f5738a9b3c2d10c`；state-stream 二进制未改。
- 实际备份：`/home/bit-bhr/r435-release/backups/production-ui-20261013-212137/`，含部署前源码、Git bundle、现场配置、旧二进制、入口和原件摘要。
- 检测日志：`.../r435-vision-1080ti-v1/outputs/hik_rgb_fast/service/service.log`；HMI 日志：`/home/bit-bhr/.local/state/jinlong-hmi/real.log`；离线／IPC／浏览器记录归档在最终部署包 evidence。
- 桌面 `open_inspection.sh` 使用 `--enable-robot-motion` 和本机私有进程会话；启动／打开页面本身没有机器人写入。当前运行使用 `--defer-camera-start` 保持故障时页面可访问；普通桌面启动也已能保留网页并报告预览初始化错误。

回滚时先确认真实部件下电空闲并保存后来修改的现场配置，停止已核对 PID／cwd 的服务。干净 tracked 检测仓库切回 `4781197`、控制仓库切回 `b93a31a`，从本次备份还原原生运动二进制并核对 `bb2513edd0ae5502ac2c985af2f301fcad4950c9d906f32adeaa9bb1a8aaac12`；配置或标定如后来更新，不能盲目覆盖。再按原 HMI owner／检测入口启动并核对运行版本，不回到 UR 程序。所有现场原件仅保留于 IPC 备份，未提交 Git。
