# 金龙／台州工控机代码交接文档（2026-10-20）

本文回答“更新了什么、代码在哪里、接手后如何核对”。交接范围为珞石双臂／腰部 HMI、海康相机和接线盒三维检测、完整路线规划及撤离返回。代码依据统一源码仓库 `70127417b9c2bee2cd3d87ba5e779c104851d46c`；现场只读核对时间为 **2026-10-20 19:09（北京时间）**。本次只编写文档、读取现有状态和核对源码摘要，没有部署、重启、采图或机器人动作。

飞书交付位置：<https://feishu.cn/docx/I2GAdun3nofskUxAOjWcxVwjnJO>，在原全流程报告末尾追加本交接文档，保留历史正文。

## 1. 先找到代码与运行入口

| 用途 | 位置 | 说明 |
| --- | --- | --- |
| 统一源码仓库（开发机） | `/home/wjr/code/jinlong-motor-arm-control` | Git 远端 `https://github.com/thedaosheng/jinlong-motor-arm-control.git`；控制在 `arm_control/`，检测在 `vision/` |
| 工控机控制代码根目录 H | `/home/bit-bhr/code/jinlong-motor-arm-control-codex-send-stop` | 对应统一仓库 `arm_control/`，目录名保留历史名称 |
| 工控机检测代码根目录 V | `/home/bit-bhr/r435-release/r435-vision-1080ti-v1` | 对应统一仓库 `vision/` |
| 本交接文档及联调报告 | `/home/wjr/code/code_vpwem/docs/` | 本仓库是报告／辅助工具仓库，不是生产服务的工作目录 |
| 部署和只读核查说明 | 本仓库 `tools/taizhou/README.md` | 包含备份、更新、单一 owner、启动锁和回滚规则 |
| 辅助审计和记录工具 | 本仓库 `tools/jinlong_hmi/` | 轨迹反馈记录、独立运动学核对、三轮重复试验等 |

后文文件路径以统一源码仓库为准；去掉前缀 `arm_control/` 后在 H 下查找，去掉 `vision/` 后在 V 下查找。旧 `r435-rokae-migration`、`jinlong-motor-arm-control-main` 等目录为历史开发来源，不能凭名称认定正在运行。

现场通过已有 SSH 别名 `jinlong-ipc` 连接；此前有线／无线地址变化，以 SSH 配置、主机密钥及实际主机身份为准。HMI 本机地址 `http://127.0.0.1:50168/`，检测工作台 `http://127.0.0.1:8765/`。

调用关系：检测工作台 → 视觉识别／路线规划 → HMI 计划与运动管理 → 原生 xCore 执行器 → 控制器。相机由 HMI RGB 预览和检测深度流程协调交接，不能同时独占。机器人状态和控制共用既有 owner；另建 SDK 客户端可能重置运动队列。

## 2. 更新内容与具体文件

| 模块 | 已更新的行为 | 统一仓库内的主要文件 |
| --- | --- | --- |
| 控制、退出与故障恢复 | 单一 owner、页面租约、停止及停稳下电确认；保留原生失败原因，退出阶段日志与连续200 ms稳定确认；恢复控制、退限位等受控入口 | `arm_control/helios_hmi/motion.py`、`plan_runner.py`、`native/xcore_motion.cpp`；检测侧 `vision/app/hk_pointcloud_registration/robot/rokae_operator.py` |
| 手眼采样和回放 | 腰部保持与左臂拖动协调、停稳采样、已保存姿态回放；头部回保存姿态；关节显示和控制器角／模型角映射修正 | `arm_control/helios_hmi/handeye.py`、`replay.py`、`teaching_preview.py`、`state.py`、`roles.py`、`native/xcore_state_stream.cpp` |
| 相机交接 | 停稳空闲时交接所有权；共享 RGB 启动重新确认 acquire，修复服务重启后本地交接标记丢失 | `arm_control/helios_hmi/camera.py`、`server.py`；`vision/app/hk_pointcloud_registration/service/shared_camera.py` |
| 完整检测路线 | 按本次三维识别生成中转、悬停、接近、检测暂停及撤离返回；校验工具、工位、限位及碰撞；分离标定腰姿恢复和检测工作腰姿恢复 | `vision/app/hk_pointcloud_registration/robot/corridor_worker.py`、`rokae_trajectory.py`、`rokae_initial.py`、`rokae_site.py` |
| 检测终点复位 | 优先复用绑定原完整路线的撤离关节段，修复重复预览／重试丢失来源、页面会话失效与最终心跳覆盖成功结果；重启后从真实到位记录恢复，拒绝猜选历史路线 | `vision/app/hk_pointcloud_registration/robot/rokae_retreat.py`、`rokae_trajectory.py`、`corridor_worker.py`、`web/rokae_trajectory.mjs` |
| 有效 DH 模型 | 从左右控制器读取各自有效 DH，以身份绑定后用于正解、末端逆解和碰撞几何；装载／执行核对模型摘要 | `arm_control/helios_hmi/controller_kinematics.py`、`kinematic_reference.py`、`native/xcore_state_stream.cpp`；检测侧 `rokae_trajectory.py`、`corridor_worker.py` |
| 操作页面 | 以完整路线为主入口，修复状态失联、保存速度、任务续接及脚本缓存；优化屏外预览和日志刷新；按用户要求隐藏工程设置、参数编辑和坐标变换区域 | `vision/app/hk_pointcloud_registration/web/` 下 `rokae_trajectory.mjs`、`rokae_initial.mjs`、`rokae_parameters.mjs`、`rokae_production.mjs` 等 |
| 最新规划提速 | 兼容归档作为 warm start；当前输入重新做终点求解、限位和碰撞检查；压缩满足条件的共线关节点；优先精确目标，并修复悬停端点跨路线衔接 | `vision/app/hk_pointcloud_registration/robot/route_reuse.py`、`corridor_worker.py`、`rokae_trajectory.py` |
| 最新执行提速 | 新左臂关节路线使用 `powered_segment`，每段由一个原生 owner 连续执行，段内不再每点上下电；仍保留逐点完成／停稳确认、零混合半径及检测暂停 | `arm_control/helios_hmi/motion.py`、`plan_runner.py`、`native/xcore_motion.cpp` |
| 可选速度预设 | `trajectory_speed_mm_s=30.0` 的单字段补丁；新生成接近／插入段限制为不高于10 mm/s，或所选更低速度 | `vision/config/rokae_efficiency.patch.json`；不能用补丁整体覆盖现场工位 JSON；旧计划需重新生成 |

有效 DH 更新是**软件采用控制器已有补偿参数**，不是重新辨识或改写控制器 DH。没有靠固定 XYZ 平移修正定位。SDK 笛卡尔 getter 更接近指令关节的正解，不能代替反馈关节到位误差或实物精度验收。

最新提速修改的源码已有提交，但不能把源码存在、状态文字或文件摘要一致视为真机提速验收。现场 `power_workflow.optimization_pending` 仍有“逐小步退出／12秒冷却”的旧提示，与新实现不一致；排查时结合计划 `execution_mode`、原生事件及实际回执判断。

## 3. 可追溯的关键源码提交

以下均为统一源码仓库的提交；与工控机两个独立仓库的提交号不同。

| 提交 | 内容 |
| --- | --- |
| `4f2371c` | 相机共享所有权与停稳上电腰部交接修复 |
| `d0e9ece` | 完整 corridor 路线成为检测主流程 |
| `1528651`、`e4d6088` | 左臂 J7 软件包络；完整路线读取已保存速度 |
| `4628ed8` | 隐藏工程设置及坐标变换区域 |
| `337a7f7`、`6dcb359`、`b8e82ea` | 终点撤离复位、重试与页面会话、最终心跳竞争修复 |
| `e0bb688`、`7943738`、`2709189` | 读取有效 DH、独立官方正解核对、模型进入求解和碰撞几何 |
| `cd3a714` | 跨服务重启保存并恢复原检测返回来源 |
| `d5fd089` | 规划复用／关节点压缩及分段持续上电执行 |
| `888f773` | 有界速度预设 |
| `638c79c`、`7012741` | 精确检测目标优先及悬停端点衔接修复 |

仓库根 `README.md` 和 `SOURCE_SNAPSHOT.json` 的2026-10-16版本表描述最初导入快照，不能用作当前部署清单。当前源码 HEAD 为 `70127417b9c2bee2cd3d87ba5e779c104851d46c`，本次核对开发机该源码仓库工作树干净。

## 4. 现场实际版本与本次核查结果

**现场源码存在未提交修改，版本必须同时看 HEAD、diff、逐文件摘要和运行接口。** 本次保留原状，不代替其他联调任务提交或覆盖这些修改。

| 检查对象 | 2026-10-20 19:09 观测 |
| --- | --- |
| HMI 磁盘 HEAD | `a7dc92f627cea7cb2a2666293d48e0e8dc65e59b` |
| 检测磁盘 HEAD | `fae02659c38757a4f394f5da7c4617dc6b8955ed` |
| 运行版本 | 两服务 revision 与各自磁盘 HEAD 一致，但均 `dirty=true`；HMI标签 `route-efficiency-888f773ef098` |
| HMI 源码差异 | `helios_hmi/motion.py`、`plan_runner.py`、`native/xcore_motion.cpp` 被修改；另有原有未跟踪备份目录／二进制备份 |
| 检测源码差异 | `app/hk_pointcloud_registration/robot/corridor_worker.py`、`rokae_trajectory.py` 被修改；`route_reuse.py` 未跟踪 |
| 六个差异核心文件 | 对上述6个核心源码逐一计算 SHA256，全部与统一仓库 `7012741` 工作树字节一致；这不等于全仓库逐文件一致或运行进程已加载每个最新文件 |
| 真实进程 | HMI PID `2043028`，检测 PID `2043152`；已用 `/proc/<pid>/cwd` 核对为 H、V。PID只供本次留档 |
| Python | HMI 为 H 下 `.venv/bin/python`；检测为 `/home/bit-bhr/miniconda3/envs/motor-inspection-1080ti/bin/python` |
| 检测启动参数 | `--host 127.0.0.1 --port 8765 --enable-robot-motion --defer-camera-start` |
| 机器人反馈 | 左臂、右臂、身体反馈新鲜，均 off/manual/idle；`busy=false`、`capturing=false`、`latched=false` |
| 相机 | HMI相机 `ok=false`，`OpenDevice 0x8006000e`。本次未采图，未核查检测深度所有权，不能直接判断硬件损坏或深度检测不可用 |
| 检测状态 | pipeline idle，无活动检测任务；旧安全摘要仍列独立精度／参数验收缺失和腰姿偏离标定姿态；`power_workflow.preconditions_met=true` 只代表其前置检查通过 |
| PID 文件 | 旧只读 probe 的两个固定 PID 路径未找到有效进程；不能据此认定服务停止，应查现有进程、监听和工作目录 |

当前检测完整路线有专门的工作腰姿变换和路线验证；旧状态摘要中的标定腰姿／工程验收提示，不能单独代表当前完整路线的可执行结论。需查看本次计划 receipt 和 HMI 预检。本次没有生成新计划或进行执行预检。

现场二进制 SHA256：

```text
H/helios_hmi/native/xcore_motion
049610ad187cae53fda8ee6ce225fabec6f84e72659fc63c8280158c78166412
H/helios_hmi/native/xcore_state_stream
8156c2fededbed58f83702c9e829e35e8e6e1796d11a62a2c2408adbe4f1cb81
```

本次仅记录文件哈希，没有重新编译或取得最新原生构建回执，不能据此宣称二进制由当前源码构建。检测运行 `source_sha256` 和辅助 probe 的磁盘摘要算法不同，不跨算法直接比较；核心源码一致性以上述逐文件比较为准。

## 5. 配置、数据、备份和回滚位置

| 现场资产 | 位置及用途 |
| --- | --- |
| 活动手眼标定 | `V/calibration/rokae_current.json`；本次 SHA256 `4b43a286955b3db2d6cd0ff3c1733c3f06ea514ceb9c56fe56716e0c3701f364` |
| 活动工位配置 | `V/config/rokae_site.json`；本次 SHA256 `47b6a17393fa70bf64bff24efc65021749910db76b5eeb8ca4764e15c397e249`；保存后查历史并回读 |
| 手眼样本及保存姿态 | `/home/bit-bhr/helios_hmi_sessions/handeye_ipc/`；保存参考 `waist_hold_reference.json` 以现有运动证据目录配置为准，不用当前姿态重建 |
| HMI计划 | `/home/bit-bhr/helios_hmi_sessions/plans/` |
| 路线输入、计划、回执 | `V/outputs/rokae_trajectory/<任务ID>/input.json`、`plan.json`、`receipt.json` |
| 跨重启返回记录 | `V/outputs/rokae_trajectory/return_checkpoint.json`；绑定真实检测暂停完成进度及归档摘要，不能手工伪造或随意删除 |
| 检测任务产物 | `V/outputs/hik_web_pipeline/` 等运行目录，按具体任务回执定位 |
| HMI原启动器日志 | `/home/bit-bhr/.local/state/jinlong-hmi/real.log`；直接启动的进程需另查实际标准输出去向 |
| 深度／规划环境 | 原相机 Conda 环境；完整规划环境 `/home/bit-bhr/.local/share/jinlong-planner/venv`；SDK、模型、mesh和第三方前端资源继续现场维护 |
| 有效DH更新备份 | `/home/bit-bhr/r435-release/backups/effective-dh-20261020-114156/` |
| 返回记录修复备份 | `/home/bit-bhr/r435-release/backups/return-checkpoint-20261020-120913/` |
| 更早源码／二进制备份 | `/home/bit-bhr/.local/state/jinlong-hmi/deployments/20261017-e226bf2/`；不是最新版本的一键回滚包 |

上述备份不证明涵盖之后的提速修改；后续更新前应另做当日完整备份。回滚须使用对应更新前 HEAD、原生二进制及配置摘要成套核对；当前存在现场修改，先保存和审阅 diff，不强制 reset／解压覆盖。

SDK、模型权重、采样数据、凭据、原生二进制和日志不提交 Git。源码仓库克隆后不是可直接运行的完整真机包。

## 6. 接手后的核查与操作顺序

1. 先从 H、V 读取 Git HEAD、`git status --short`、diff和现有服务版本；核对进程工作目录及 Python。不要只照旧 PID 或旧 README 判断版本。
2. 只读查看 HMI `/inspection/state`、检测 `/api/version` 和 `/api/robot/status`；检查反馈新鲜度、三部件状态、任务、相机所有权与软件锁。可运行本仓库 `tools/taizhou/probe_readonly.py`，但其固定PID路径失效时须补查进程。
3. 使用检测工作台现有“完整路线”流程：选择工装及成功视觉任务 → 核对本次目标与路线 → 路线预检 → 页面确认后执行至检测暂停 → 检查后沿原路线撤离返回。检测到位后优先用原路线返回入口；独立复位依赖真实到位来源恢复，不能猜测旧计划。
4. 每次视觉、标定、工具、工位或模型改变后重新生成／校验计划。软件 J7 包络与控制器限位取交集，保留既有余量，不通过放宽限位绕过规划失败。
5. 重启或更新前，在新鲜、停稳、下电／手动／空闲和任务结束条件下备份；保存实时 `motion.latched`。`HELIOS_HMI_START_LATCHED` 只是旧进程启动值，不是实时锁：现场已恢复控制时不得从旧环境把1再次带入。读取失败或退出不明确时停止部署；已有锁通过原恢复流程处理。
6. 真机入口优先沿用现场桌面启动器及实际参数。旧 `open_real.sh` 仍检查历史 revision 标签；当前HMI标签已变化，直接重跑可能报模式不匹配，本次未测试桌面重开。查明原 owner 和启动记录后再维护入口，不并行开第二套 SDK 服务。
7. 新源码验证、原生编译和运行版本核对完成后再切换服务。后续验证至少覆盖状态、图像、路线生成／装载、预检、停止和返回；提速应另测阶段耗时及真实退出。每个动作均按现场既有确认流程进行。

## 7. 验证依据及尚未关闭的事项

历史验证：有效DH独立正解、MuJoCo与官方 `calcFk` 的10组数值查询一致（计算精度，不是实物精度）；返回来源修复的本地及现场候选各154项回归通过、3项跳过，另1项HTTP缓存回归通过。三轮真实检测／返回已有记录，但不能替代更新有效DH及提速后的重新验收。本次文档任务未重跑软件全量测试或执行真机动作。

接手时仍需闭环：提速后整段执行和停止效果；新模型下的独立实物精度；腰部J4下电保持偏移与头部 `Ext Axis 1 / 0x3110` 报警的硬件根因；右臂间歇全零反馈；相机当前所有权／可用图像；现场两仓库未提交的源码和旧启动器版本检查。当前下电空闲和锁已解除，不代表上述历史故障完成根因修复。

本次原始只读回执保存在本报告仓库的 Git 忽略目录 `outputs/diagnostics/ipc-handover-20261020/`，含 `probe.json`、`source_comparison.json`。它们是本地排查留档，不是飞书可下载附件；关键结论、路径、版本及摘要已写入正文。

相关记录（均在本报告仓库 `docs/`）：`jinlong_deployment_20261017.md`、`jinlong_three_round_repeatability_20261020.md`、`taizhou_controller_kinematics_alignment_20261020.md`、`taizhou_effective_dh_usage_expectation_20261020.md`、`taizhou_original_return_restore_fix_20261020.md`、`taizhou_execution_startup_latch_regression_20261020.md`。旧报告的状态带有观测时间，以本次只读状态及后续现场回执更新。
