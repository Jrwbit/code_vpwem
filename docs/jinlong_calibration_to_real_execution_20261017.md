# 新标定导入与左臂真机执行流程（2026-10-17）

适用方案：右臂持板完成标定，使用换算到物理左臂 SDK 基座的结果，由左臂执行电机检测。用户已确认此方案。
本文依据本地源码和已有现场记录整理；本次没有连接 IPC、修改活动标定或执行机器人运动。现场状态须重新读取。

## 1. 当前依据与坐标约定

最近现场记录见 [右臂标定与回放记录](jinlong_handeye_replay_20261017.md)：

- 候选会话：`~/helios_hmi_sessions/handeye_ipc/20261016_210737_3233a5f3f80f`，右臂、7 组、PARK，内部质量门通过。
- 当时检测活动标定仍为 `20261014_100117_95c3977a42a7`，独立点验收未通过。此状态是历史快照。
- 检测及生产执行绑定左臂 `192.168.71.161`；标定持板右臂为 `.160`。

新 HMI 的 `solve()` 保存三个不同用途的矩阵：

| 字段 | 含义 | 本流程用途 |
| --- | --- | --- |
| `T_armbase_cam` | 相机到持板手臂 SDK 基座，本次为右臂 | 保留原始解算证据 |
| `T_base_cam` | 相机到物理左臂 SDK 基座 | 导入检测系统 |
| `T_base_link_cam_urdf` | 相机到整机 URDF 基座 | 仿真及坐标核对 |

变换约定为 `A_T_B` 把 B 系坐标变到 A 系，矩阵平移单位为米。
右臂解算换算为左臂结果：`left_T_camera = left_T_right × right_T_camera`。
HMI 已完成此换算，不要再重复乘一次；不要将右臂 `T_armbase_cam` 直接写入左臂活动标定。
`left_T_right` 使用 URDF 名义安装关系，装配公差未实测，因此必须在左臂坐标系完成独立点验证。

## 2. 新标定导入全流程

### 2.1 核对来源并备份

先在现有 HMI 和检测页面确认：没有运动、拖动、采样、回放执行、视觉任务或待执行计划；左右臂和腰部均下电、手动、空闲、停稳，反馈新鲜，没有退出未确认状态。
导入脚本只有文件写锁，没有完整设备空闲检查，不能在任务运行中直接调用。

在 IPC 保留原会话、旧活动标定、工位配置与摘要。以下命令只用于完成现场空闲核对后，按已记录路径操作：

```bash
session_dir="$HOME/helios_hmi_sessions/handeye_ipc/20261016_210737_3233a5f3f80f"
vision_root="$HOME/r435-release/r435-vision-1080ti-v1"
backup_dir="$HOME/r435-release/backups/handeye-import-$(date +%Y%m%d-%H%M%S)"
mkdir -m 700 "$backup_dir"
cp -a "$session_dir" "$backup_dir/"
cp -a "$vision_root/calibration/rokae_current.json" "$backup_dir/rokae_current.before.json"
cp -a "$vision_root/config/rokae_site.json" "$backup_dir/rokae_site.before.json"
sha256sum "$session_dir/result.json" \
  "$backup_dir/rokae_current.before.json" "$backup_dir/rokae_site.before.json" \
  > "$backup_dir/before.sha256"
```

任一步失败先解决，不继续导入。备份和标定数据留在现场，不提交 Git。
确认 `result.json` 的 `ok/pass` 都为 true、`calibration_arm=right_arm`、样本数及来源正确，
`frames.T_base_cam` 声明物理左臂 SDK 基座，并复核保存的右转左矩阵关系和标定时腰部四轴基准。
不要仅靠修改 frames 文本让不匹配的矩阵通过校验。

### 2.2 使用已有导入入口

确认现场运行源码具有同版本导入脚本后，用检测环境执行：

```bash
cd "$vision_root"
/home/bit-bhr/miniconda3/envs/motor-inspection-1080ti/bin/python \
  scripts/import_rokae_calibration.py "$session_dir"
sha256sum calibration/rokae_current.json
```

脚本检查内部质量、刚体矩阵、左臂坐标声明和腰部基准，将 `T_base_cam` 写为活动文件的 `base_T_camera`，
绑定左臂及相机身份，保存源文件摘要、会话、样本数和内部统计，以写锁和原子替换更新文件。
它会将 `independent_validation.accepted` 置为 false；旧标定的验收不能沿用。

### 2.3 确认检测服务确实读到新标定

打开 `http://127.0.0.1:8765/`，刷新机械臂状态，或只读获取：

```bash
curl --fail --silent --show-error http://127.0.0.1:8765/api/robot/status
```

核对 `calibration.available=true`，`version` 为新会话名，`base_frame=rokae_left_sdk_base`，
`sample_count=7`，身份和文件 SHA-256 正确；页面应显示独立精度未验收。
如果服务仍显示旧版本，排查实际运行目录、端口进程和配置路径，不启动第二个 SDK owner。
不因导入而默认需要重启服务；以真实状态回执为准。

导入后废弃旧视觉结果和旧运动计划，重新采图运行全流程检测。
确认新产物中的标定摘要与当前活动文件一致；历史视频、旧盒位和导入前的检测结果不能复用。

## 3. 从新标定到真机可执行

### 3.1 独立定位验收

在工程参数中先填写现场允许位置误差 `acceptance_metrics.position_error_mm`。
采集至少三个不同、非共线且未参与标定的现场点，覆盖实际作业区域；每点提供：

- 相机坐标 `point_camera_m`。
- 独立测量的物理左臂 SDK 基座坐标 `measured_point_base_m`。
- 唯一 ID、`used_for_calibration=false` 和测量方法／记录 `measurement_evidence`。

页面生成验收模板，填入真实测量值和当前标定摘要，确认 `field_measured=true` 后提交。
现有接口为 `POST /api/robot/calibration/validate`，后台计算每点误差，最大误差不超过配置门限才通过。
实测基座坐标必须来自独立测量，不能用待验收矩阵反算后当作真值。

该验证检查点定位；工装姿态及实际插接能力还需分别验证。

### 3.2 补齐工位参数与验证记录

通过页面保存真实工位参数，数据落在 `config/rokae_site.json`：

| 内容 | 对应参数／操作 |
| --- | --- |
| 法兰到工具的完整变换 | `flange_T_tcp`，包含平移和旋转 |
| 各工装安装关系与型号 | `tcp_T_objects`、`selected_fixture`，JL100／JL132／JM160 |
| 安全初始姿态 | 在已示教并确认的姿态下，下电停稳后“保存当前位置为初始位置”，保存法兰和七轴反馈 |
| 空间限位与过渡路径 | `workspace_m`、`transit_direction_base`、`transit_clearance_mm`、`approach_mm` |
| 动作参数 | `linear_speed_mm_s`、`speed_percent`、`dwell_s` |
| 工装补偿 | `tcp_z_adjustment_mm`、`fixture_press_mm`；TCP 补偿和接触后下压量分开 |
| 到位验收标准 | `arrival_error_mm`、`arrival_rotation_deg` 等现场约定指标 |

读取当前部署版对速度的限制；核对的本地生产参数校验要求关节速度 1%、直线速度 `(0,10] mm/s`。
自动标定使用的 20% 速度不是生产动作的速度设置。
工装坐标轴方向必须实测，不能凭世界坐标 Z 判断接近／撤离方向。

完成 TCP、工装、工作空间、路径、姿态、控制及接触的实际验证，再记录对应验收状态。
修改几何或路径会撤销有关验收及计划，不可批量将 `*_verified` 改为 true 绕过检查。

### 3.3 确认真机入口与首次路径

使用现场已有单一 HMI owner，确认 HMI 写入开放、检测服务启用 `--enable-robot-motion`，
所有部件反馈新鲜且停稳，无报警阻断、busy/capturing/latched，腰部与新标定基准一致。
腰部“回正”使用保存且与本次标定兼容的姿态；不是关节归零。

先用当前相机观测、当前工具几何及当前机器人姿态验证完整路径，包括机器人本体、工装和现场障碍。
空间 XYZ 限位不等于完整碰撞检测。10 月 16 日回放使用历史盒位，不能作为本次路径验证。
较新源码中的 `hover_preview` 输出 `executable=false`，只生成非接触目标，不构成真机执行入口；不能把它或 MP4 当作已完成空跑。
首次联调先验证明确保持间隙的非接触路线，再开展轻触和下压的现场验证，不直接启动整套插接。

## 4. 正式执行按钮顺序

完成以上条件后，在检测页面操作：

1. 确认腰部基准、相机在线及安装不变，选择与实物一致的工装及检测型号。
2. 按现场方案运行四或五的完整视觉检测，等待成功和位姿质量门通过。
3. 点击“生成位置”，检查新计划绑定的任务、标定摘要、工装和目标，查看全部阻断原因。
4. 阻断消除、路径已验证后，逐步点击执行按钮，每步等待真实到位及退出回执。

生产动作按六阶段执行：

`保存的初始位置 → 旋转工装并到盒上方 → 名义轻触位置 → 按型号补偿继续下压 → 检测／保持 → 撤离并回初始位置`

生产按钮由既有控制通路处理模式切换、上电及真实反馈；不要额外启动机器人脚本或以手动保持上电替代执行入口。
现有流程保留分步下电和冷却，逐步执行不等于一次点击自动跑完整周期。
页面心跳、计划有效期和标定／配置摘要均参与检查，切换页面、停止、状态过期或修改相关参数后需要重新生成有效计划。
停止未确认时使用现场急停，确认真实停稳后再处理。

“轻触”是位置目标，不能仅凭到位宣称已有力控接触证据；“检测／保持”沿用停留逻辑，已有记录尚未确认外部电测接口，不能据此宣布电测通过。

## 5. 完成依据与回退

导入完成：活动文件、页面和新视觉产物都指向新会话及一致摘要。
定位通过：左臂坐标中的独立点实测误差达到现场门限。
真机通过：保存每阶段目标、真实到位误差、退出回执、接触／下压实测和最终回初始姿态证据。
相关运行产物包括 `rokae_target_preview.json`、`rokae_execution.jsonl`，保留在现场，不提交 Git。

需要回退时，先停止并确认设备空闲，保存本轮标定和配置，以标定写锁及原子替换方式恢复本次备份的旧活动文件。
不要覆盖之后新增的工位参数。回退后重新核对版本、腰部基准和验收状态，废弃新旧计划并重新视觉检测。

源码核对位置：检测端 `scripts/import_rokae_calibration.py`、`rokae_calibration.py`、
`robot/rokae_site.py`、`robot/rokae_controller.py`、`robot/rokae_planner.py`、`web/rokae.js`；
新版控制端 `arm_control/helios_hmi/server.py` 的 `solve()`。
更多既有动作与联锁说明见 [机械臂操作记录](taizhou_robot_workflow_20261013.md)。
