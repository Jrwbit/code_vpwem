# 右臂标定、三工装回放与检测界面（2026-10-17）

本次通过工控机 Wi-Fi 地址 192.168.0.145 读取保存记录并打开桌面浏览器。
没有执行运动、采样、重新解算或应用标定。

## 手眼标定实际状态

工控机会话
`~/helios_hmi_sessions/handeye_ipc/20261016_210737_3233a5f3f80f/result.json`
明确保存 `calibration_arm=right_arm`、7 组样本、PARK 方法，`ok=true`、`pass=true`。
平移离散为 1.4037 mm，旋转离散为 0.2770°，通过本次内部一致性门限。
这些是参与标定样本的内部统计，不是独立定位精度验收。

结果同时保存右臂基座中的相机位姿和转换后的左臂 SDK 基座相机位姿；
跨左右臂转换使用 URDF 安装关系，文件注明安装公差未实测。

检测端活动文件 `~/r435-release/r435-vision-1080ti-v1/calibration/rokae_current.json`
仍指向 10 月 14 日的 `20261014_100117_95c3977a42a7` 会话，18 组样本、HORAUD 方法，
`independent_validation.accepted=false`。因此“右臂已完成标定解算”与
“检测项目已采用右臂标定”是不同状态；后者尚未完成，本次未修改活动标定。

## 三工装仿真回放

来源为 `thedaosheng/jinlong-motor-inspection` 的
`fca8b7ff993c37404ff84ab6b2830134695cc391`，
取其中 `output/20261016-helios-compact/` 的 `full.mp4`、`dryrun.mp4`、
`insert_closeup.mp4`、`home.png` 和 `report.json`，复制到工控机：

```text
/home/bit-bhr/jinlong-fixture-replay/20261016-compact/
```

五个文件传输后 SHA-256 全部一致；三个视频用工控机现有 OpenCV 成功打开和解码首帧。
完整视频为 4048 帧／30 fps，约 135 秒。视频是历史规划回放，
不依赖现场机器人在线，不会向机器人下发命令；不是根据当前相机观测重新规划的结果。
来源说明记载该方案已使用新相机安装信息，但盒位仍沿用旧观测，不能直接拿来当前插接。

新增桌面入口“三工装仿真回放”，源文件在 `tools/jinlong_hmi/`：

- `Fixture-Replay.desktop` 安装至 `~/桌面/Fixture-Replay.desktop`。
- `open_fixture_replay.sh` 安装至 `~/.local/bin/open_jinlong_fixture_replay.sh`。
  默认打开完整版，也可传 `dryrun` 或 `insert_closeup` 打开空跑或插接特写。

脚本通过 `bash -n`，桌面文件通过 `desktop-file-validate`，已设置执行权限和桌面信任标记。
视频、报告和标定文件没有加入本仓库。

工控机控制端 venv 和视觉 Conda 环境均没有 mujoco、qpsolvers、daqp。
所以当前已经部署的是可直接打开的视频回放，可交互／重新求解的仿真环境尚未部署。

## 已打开的界面

在工控机当前 GNOME 桌面启动 Firefox，浏览器恢复记录确认存在：

- `http://127.0.0.1:8765/`，标题“台州电机检测 · 珞石左臂”。
- `file:///home/bit-bhr/jinlong-fixture-replay/20261016-compact/full.mp4`，标题“full.mp4”。

这是实际桌面浏览器中的两个窗口；没有只用后台 HTTP 检查代替打开页面。
机器人／相机在线状态及活动标定的问题仍需另外处理，打开界面和观看回放不会解决它们。
