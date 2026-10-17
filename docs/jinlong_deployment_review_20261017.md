# 金龙仓库更新与工控机部署核对（2026-10-17）

后续已恢复连接并完成软件更新，实际结果见
[部署记录](jinlong_deployment_20261017.md)。下文保留首次核对时的状态和通用部署方案。

## 本次结论

已直接读取远端 Git 仓库并克隆检查：
<https://github.com/thedaosheng/jinlong-motor-arm-control>。
默认分支 HEAD 为 `e226bf2d59fb96096343c0f9504990791ba4daed`，
提交时间换算为北京时间是 2026-10-16 22:46:53。
昨天的更新包括右手持板手眼采样、腰部保持时右臂拖动、右臂出限位、
右臂位置运动速度参数及对应速度监督修正。最后一项修改了 C++，必须重新编译。

本次分别尝试 `jinlong-ipc`（192.168.1.67）和历史地址 192.168.10.66，
SSH 均连接超时。因此没有读取当前工控机版本，没有上传、停服务、切换入口或连接硬件。
既有部署记录只用于定位，不能证明工控机当前状态。

这是源码仓库，顶层分为 `arm_control/` 与 `vision/`。
不能把它当成原单目录项目直接覆盖或 `git pull` 到原现场仓库。
建议在现有 bit-bhr 工控机旁部署一个固定版本的新目录，保留旧版与现场资产，
先模拟验证，再安排现场切换；新工控机则还需要安装 SDK、驱动及依赖并重新核对网络和标定。

## 必须配套处理的内容

| 项目 | 源码情况及部署处理 |
| --- | --- |
| 控制端 Python | `arm_control/requirements.txt` 为 NumPy 和 opencv-contrib-python；现有工控机可先核对、复用旧 `.venv/bin/python` 的绝对路径，不能直接搬迁虚拟环境 |
| 控制端模型 | Git 不含 `helios_hmi/model/`，需从现场保留完整 URDF 和 meshes |
| 控制端前端资源 | Git 不含 `helios_hmi/web/vendor/`，需保留 Three.js、URDFLoader 等资源 |
| 控制端执行文件 | Git 不含 `xcore_state_stream`、`xcore_motion`；用本次源码和官方 xCore C++ SDK 0.7.1 的 x86_64 库重新编译，核对目标系统动态库兼容性 |
| 海康 SDK | Git 不含 SDK；`run_ipc.sh` 默认从 `$HOME/r435-release/r435-vision-1080ti-v1/sdk/haikangSDK` 加载，需保留 SDK 及本机动态库配置 |
| 视觉资产 | 保留 `sdk/`、`models/`、现场 `config/` 和 `calibration/`；不能用新包默认配置覆盖现场参数 |
| 采样和运动基准 | 保留整个 `$HOME/helios_hmi_sessions/`，包括 `motion/waist_hold_reference.json`；不要当作源码同步或提交 |
| 桌面启动器 | `arm_control/open_real.sh` 仍写死旧控制目录，不能直接用来启动新目录；需切换其中 `app` 路径 |
| 视觉与控制关联 | `vision/open_inspection.sh` 默认使用旧控制目录，可用 `R435_HMI_ROOT` 指向新 `arm_control/`；控制端可用 `VISION_ROOT` 指向新视觉运行目录 |

仓库 `vision/README.md` 还保留 UR5、192.168.1.12 相机等旧说明。
当前 `vision/common.sh` 实际设置相机序列号 `00DA8664418`、本机相机接口
`192.168.1.68`，视觉通过 `127.0.0.1:50168` 的 HMI 共享相机和机器人接口。
不能照旧 README 重新配置网卡或启动另一套相机所有者。

## 准备新版本目录

以下命令是供工控机恢复连接后执行的步骤，本次没有在工控机执行。
先核对旧目录是否仍在使用、有无现场修改以及当前启动命令，保存进程、端口和配置清单。
将新源码克隆到新目录；目标机没有 Git 或不能访问 GitHub 时，在开发机克隆后传输源码。

```bash
git clone https://github.com/thedaosheng/jinlong-motor-arm-control.git \
  "$HOME/code/jinlong-release-e226bf2"
cd "$HOME/code/jinlong-release-e226bf2"
git checkout --detach e226bf2d59fb96096343c0f9504990791ba4daed
```

将现场模型、前端资源、SDK、模型权重、配置和标定复制到新版本对应位置，
复制前后记录 SHA-256。对新源码已有的 `vision/config/` 文件逐项比较、合并，
不要整目录覆盖。保留原目录和原桌面入口，尚不停止旧服务。

在有编译环境的 x86_64 机器上执行（SDK 路径按实际填写）：

```bash
cd "$HOME/code/jinlong-release-e226bf2/arm_control"
XCORE_SDK_ROOT=/path/to/xCoreSDK-CPP bash helios_hmi/native/build.sh
```

构建脚本需要 g++、SDK 的 `include/rokae/robot.h`、
`lib/Linux/x86_64/libxCoreSDK.a`、`libxMateModel.a` 和 Eigen。
若在开发机编译，将生成的两个执行文件复制至工控机新版本
`arm_control/helios_hmi/native/`，检查执行权限、`file`、`ldd` 和文件哈希。
旧版二进制不能代表本次 C++ 修改已部署。

先用现有 Python 环境进行依赖和测试检查；首次部署则新建 Python 3.10 环境并安装 requirements。
补齐模型和前端资源后，可在工控机新目录做模拟检查：

```bash
cd "$HOME/code/jinlong-release-e226bf2/arm_control"
ipc_python="$HOME/code/jinlong-motor-arm-control-codex-send-stop/.venv/bin/python"
"$ipc_python" -m pip check
"$ipc_python" -m unittest discover -s helios_hmi/tests -v
"$ipc_python" -m unittest discover -s tests -v
env HELIOS_HMI_ALLOW_REAL=0 "$ipc_python" -m helios_hmi \
  --fake --no-camera --host 127.0.0.1 --port 50169 \
  --sessions /tmp/jinlong-e226bf2-simulation
```

浏览器检查 `http://127.0.0.1:50169/` 的模型和资源，检查后退出模拟服务。
该检查不占用原 50168 端口，也不访问机器人或相机。

## 现场切换与回退

1. 先确认没有动作、拖动、保持或采样会话，正常结束旧会话并退出旧服务；
   确认旧进程退出、50168/8765 端口释放及 RobotAssist 等控制客户端完成交接。
   不抢锁、不删除动作锁文件，不同时启动新旧两个 SDK 所有者。
2. 调整新启动器的目录、`HMI_PYTHON`、`VISION_ROOT`，先启动不带
   `--enable-write` 的真实反馈模式。真实状态连接也会创建 SDK 对象，必须在交接之后执行。
   核对左右臂、本体反馈新鲜度，控制器身份、20 个模型关节和实际网卡地址。
3. 使用新视觉运行目录和现有视觉 Python 环境，先以
   `bash start.sh start --defer-camera-start` 检查网页入口，再核对相机共享和真实帧。
   此前必须补齐现场标定文件，延迟启动相机不会绕过标定身份检查。
4. 核对实际进程命令、源码路径、Python 路径、两个原生程序的哈希、页面资源和真实状态，
   再将桌面入口切到新版本。旧服务仍在时启动器会复用原端口；打开网页不证明完成升级。
5. 如需恢复现场受控功能，再以原有授权范围启动写入模式。
   `run_ipc.sh` 默认 `HELIOS_HMI_ALLOWED_GROUPS=left_arm,waist`，
   `HELIOS_HMI_DRAG_ONLY_GROUPS=right_arm`；右臂默认仅拖动，不开放位置下发。
   需要右臂位置运动时须显式将 right_arm 纳入 ALLOWED_GROUPS 并现场验收。
   `vision/open_inspection.sh` 会带 `--enable-write` 和 `--enable-robot-motion`，
   不适合作为首次只读检查入口。
6. 回退时先正常结束新版本会话、停止新服务、确认退出，再恢复旧桌面入口和原启动命令。
   旧代码、环境、现场资产和采样目录始终保留，不通过覆盖文件来回退。

控制界面为 `http://127.0.0.1:50168/`，视觉界面为 `http://127.0.0.1:8765/`。
远程查看优先使用 SSH 端口转发；控制端写入模式要求回环监听。

## 本次离线验证

针对远端 `e226bf2`，在开发机临时克隆目录检查，不修改远端源码或现场文件。

- 原生离线测试 114 项，113 项通过、1 项错误：
  `NativeCleanupRegressionTests.test_cleanup_and_latest_delta_regressions`
  内部程序退出码 18，对应测试仍要求 12 秒冷却，而实现已改为 1 秒。
  上游 README 也记录了这个既有不一致；本次没有将其改为通过。
- 视觉全新克隆直接运行 `scripts/run_offline_tests.py --isolated` 会因缺少
  `calibration/` 目录报 FileNotFoundError。仅在临时克隆中建立空目录后重试，
  276 项测试成功，其中 3 项跳过；隔离脚本使用明确未验收的合成标定，不是现场标定。
- 控制端 HMI 回归 269 项成功，其中 5 项因缺少 Node 跳过；临时使用开发机旧项目已有
  模型及前端资源，测试进程清除代理变量，避免本机 HTTP 测试受代理干扰。
- 本文 3 段 shell 示例通过 `bash -n` 静态语法检查；没有实际运行其中的部署命令。

`VALIDATION.md` 记录的是 10 月 16 日初始源码导入验证，不能当作最新所有控制改动的验证结果。
源码快照清单 `SOURCE_SNAPSHOT.json` 同样描述初次导入；更新后的完整运行包应重新生成部署清单。
视觉 `verify.sh bundle/installed` 依赖 Git 未收录的发布清单，不能直接宣称新克隆通过发布校验。
控制端 `source_version()` 仅识别子项目目录内 `.git` 或 `SOURCE_MANIFEST.json`，
新仓库 `.git` 在上层，可能报告 source_version_unavailable；上线需提供对应子项目的
真实来源清单或修正版本发现逻辑，并记录二进制哈希，不能只用页面固定 revision 判断版本。

运行日志保存在开发机 `/tmp/jinlong-deploy-*-20261017*.log`，不入库。
本次没有完成目标机环境检查、编译链接、真实图像或运动验收。
