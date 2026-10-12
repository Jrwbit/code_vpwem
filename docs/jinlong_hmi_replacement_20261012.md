# bit-bhr 界面版本替换

后续已完成真机后端适配并切换为真实模式，见文末。以下首次替换记录中的模拟状态不再代表当前运行状态。

2026-10-12（Asia/Shanghai），使用用户提供的
`/home/wjr/下载/jinlong-motor-arm-control-codex-send-stop.zip`
替换工控机旧版。通过 `jinlong-ipc`（`bit-bhr@192.168.1.67`）部署。

- 压缩包 SHA-256：`a51c5a0188e2ab83256ee51c4d1b6068e7ed0caca8100f364683c42476deb865`，两台机器核对一致。
- 新目录：`/home/bit-bhr/code/jinlong-motor-arm-control-codex-send-stop`。
- 复用旧版 `.venv` 的依赖，并用现有 Conda Python 3.10 重新生成虚拟环境入口，修正搬迁后的路径。
- 删除工控机旧目录 `/home/bit-bhr/code/jinlong-motor-arm-control-main` 和旧压缩包 `/home/bit-bhr/Downloads/jinlong-motor-arm-control-main.zip`。
- 旧版 `logs`、`Mv3dRgbdLog`、`_rokae_log_` 移至 `/home/bit-bhr/.local/state/jinlong-hmi/previous-version`；标定数据仍在 `/home/bit-bhr/helios_hmi_sessions`。
- 启动器源文件在 `tools/jinlong_hmi/`，分别部署为新目录下的 `open_simulation.sh` 和桌面的 `Jinlong-HMI.desktop`。

## 打开与运行模式

工控机桌面双击“金龙机械臂界面（新版模拟）”，或执行：

```bash
bash /home/bit-bhr/code/jinlong-motor-arm-control-codex-send-stop/open_simulation.sh
```

服务使用 `--fake --no-camera`，只监听 `127.0.0.1:50168`；Firefox 已在当前桌面打开“珞石控制窗”。启动器日志和进程号位于 `/home/bit-bhr/.local/state/jinlong-hmi/`。

新包的真实设备后端仍使用 ARM/NX C++ SDK、本机地址 `192.168.71.51` 和 `/home/admin` 路径；本次没有移植真实状态、相机或运动后端。模拟界面返回的20个关节是合成数据，`write_enabled=false`，未连接控制器或相机。真实模式需要另行适配 x86_64 工控机。

## 验证

- `.venv/bin/python -m pip check` 通过。
- `.venv/bin/python -m unittest discover -s helios_hmi/tests -v`：135项，结果 OK；缺少 Node，跳过1项浏览器运动租约测试。
- 启动器 `bash -n` 和桌面入口 `desktop-file-validate` 通过。
- 主页、状态、Three.js、URDFLoader、机器人 URDF 均返回 HTTP 200。
- X11 窗口树确认“珞石控制窗 — Mozilla Firefox”可见；未自动验证浏览器内3D渲染或任何真实运动。

压缩包、模型、依赖环境、标定数据和运行日志不提交到本仓库。

## 后续真机适配与切换

按用户要求解决模拟启动和后端不兼容两个问题，源码适配提交为
`f0bef3b`，位于开发机 `/home/wjr/code/jinlong-motor-arm-control-codex-send-stop`，
已部署到工控机同名目录。源代码导入提交为 `1a89c8f`。
详见该项目 `README_IPC.md`。SDK、原始模型和编译二进制不入库。

- 从珞石官方C++ SDK v0.7.1获取x86_64静态库，校验发布包digest；使用SDK自带Eigen在开发机编译，工控机无需安装编译器。
- 修正机器人本机IP为192.168.71.157；使用官方七轴接口xMateErProRobot，核对真机型号、控制器ID、关节数量和软限位。
- 状态读取器的SDK重连与运动执行器互斥，避免运动期间构造新SDK对象重置队列；修正保留轴反馈的停稳检查。
- 相机按序列号和192.168.1.68接口精确选择；采用已验证的RGB配置并回读参数。
- 服务以`--enable-write`启动，监听127.0.0.1:50168，允许运动组仅left_arm；独立腰部、右臂入口禁用。左臂示教保留原有腰部当前位置保持逻辑。
- RobotAssist的控制器连接已由用户确认断开，并用socket状态复核；切换后读到全部20个真实模型关节，三部分off/manual/idle。
- `open_real.sh`和桌面入口已更新，Firefox已打开真实模式网页。启动不发送上电、运动或拖动命令。
- 左臂0.1°仿真目标的真机预览通过，之后恢复真实目标；未提交运动。未确认拖动请求返回409，右臂请求返回group_not_enabled。
- 网页保存1组真实停稳样本：80角点，重投影误差0.288528px，位于`/home/bit-bhr/helios_hmi_sessions/handeye_ipc/20261012_194851_936c18471a05`。
- 工控机138项Python测试通过（缺Node跳过1项），新增启动限制测试另行通过；开发机52项C++离线测试通过。依赖、脚本语法和桌面入口校验通过。

实际左臂拖动、松键停止、暂停恢复和腰部保持尚需现场验证；本次没有发送实际运动、拖动或上下电指令。当前已有真机控制路径与真实采样，仍需足够多的不同姿态才能完成标定解算和精度验收。
