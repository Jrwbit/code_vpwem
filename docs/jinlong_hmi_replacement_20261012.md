# bit-bhr 界面版本替换

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
