# 金龙机械臂界面工控机检查（2026-10-12）

## 部署位置与环境

- 工控机：`bit-bhr@192.168.10.66`，Ubuntu 24.04.3，x86_64。
- 原压缩包：`/home/bit-bhr/Downloads/jinlong-motor-arm-control-main.zip`。
- SHA-256：`0097f44441066dad1d9538d15207e0d84539206c41a5bd4e7203ff6612bf3889`。
- 源码：`/home/bit-bhr/code/jinlong-motor-arm-control-main`。
- 独立虚拟环境：源码目录下 `.venv`，Python 3.10.21、NumPy 2.2.6、opencv-contrib-python 4.14.0.94。
- 使用现有 Conda 环境的 Python 创建独立 venv，未修改现有环境；依赖通过清华 PyPI 镜像安装。

## 可用范围与验证

这是只读机器人状态界面、模拟目标编辑和眼在手外标定程序。`POST /send` 固定返回 403 / `write_loop_disabled`，不能执行运动控制。

`pip check` 通过；`python -m unittest discover -s helios_hmi/tests -v` 的 14 项测试全部通过，覆盖运动学、合成标定、状态映射及 HTTP 接口。

使用临时目录和随机本地端口启动 `--fake --no-camera`：模拟状态就绪，写入禁用；主页、Three.js、OrbitControls、URDFLoader、机器人 URDF 均返回 HTTP 200。验证后进程停止，临时目录清理。未验证浏览器实际渲染和真实硬件采集。

工控机本地模拟启动命令：

```bash
cd /home/bit-bhr/code/jinlong-motor-arm-control-main
.venv/bin/python -m helios_hmi --fake --no-camera --host 127.0.0.1 --port 50168
```

浏览器访问 `http://127.0.0.1:50168/`。远程查看可通过 SSH 转发该端口。

## 真实设备模式的阻碍

1. `helios_hmi/run_nx.sh` 的 Python、海康库及 SDK 路径为 `/home/admin/...` 和 aarch64；本机是 x86_64，不应直接运行该脚本。
2. `native/build.sh` 固定使用 aarch64 的 C++ xCore SDK 静态库。目标机检查范围内未找到 `libxCoreSDK.a`、`libxMateModel.a`、`rokae/robot.h`，也没有 g++、Eigen 和 Git 可执行文件。已有 `/home/bit-bhr/sdk/xcore-0.7.1/Release/linux` Python 扩展，但不能直接用于该 C++ 构建脚本。
3. `state.py` 本机地址固定为 `192.168.71.51`；目标机当前机器人网段地址为 `192.168.71.157`。SDK 库路径也固定为 aarch64。
4. `camera.py` Python SDK 路径固定为 NX 路径。目标机已有海康 SDK 模块：`/home/bit-bhr/r435-release/r435-vision-1080ti-v1/sdk/haikangSDK/Res/demo/python/Mv3dRgbdImport/Mv3dRgbdApi.py`，需要适配路径及动态库环境。
5. 相机选择逻辑没有匹配网卡时默认选择设备 0；相机初始化会加载用户配置和设置输出模式。真实采集前应确认选中设备及现有程序是否占用。
6. `HandEyeSession.remove` 只从内存移除样本，下一次采集用当前样本数作为编号；删除非末尾样本后继续采集可能重复编号并覆盖目录。正式标定前应修复。

本次仅解压、安装独立环境、阅读代码与执行模拟检查；未修改项目源代码，未启用自启动，未发送机器人指令或打开相机。
