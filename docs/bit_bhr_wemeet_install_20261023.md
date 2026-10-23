# 工控机腾讯会议安装记录（2026-10-23）

腾讯会议 **3.26.10.401** 已在工控机 `bit-bhr` 的当前用户下安装并启动，应用菜单和桌面均已添加入口。安装采用用户目录解包方式，**未注册为 dpkg 系统软件包**；原因是本机 sudo 操作需要密码。

## 安装对象与位置

| 项目 | 核对结果 |
| --- | --- |
| 目标 | SSH `192.168.1.67`，主机及用户均为 `bit-bhr` |
| 系统 | Ubuntu 24.04.3 LTS，amd64 |
| 下载包 | `/home/bit-bhr/下载/TencentMeeting_0300000000_3.26.10.401_x86_64_default.publish.officialwebsite.deb` |
| 包信息 | Package `wemeet`，Version `3.26.10.401` |
| SHA256 | `70f37b029209c0ef91be9813eebf6650c6cb9a00f7c718490158773e2fe6acaf` |
| 程序目录 | `/home/bit-bhr/.local/opt/wemeet-3.26.10.401/opt/wemeet` |
| 应用菜单入口 | `/home/bit-bhr/.local/share/applications/wemeetapp.desktop` |
| 桌面入口 | `/home/bit-bhr/桌面/wemeetapp.desktop` |
| 启动器源文件归档 | 本仓库 `tools/desktop/wemeetapp-bit-bhr.desktop` |
| 启动日志 | `/home/bit-bhr/.local/state/wemeet-install/launch.log`，不纳入 Git |

保留原下载包，沿用厂商使用相对路径的启动脚本，未修改该脚本。桌面入口已设置可执行及 `metadata::trusted=true`。

## 验证及范围

- 包声明的依赖均已安装；将 `LD_LIBRARY_PATH` 指向随包提供的 `lib` 后，`ldd` 未发现 `not found`。
- `desktop-file-validate` 通过，`update-desktop-database` 执行成功。`xdg-mime query default x-scheme-handler/wemeet` 返回 `wemeetapp.desktop`，会议链接协议已关联该入口。
- 工控机两处启动器与本仓库归档文件的 SHA256 均为 `5f2e8bdcec112fa382e99452f72137818920fea32259ab80bc51e444e5a5d3d2`；`gio` 回读桌面入口 `trusted=true`。
- 通过 `gtk-launch wemeetapp` 成功启动；当次进程 PID 为 `4075410`，`/proc/4075410/exe` 指向上述安装目录。`xwininfo -id 0x800017` 确认标题为“腾讯会议”、尺寸为 `377×669` 的窗口状态为 `IsViewable`。
- 原桌面会话为 Wayland，厂商启动脚本自动使用 Xwayland。启动日志出现一条 `libEGL DRI3` 警告，但窗口已创建；尚未据此确认音视频功能是否受影响。

本次未登录腾讯会议，未测试音频、视频或屏幕共享，安装与窗口创建成功不等于会议功能验收。未修改机器人服务。
