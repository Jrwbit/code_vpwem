# bit-bhr 图形代理客户端安装记录

日期：2026-10-12（Asia/Shanghai）。

- 目标：`bit-bhr`，通过已有 SSH 别名 `jinlong-ipc` 连接 `192.168.1.67`。
- 系统实测：Ubuntu 24.04.3 LTS，x86_64；bit-bhr 用户已登录 X11 桌面。
- 安装：Clash Verge Rev 2.5.8，使用官方 amd64 DEB，通过 `sudo apt-get install -y /tmp/Clash.Verge_2.5.8_amd64.deb` 完成系统级安装。仅新增 clash-verge 包，已有依赖满足要求，无其他包升级或卸载。
- 发布页：https://github.com/clash-verge-rev/clash-verge-rev/releases/tag/v2.5.8
- 安装包 SHA-256：`ceeb7578c6f030b33ddf6346486db24013cf4557cc340de5f312f98a924a8546`；下载后在开发机和目标机核对，均与 GitHub Releases API 的 digest 一致。

## 验证结果

- `dpkg-query -W clash-verge` 返回 `2.5.8`；`dpkg --audit` 无异常。
- `/usr/share/applications/Clash Verge.desktop` 通过 `desktop-file-validate`。
- 从已登录用户的桌面会话启动 `/usr/bin/clash-verge`，X11 窗口树确认 `Clash Verge` 主窗口（940×700）及托盘图标存在。
- `/usr/bin/verge-mihomo -v` 返回 Mihomo Meta v1.19.32；进程检查确认内核启动，应用日志显示配置校验成功。
- 初次启动日志提示未选择配置；这是尚未导入订阅的状态，未验证实际代理节点连通性。
- GNOME 系统代理模式为 `none`。未安装特权后台服务，普通图形客户端使用 sidecar 内核；TUN 模式未配置或验证。
- 临时用户目录解包副本和目标机 `/tmp` 安装包已清理。应用自动生成的配置、日志和二进制均不入库，登录凭据未写入文件。

## 使用

在应用菜单搜索 **Clash Verge** 打开。进入“订阅”导入自己的 Clash 订阅链接或配置文件，选择节点后再开启“系统代理”。安装完成时窗口已打开，但尚无可用订阅。
