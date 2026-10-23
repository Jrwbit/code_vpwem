# 2026-10-23 台州检测与金龙页面入口修复

用户反馈工控机的台州检测和金龙页面打不开。本次已修复两个桌面启动器，部署到工控机并执行验证。两个网页服务原本正常，故障发生在启动器打开浏览器之前。

## 现场原因与修复

| 入口 | 复现证据 | 修复 |
| --- | --- | --- |
| 金龙机械臂界面 | `open_real.sh` 固定要求 revision 为 `codex-send-stop-ipc`，活动服务为 `gpu-task-b8b459b74d10`，虽是真机可写模式仍以“Port 50168 belongs to another mode”退出1 | 按可写模式、HMI状态结构、左右臂物理角色校验，不再用发布标签判断运行模式 |
| 台州电机检测 | `/health` 返回503、提示“相机已交接给检测任务”；`/status` 和首页均200。启动器却因PID存在但相机健康检查失败退出1 | 桌面入口使用 `--defer-camera-start`，以网页响应判断页面可打开；相机由页面原流程管理。普通相机启动模式仍检查 `/health` |

同时关闭后台服务和浏览器继承的启动锁描述符，避免后台进程意外持有启动锁。保留原有唯一SDK所有者检查。修复不依赖清除机器人锁、不自动启动相机，也没有改动运动控制逻辑。

## 验证与范围

- 工控机服务主页 `http://127.0.0.1:50168/` 和 `http://127.0.0.1:8765/` 均返回200。
- 修改前，两个原始启动器分别退出1；修改后，以工控机桌面会话环境执行，两个均退出0，台州报告“服务已就绪”。
- 通过SSH转发对工控机实际页面做浏览器加载检查：金龙标题“珞石控制窗”，台州标题“台州电机检测 · 珞石左臂”；两页均无JavaScript异常、无失败请求、无写请求。下面为独立浏览器渲染截图，不是工控机桌面截屏。
- 六项隔离回归全部通过：更新版本且锁存时可开金龙、错误模式/角色拒绝、相机健康503时可开台州、相机状态503时可开台州、真正网页HTTP错误拒绝、普通启动仍要求相机健康；另通过Bash语法和Git差异检查。
- 整个部署未重启服务：HMI PID仍为2174628，检测PID仍为2923440。10:27（北京时间）复核 `busy=false`、`capturing=false`、`latched=true`；原有机器人安全退出锁保留。这次交付是页面入口恢复，不是机械动作、电测或此前J4故障的验收。

![金龙页面正常加载，保留故障锁提示](../outputs/diagnostics/desktop-entry-20261023/jinlong.png)

![台州检测正常进入视觉任务选择页面](../outputs/diagnostics/desktop-entry-20261023/taizhou.png)

## 代码、部署与留档

统一源码仓库 `/home/wjr/code/jinlong-motor-arm-control` 提交 `885a22713afd440df01d2171136b1025d091ce1e`，变更为 `arm_control/open_real.sh`、`vision/open_inspection.sh`、`vision/start.sh` 及隔离回归 `arm_control/tests/test_desktop_launchers.py`。本报告仓库同步修正 `tools/jinlong_hmi/open_real.sh`，避免后续重新部署旧判断。

工控机已部署三个脚本，逐文件核对原摘要、备份、Bash语法及新摘要。备份目录为 `/home/bit-bhr/r435-release/backups/desktop-entry-20261023-102728/`，含原文件和部署manifest。现场两个仓库的既有未提交变动原样保留；本次分别只提交启动器：HMI `84fea0987959a9e9da930ddc384d6ba8920b197d`，检测 `8cd0987bf07ac156bd6d0b30c612e39e1d1a08b5`。现场Git缺少身份配置，本次沿用统一源码仓库身份进行单次提交，没有改全局配置。

脱敏核查、部署摘要和截图保存在本仓库Git忽略目录 `outputs/diagnostics/desktop-entry-20261023/`。截图随正文追加到[既有飞书全流程报告](https://feishu.cn/docx/I2GAdun3nofskUxAOjWcxVwjnJO)，不将本地路径作为飞书下载附件。
