# RobotAssist 卡住后的程序重启（2026-10-21 10:43，北京时间）

用户要求“控制器卡死了，重启一下”。现场已有控制器状态流持续返回新鲜反馈，三部件均off/manual/idle、停稳，HMI无活动运动或采样，busy=false、latched=false。按工控机原厂控制界面卡住处理，本轮重启了RobotAssist桌面程序；没有重启控制柜硬件、整台机器人或工控机，不能据此宣称硬件控制器重启完成。

实际程序为 `/home/bit-bhr/apps/UbuntuRobotHMI_3.2_AR/RobotHMI/RobotAssist`。向核对过的旧进程1490648发送SIGTERM后正常退出，未使用SIGKILL；沿用原启动参数、工作目录和桌面环境启动新进程2592235。X窗口树确认新RobotAssist主窗口已打开，大小1097×686；随后复查新进程仍存活。

重启后左右臂、身体反馈年龄约0.03～0.04秒，均下电、手动、空闲，最大反馈速度不超过0.00105 rad/s。busy=false、latched=false、capturing=false。原运动失败记录仍保留；本次没有自动清警、上电、腰部复位或恢复检测。未验证RobotAssist自身是否已连接控制器、当前报警消失或交互问题长期解决；持续新鲜反馈来自既有HMI状态流。

现场重启记录及程序输出位于 `/home/bit-bhr/.local/state/robotassist-restart/20261021-104311/`，含restart.json、application.log、windows.txt。启动环境及程序输出不提交Git。未修改生产源码、参数或其他服务；仅将本报告提交Git，并追加飞书原全流程报告后回读核对。
