# 台州路线已生成但不能执行：启动锁状态回归（2026-10-20）

用户进一步指出生成完成后仍不能执行，之前可以正常执行。本轮核对确认：这不是新路线无解，也不能仅称为“原有安全锁”。此前交付虽然恢复了路线生成，但没有恢复用户此前可执行的控制状态。

## 证据及对前两节的更正

- 更新前保存的`controller_config_v2.json`，观测时间2026-10-20 11:33:42 +08:00：motion.latched=false，phase=completed，最后一次原生动作回执safe_exit_confirmed=true。
- 当前HMI PID 1910707启动环境中`HELIOS_HMI_START_LATCHED=1`；MotionManager构造函数以该变量初始化latched。因此之前已在运行中恢复的控制，在更新重启时再次被锁定。
- 当前motion.latched=true，busy=false、capturing=false、phase=idle、last_result=null；三部件反馈正常、下电、手动、空闲。当前执行前提失败项只有“已有设备所有者、采样中或安全退出未确认”，实际对应latched条件。
- 当前用户后来生成的返回任务`6e54c93d9bbe435da2d61de36f6bdf78`为ready，已经装载HMI；没有新原生返回动作失败的回执。

因此前面将执行阻塞表述为“原有安全锁保持”容易误导：这个启动参数虽来自老进程环境，但更新前实际锁已经解除，更新重启又按旧环境重设为true，是软件重启状态保存的回归。不能把它解释为当前新发现的硬件报警。

## 为什么只复制进程环境不够

HELIOS_HMI_START_LATCHED只表示进程最初启动时的锁状态。运行中的recover_control会将self.latched改为false，但不会回写进程启动环境。后续部署若原样复制旧environ，而不读取实时锁，就可能把旧的1带入新实例，再次锁住已恢复的控制。

本仓库`tools/taizhou/README.md`已补充部署规则：重启前记录实时motion.latched和新鲜下电空闲反馈，按实时值显式设置新进程启动锁，部署后核对；反馈或退出状态不明确时停止部署。控制源码仓库既有`tools/deploy_fault_recovery.py`已经有按实时锁设置启动参数的正确示例，本次应沿用该逻辑，而非直接复制旧启动值。

## 当前恢复所需条件

恢复使用原有`/api/robot/operator`的recover_control动作，经HMI `/motion/recover`执行。该接口不做上电、运动或清警，只在现场确认及新鲜设备检查通过后解除软件锁。

源码`rokae_operator.py::_recovery_action`明确要求stop_checked=true；`motion.py::recover_control`要求confirmation、field_confirmed、faults_checked均为true，并重新读取身份、下电/手动/空闲、速度和碰撞状态。现有SDK检查不能代替现场当前报警确认，因此不能由诊断方代填。

本节写入时，已向用户请求确认“现场全身已停止、下电，当前控制器报警已核对”。尚未取得该现场确认，软件锁仍为true。未调用恢复控制、上电、清警或运动接口，未通过重启、改内存或删除保护绕开该确认。

收到确认后应直接调用原有恢复接口，按恢复产生的新控制代次重新生成原返回段，再核对power_workflow与HMI计划预检；不需再请用户重复批准相同的软件锁恢复动作。解除锁不等于已经执行返回。

本地证据摘要保存在Git忽略目录`outputs/diagnostics/taizhou-reset-20261020/execution_lock_regression.json`。本节用于更正已有飞书报告的解释并明确当前恢复边界，不宣称执行能力已经恢复。
