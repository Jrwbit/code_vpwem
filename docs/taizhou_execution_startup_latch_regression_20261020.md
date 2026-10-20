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


## 15:02 用户确认后的恢复结果

用户明确确认“已确认，恢复控制”，授权按原接口解除软件锁。本节更新上文等待确认的状态，不撤销前述对重启回归的更正。

已调用原有recover_control接口，回执为`ok=true、status=control_recovered、controller_writes=false`。随后读取确认`latched=false`、`busy=false`、`power_workflow.preconditions_met=true`。

恢复会按原保护流程使旧任务失效，已沿相同来源重新生成返回任务`c4aafd6f59d045c0ad359a90beae9b80`，来源仍是原完整路线`97890770d64940489b9fab7dcc7dcbac`。当前ready，四段13步，已经装载HMI。没有重新搜索任意收臂路径。

对第一段执行了一次原HMI `/plan/preview`，检查通过：segment=0、左臂、关节运动、速度21 mm/s、最大关节变化约1.49°。没有调用`/plan/start`，没有上电、清警或真实运动。临时预检令牌未保存、未消耗，实际执行仍由页面启动时重新取得。

现场截图中HMI已显示“单次下发已开启”，不再显示等待故障恢复。当前页面滚动在恢复区，未据截图声称已看见下方返回按钮；实际可执行条件依据实时接口及第一段预检通过。现场核对路线勾选后可在返回区域点击执行。

![恢复控制后现场状态](../outputs/diagnostics/taizhou-reset-20261020/execution_unblocked.png)

恢复回执与非令牌预检证据见忽略目录`outputs/diagnostics/taizhou-reset-20261020/control_recovery.json`及`execution_unblocked.json`；现场备份目录下保留相同回执。当前执行锁阻塞已解除，本次未执行返回。
