# 2026-10-22 先进入共用腰姿再检测：路线复核

用户明确要求先控制机器人进入共用腰姿，再重新检测并规划左臂。将上一轮“左臂插接路线尚未通过”作为整个任务停止条件不充分，本轮单独求解进入腰姿的路线。下述第一阶段记录原先的离线路线检查；用户随后现场确认并执行了收臂及部分腰部动作，最终因下电后第4轴偏移、安全退出失败锁存而停止，未到完整候选腰姿，未完成后续检测或插接。执行续报详见末节。

左臂保持现场角度直接联动腰部四轴时，生产关节线检查拒绝，途中工具与桌面模型最差重叠40.740 mm，最差配对为table/tool_disc_32；因此不能直接照抄候选腰角发送。

先收左臂、再按腰部第2、4、3、1轴顺序分段运动，已找到通过原检查的路线。目标模型腰角仍为 `[-25.507413, 15.901926, 35.563485, -122.916206]` 度。左臂避让角（弧度）为 `[-1.432826609, -2.007128640, 1.316061868, 0.497740107, 1.189695716, 0.895000094, 0.587892953]`，只是中间避让姿态，不是检测终点。

首次搜索242个候选，用时2.222秒；仅同时移动腰部四轴的初次1500候选搜索未找到路线，增加轴序搜索后找到。独立新进程使用新鲜/inspection/state核对有效DH签名、实际起点以及限位，重新检查五段路线：左臂123帧，四段腰部40/155/59/18帧，合计395帧；所有段调用cr.check_line(adjust=True)，无接触豁免、无放宽净空、无allow_start。每段模型最小间隙均报告30 mm。该数值是既有模型返回值，不是实测间隙或物理安全认证。桌面/盒位采用原保存场景，实际现场是否有新增物体仍需现场确认。

新增只读离线脚本tools/jinlong_hmi/search_waist_entry.py，读取显式状态/场景/候选路径，不调用网络或控制器SDK；保留现场限位交集和全部生产调姿检查。完成编译、CLI检查及IPC隔离环境实际重放，242个候选、2.273秒得到同一路线。运行快照和路线保留IPC诊断目录waist-only-preflight，不提交Git。

下发前现场程序helios_hmi/motion.py的send()明确要求field_confirmed=true与faults_checked=true，且原生执行器要求operator-faults-confirmed。已向用户请求事实确认：运动范围无人且无新增障碍、示教器无当前故障、现场可随时急停。不能把用户要求执行自动等同于已经看过示教器故障页，不能代填现场确认或借用昨日回执。

此路线只解决进入腰姿，不代表新腰姿视觉坐标处理、检测后的左臂接近/插接/撤离已经通过。收到现场确认后仍应核对新鲜状态及路线起点一致，按检查过的关节段进入腰姿，随后重新采集并基于正确的world/camera/base关系规划；不得直接下发旧的检测终点或伪造标定对齐状态。

证据：IPC `/home/bit-bhr/helios_hmi_sessions/diagnostics/ik160-20261022/waist-only-preflight/` 下的waist_sweep.json、stow_search.json、independent_entry_receipt.json及cli-replay/stow_search.json。没有加载执行计划或改动现场标定。

## 实机执行续报：J4下电后偏移、安全退出失败

用户答复“已现场确认，可执行”后，使用原HMI受监督接口实际执行。新增执行脚本execute_checked_waist_entry.py，每个动作核对新鲜反馈、起点和保持部件，使用原生预检、心跳租约、速度/限位保护和到位回执；任何失败均停止后续段，不自动清锁。没有直连SDK下发、修改标定或放宽生产门限。

实际经过：左臂已到避让姿态；腰部J2到目标，J4曾到约−122.916°，J3到约35.564°。J3最后一段结束时发生安全退出失败，J1尚未执行。因此不能宣称共用腰姿完整到位，也未触发新的二维码/三维任务或检测插接。

先前1%对部分轴过慢，先受控停止并根据实际停稳位置重验剩余线段；J4在5%启动时触发速度保护，测得0.118394 rad/s超过原0.100000 rad/s，safe_exit_confirmed=true。随后使用2%、每段最多8°，不提高速度门限；J4十个小段及J3前三段均回执完成。每次停止后都独立重建剩余路径检查，未盲目复用旧起点。执行脚本最终按左臂5%/最多10°、腰部2%/最多8°保存。不能用该速度下成功过的片段推断下电保持可靠。

本次失败操作ID为IYAuqJt8CodZXQdXDwt3JE2qv-M，原生日志20261022_150733_b54c7ddb6cc2.jsonl。最后progress原始控制器弧度为 `[-0.589666652,-1.293251978,0.620699654,-2.145316503]`。之后stop、wait_idle、move_reset、power_off、wait_power_off、manual_mode各调用回执ok，但116个shutdown_state采样中J4仍变化：首帧J4=−2.145921855 rad、qd4=−0.010385272 rad/s，末帧J4=−2.379061580 rad、qd4=−0.041541087 rad/s。此时power=1为off、mode=0为manual、op=0为idle，不能把枚举1误读为仍上电。

最终回执SAFE_EXIT_UNCONFIRMED，cause为safe shutdown unconfirmed，safe_exit_confirmed=false，HMI latched=true。后续新鲜模型腰角约 `[-33.785410,15.902120,35.564015,-143.274339]`°；J4偏离候选目标约−20.358°。连续5秒独立观察J4不变、body qd=0，三部件均off/manual/idle。用户现场也回复“停止了”。没有自动清锁、清警或重新上电，不能继续用原离线候选当作实际起点。

用户要求自行查看示教器后，已读取并操作IPC上的RobotAssist只读界面：最初处于与192.168.71.162断开状态，随后通过已核对的“连接”按钮接入，关闭“工作区已同步”的信息提示，查看控制器日志及状态监控。没有点击清除报警、运动、上电或修改控制柜配置。实时关节页显示J4约−143.275°，与独立SDK反馈一致；页面J2约−74.098°为控制器原始角，模型角加90°为15.902°。日志中35654安全板配置及31015伺服告警是历史项，SDK也明确标识historical_only，不能据此断言它们就是本次当前故障。控制器显示日期为2024年，不能仅用它和IPC日期不同来判断本次事件顺序，须保留操作ID与日志关联。

![RobotAssist实时关节监控：J4约−143.275度](../outputs/diagnostics/ik160-20261022/waist-only-preflight/robotassist-current-state.png)

![RobotAssist控制器错误日志：历史记录不等同于当前故障](../outputs/diagnostics/ik160-20261022/waist-only-preflight/robotassist-errors.png)

现阶段阻塞是实机在该姿态下下电后的J4保持行为与退出确认，不是扫码或IK。机械/制动/控制退出流程中的具体原因尚未确定；不能凭仿真路径无碰撞或用户确认已停止就自动解锁继续。完整任务未完成，需先核查此次偏移，再重新验证实际起点、保持及运动。

新增证据包括execution-resume、execution-waist-resume、execution-waist-2percent逐段events.jsonl，latched_final_state.json、post_fault_observations.json、post_fault_native_inspect.json及RobotAssist截图。原始会话数据不提交Git。路线搜索工具提交8f569e2，监督执行工具1a92bd1，速度/小段修正f166456，均已完成必要编译和实际使用验证。

已将实机续报追加至[issue #15](https://github.com/thedaosheng/jinlong-motor-arm-control/issues/15)，保留原终点/净空调查正文，标题突出J4下电后约20度偏移及安全退出锁存，回读正文一致。

飞书同步：本文完整追加至[既有全流程报告](https://feishu.cn/docx/I2GAdun3nofskUxAOjWcxVwjnJO)，保留原1182个顶层块；19个正文块回读一致，两张RobotAssist截图上传后回读image.token一致。交付前再次读取：busy=false、latched=true，三部件均下电/手动/空闲、反馈年龄不超过0.1秒，未继续运动。
