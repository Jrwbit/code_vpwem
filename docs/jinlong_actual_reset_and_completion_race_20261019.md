# 实际撤离复位完成及收尾状态修复（2026-10-19）

用户再次报告路线失效／已停止，随后明确错误为“请使用本机控制会话”，要求尽量直接处理并完成复位。此前日志显示页面preview/start请求返回409后发cancel，实际没有产生新原生运动。新页面代码每次带授权请求会先读取session，但这些用户失败请求之前没有对应session读取，且页面未重新请求更新脚本；结合用户错误文本，符合旧页面使用重启前令牌的情况。未绕过会话校验。

## 本次实际执行

使用当前本机会话，读取最新已生成路线`6e5efa5402c64970ac5e6cbc0b0690b2`，确认4段全部left_arm、碰撞校验通过，全身下电／手动／空闲，实时反馈新鲜，busy=false、latched=false，停稳及上电工作流预检通过。沿用现场21 mm/s速度及所有保护，没有腰部运动指令。

22:28:46提交reset start，检测侧执行ID为`15b83286b60943cb9dca03ddc0c4a90e`。外部监控持续发送400 ms控制心跳，读取进度和腰部反馈；按原计划完成：

1. 检测对位沿原路径退出，1小步。
2. 盒外10 mm退回悬停，2小步。
3. 悬停退回中转，4小步。
4. 中转收臂home，6小步。

HMI逐段done为`1/1、2/2、4/4、6/6`，phase=segment_done、segment=3、step=6，13小步全部成功。末次原生执行日志为`/home/bit-bhr/helios_hmi_sessions/motion/20261019_223113_d54ec236c7d0.jsonl`，status=completed、safe_exit_confirmed=true、native_returncode=0。最终左臂相对保存home的最大关节差约**0.03793°**；全身下电／手动／空闲，busy=false、latched=false。

进度采样中腰部变化约0.003°量级，J4起始−0.804687021 rad、结束约−0.8047313923 rad，变化约−0.00254°。这不是独立机械测量，也不证明历史保持故障永久关闭。

## 本次监控发现并修复的完成竞态

最后一次控制心跳先于下一次状态读取；任务恰好已结束、busy=false，原heartbeat返回409“执行会话已失效”。本次监控程序的通用异常处理随后发cancel，旧cancel把已经完成的trajectory覆盖成cancelled，next_segment仍为4。这是监控收尾处理的遗漏；不能将其写成实际机器人复位失败。

修复如下：

- 轨迹启动时绑定操作ID；仅当心跳属于同一已完成／到位暂停操作时返回`active=false`，不续租、不启动运动，不接受其他操作ID。
- 页面收到`active=false`即结束控制心跳。
- cancel仍始终发送真实停止请求，但保留completed／failed结果，避免抹去历史事实。
- 原撤离复位已经全部执行时，不再尝试从检测终点重放；回到普通home复位检查。

源码提交`b8e82ea431061856a4bbc9bf87642fcd83351e8a`（`/home/wjr/code/jinlong-motor-arm-control`）。本地Python148项全部通过，前端轨迹／生命周期6项通过；现场Python148项通过、3项缺Node跳过。新增用例覆盖末次心跳不复活租约、错误ID拒绝、停止保留终态且仍到硬件，以及完整返回后的普通复位检查。

现场提交`ffe2e585fad94ef7d1ac2a130c717115bb2f5bea`，备份`/home/bit-bhr/r435-release/backups/route-recovery-20261019-223320/`。在动作结束并确认下电空闲后，只重启视觉服务；HMI、安全锁、工位参数、活动标定保持。运行数据、会话和私有启动信息不提交Git。

## 最终无运动复核与页面更新

新复位预览`9c909ca2935e4a29867ff5d80b7be889`返回`completed、already_initial=true、segments=[]`，无需上电或运动，证明现场已经回到保存初始位。

直接在工控机打开新Firefox窗口`http://127.0.0.1:8765/?ui=20261019-reset-completion`。服务日志确认请求新HTML、新`rokae_trajectory.mjs?v=20261019-reset-completion`及新的本机会话；不再依赖用户在旧标签页手动刷新。没有额外运动请求。

本轮完成的是检测终点撤离返回及收尾状态修复。此前约3 mm左右定位偏差、硬件间歇报警根因与电测闭环仍未关闭。
