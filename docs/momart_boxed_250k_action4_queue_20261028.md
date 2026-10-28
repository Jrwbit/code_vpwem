# MoMaRT 250k 带框 No-History：预测9步、执行4步，seed100/200/300排队评测

日期：2026-10-28，Asia/Shanghai。用户要求继续排队评测250k带框No-History，保持预测9步，每次只执行4步，评测seed为100、200、300。

## 已提交状态

后台协调器PID **3051605**，提交时间 **2026-10-28 21:23:28+08:00**。已回读进程、queue_status.json、summary.json与supervisor.log；当前状态 **waiting_predecessor**。三个seed均为0/10任务完成，尚无本轮成功率。

先等待既有250k八步队列结束，包括seed100三次重复和随后seed200/300。用该队列已有的 `.supervisor.lock` 等待其释放，再提交每卡GPU gate；空闲显存至少20000MiB、利用率不高于20%、连续3次检查满足后启动。没有停止、替换或抢占已有训练/评测。前序队列若失败退出也会释放锁，新任务随后独立排队，不复用前序失败结果。

| 评测seed | 物理GPU | 任务数 | 回合数 |
|---|---|---|---|
| 100 | 3 | 10配置，各30回合 | 300 |
| 200 | 4 | 10配置，各30回合 | 300 |
| 300 | 6 | 10配置，各30回合 | 300 |

总共30个任务配置运行、900回合。每卡同时只运行一个任务，10个环境并行；同seed内十个配置依次执行。此处100/200/300是评测seed，模型训练seed仍为100。

## 固定协议

- 方法：仅在主RGB画框的No-History，不是坐标/ROI token版。
- 权重：已有model_250000.pt，不重新训练；最近观测2帧。
- 模型预测长度horizon=9；实际执行动作索引1:5，共4步。
- 原训练配置action_steps=8保留；评测通过既有 `--action-steps 4` 入口覆盖执行步数，输出记录training_action_steps=8和execution_action_steps=4。
- DDPM50步、EMA、每回合环境步数上限1500；不录制视频。
- 使用冻结combined_blend在线框和与既有八步实验相同的计分方式。
- 为保证四步/八步比较只主动改变执行长度，检测器继续使用既有八步队列版本。没有混入新发现的reset补帧修复，也没有改变裁剪或坐标语义；`tracker_reset_fix=false`记录在manifest。修复版应另做对照，不能混入本组归因。

执行块缩短会增加观测、tracker更新及成功标志检查的频率；即便其他设置相同，也不能把未来的全部分数变化单独归因于动作预测质量。三个独立评测seed比单seed更有参考性，但不等于三个独立训练seed。

## 输入与代码验证

250k checkpoint SHA256：`00ccf12e6f4f9dec1965c6139da9da385cf3792f0e2a23ca9b2945cfafc5b203`。

训练配置SHA256：`ffb26097aa459dbbbc91514313c4563cdd1cd9ccb9b3db41c69f78650153a4bb`。

normalizers.npz SHA256：`6e21389e36c17b550f525842e5437fb5a507ad9d82511b53000e5cb84117ef9b`。

detector bundle SHA256：`af62860467a1c95d2e60a9df5d89ee1b9001116cb45c1aa7a285c1810f6f7c07`。

准备阶段在CPU严格加载实际250k的model和model_ema，使用horizon9、obs_steps2、执行步数4的agent配置，通过；评测CLI入口与十套输入目录检查通过。三个关键检测器源文件与正在运行的八步工作树SHA256相同。隔离评测工作树基于已有action4入口提交 `bada2c03478689be3854c3efa40afd74314f6640`。

本地队列与结果校验实现提交 `b939cb4`。四项测试通过，覆盖正确结果、拒绝错误checkpoint/seed/执行步长/预测长度、拒绝不完整或不一致回合，以及部分结果不伪报最终成功率。新脚本语法检查和git diff --check通过。

本地和服务器队列脚本SHA256均为 `db9a8feaff1db6cca4a287d1962b4d7653be276cc28f1b7f7efd4807ae5dc1a4`，已确认部署内容一致。

## 运行入口与结果位置

- 本地源码：`experiments/momart_boxed_250k_action4_20261028/run.py`。
- 本地测试：`tests/test_momart_250k_action4_queue.py`。
- BHR协调器：`/home/jrw/vpwem_jobs/momart_boxed_250k_action4_20261028.py`。
- 隔离策略工作树：`/home/jrw/code_vpwem_boxed_250k_action4_20261028`。
- 训练目录：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100`。
- 输出根目录：上述训练目录下 `eval_online_combined_blend_250k_action4_seeds100_200_300_20261028`。
- 根目录manifest.json保存输入哈希及协议；queue_status.json保存前序等待/调度状态；summary.json保存逐seed进度；supervisor.pid和supervisor.log保存进程及日志；结束生成completion.json。
- 各seed的结果：`seed100`、`seed200`、`seed300` 下 `step_250000/<task>/result.npy`。各seed还有status.json、queue.log和logs/。结果验证后才计入汇总，不以文件存在直接视为有效。

已将此组三seed四步实验加入既有2026-10-29 08:00汇总任务的读取范围，没有新增定时任务。届时若仍排队或未完成，按真实状态报告，不把部分结果当完整300回合成绩。
