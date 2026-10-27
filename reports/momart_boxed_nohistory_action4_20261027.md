# MoMaRT 带框 No-History 150k：GPU 3/5/6 四步动作评测

日期：2026-10-27（Asia/Shanghai）。用户授权使用物理 GPU 3、5、6 单独评测原 150k 带框 No-History，将每次执行动作数从 8 改为 4。

## 启动状态

后台队列于 **2026-10-27 18:28 CST** 启动，supervisor PID 为 **1513821**。三张卡启动前均为空闲，各占用约 3 MiB；主机可用内存约 446 GiB。启动后，三张卡均已加载策略并进入 iGibson 环境初始化。

**18:33 CST 首批验证**：GPU 3、5、6 的任务日志均已输出 `batch=1 steps=200/1500`，执行参数均为 action_steps=4、horizon=9、obs_steps=2。三卡显存约 18.8–19.3 GiB，未发现 Traceback、RuntimeError 或 CUDA OOM。此时完整任务为 0/10，确认的是实际 rollout 已开始，不是评测已经完成。

当前是运行记录，尚无完整成功率；不能把旧的 59.33% 或未完成任务的平均值写成本次结果。完整实验为 10 个任务配置、每项 30 episodes，共 300 episodes。队列持续在后台运行，结束时自动保存 summary.json 和 completion.json。

## 固定协议

| 项目 | 本次设置 |
|---|---|
| 方法和 checkpoint | boxed_no_history，原 model_150000.pt |
| 唯一主动修改的策略参数 | execution action steps：8→4 |
| 模型输入/输出尺寸 | obs_steps=2，horizon=9，不变 |
| 动作截取 | 从 obs_steps−1 开始执行 4 个动作，即索引 1:5 |
| 扩散推理 | DDPM 50 次采样，temperature=1，EMA |
| 检测器 | 原 combined_blend 在线检测/跟踪 bundle |
| 评测 seed | 20260917，每个任务独立进程 |
| 评测量 | 每任务 30 episodes，10 个并行环境，3 批 |
| 环境步数上限 | 1500，沿用原 wrapper 逐动作截断 |
| 原 8 步参考 | 178/300，59.33% |
| 训练和原始产物 | 不重训；原 train_config、权重、normalizer 和旧评测结果不改动 |

原训练配置的 action_steps 仍为 8；新增 CLI override 只在本次 evaluator 进程内改执行步长。结果明确记录 training_action_steps=8、execution_action_steps=4、horizon=9、obs_steps=2，防止复用错误结果。

## 三卡分配

| 物理 GPU | 顺序执行的任务配置 |
|---|---|
| 3 | unload / suboptimal；cleanup sink / expert；setup dishwasher / suboptimal；setup dresser / expert |
| 5 | unload / expert；cleanup dishwasher / suboptimal；setup dishwasher / expert |
| 6 | cleanup sink / suboptimal；cleanup dishwasher / expert；setup dresser / suboptimal |

每张卡只有一个评测进程队列，每任务保持 10 环境；任务之间互不重复。沿用已有 GPU queue lock。任务失败会明确记录 returncode 和日志，其他任务继续；不将失败视为完成。

## 结果读取与解释

每个任务保存逐 episode 的成功标志、奖励、步数、策略推理时间，以及有效框统计；队列另存每任务 wall time。结果先写临时文件，再原子替换 result.npy，避免 supervisor 读到未写完的文件。

汇总程序每 30 秒检查结果。只有 10/10 项全部通过 step、seed、动作步长、horizon、obs_steps 和 30 episodes 校验，才计算总成功率和相对旧 8 步的百分点变化；未完成时只报告进度和已完成任务。

这项实验改变闭环观测频率，因此在线 tracker 的更新频率和 max_age=3 对应的环境时间跨度也随之改变。分数不能单独归因于动作截断。

另已核实原 wrapper 仅向 evaluator 回传每个动作块最后两个 info，计分代码累积其中的 task success。此次保持原计分实现；四步时检查时刻更密，若成功标志不是持续保持的，分数可能包含更密集地捕获短暂成功的影响。同 seed 也不表示不同推理次数下扩散噪声严格逐 episode 配对。原八步外层 elapsed 可能写到 1504，但 wrapper 实际在 1500 步截断，未因此给旧基线额外执行机会。

## 产物与复核入口

- 服务器：BHR，SSH alias bitbot421。
- 隔离代码：`/home/jrw/code_vpwem_boxed_action4_20261027`。
- 原始代码 commit：`dbf6779ac90c7a19b7543a1eb3d5509ef2253f13`；本次评测 commit：`bada2c0`。
- 输出根目录：`/home/jrw/vpwem_outputs/momart_boxed_no_history_150k_action4_20261027_gpu356`。
- 输出下的 manifest.json 记录权重/检测器/配置 SHA256、任务分卡与 supervisor PID。
- 输出下的 gpu3_status.json、gpu5_status.json、gpu6_status.json 记录任务、进程与命令。
- 输出下的 step_150000/<task>/result.npy 保存逐任务原始结果；logs/ 保存完整日志。
- summary.json 保存进度和最终对照；completion.json 保存三条队列退出码。
- checkpoint SHA256：`b33b859502decba0c734734bedf91df2b3ba28a35c17b7eeff7eae25989a59ab`。
- detector bundle SHA256：`af62860467a1c95d2e60a9df5d89ee1b9001116cb45c1aa7a285c1810f6f7c07`。
- 本地可复现脚本：`experiments/momart_boxed_action4_20261027/run.py`。
- 前置诊断：[150k 带框 No-History 提分诊断](https://feishu.cn/docx/PQMmdV3zjoSgpBxU3wFcyG8vnIe)。

验证：独立审查任务划分与口径；脚本语法检查、git diff --check 通过；任务分配无重复、有效 action4 结果通过、旧 action8 结果被拒绝的检查通过；prepare 阶段核验 checkpoint 与 bundle SHA256；隔离 evaluator 已严格加载原 online/EMA 权重。完整在线评测仍在运行。
