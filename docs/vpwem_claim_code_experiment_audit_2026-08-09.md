# VPWEM / PER 论文主张—代码—实验完整追溯审计

审计日期：2026-08-09  
代码目录：`/data/wjr/vpwem/code_vpwem`  
原论文：`literature/vpwem/vpwem.pdf`，arXiv:2603.04910v1

## 一、结论

1. 原 VPWEM 的核心结构不是虚构的。固定长度 Q-Former memory tokens、逐层 summary cache、observation cache、近期观测和长期记忆共同条件化 diffusion decoder、以及联合行为克隆训练均有实际代码调用链。
2. 原论文的“完整轨迹历史”表述比训练代码更强。训练 sampler 会分段随机采样，并把 memory 截断到 `maximum_memory_length`，默认值为 16；因此不能说每个训练样本实际输入了任意长度的完整历史。
3. 发布版 `HEAD` 与论文 Table I 存在关键配置差异。论文写 8-layer diffusion 和 2-layer compressor；发布版 pipeline 硬编码 4-layer diffusion，多份 YAML 使用 1-layer Q-Former。当前工作区已修复部分配置，但已保存的多数 Robomimic checkpoint 仍是 4-layer diffusion。
4. 当前仓库不能独立复核原论文全部主结果。MIKASA 和 MoMaRT 没有本地正式 result 文件；MaIL-VPWEM 只有网络类，没有对应训练/评估 pipeline、正式配置和结果产物。
5. 本地 Robomimic 标准任务有三 seed 结果，但使用的是 4-layer diffusion，不能称为严格论文配置复现。LongSquare 的 4-layer 三 seed结果为 `1.11% +/- 1.92%`；唯一 8-layer seed 100 也是 0%，没有复现论文约 70% 的表现。
6. 新增 PER 不是只有代码。Transport/mh 已有三训练 seed、每 seed 300 episodes 的 PER-delta 与 FIFO 对照。PER 为 42.56%，FIFO 为 39.67%，平均差为 `+2.89 pp`；但三 seed 差值的 95% t 区间约为 `[-6.34, 12.12] pp`，目前不能声称稳定显著提升。
7. 本机 `vpwem5090` 环境全量单元测试为 `21 passed, 0 failed`。这证明测试覆盖的接口行为成立，不证明论文性能结论。

## 二、审计判定标准

| 判定 | 含义 |
| --- | --- |
| 一致 | 论文主张、实际调用链、配置和结果证据一致 |
| 部分一致 | 机制存在，但范围、配置或实验支撑弱于论文措辞 |
| 配置漂移 | 论文设置、发布版配置、当前配置或 checkpoint 结构不一致 |
| 无本地证据 | 论文报告了结果，但当前工作区没有可追溯产物 |
| 不一致 | 当前代码或产物直接反驳该表述 |
| 待验证 | 新方法已有实现，但尚缺足够实验 |

结果证据最低要求：配置快照、checkpoint、训练 seed、评估 seed、episode 数和原始 result 文件必须同时可追溯。

## 三、原 VPWEM 主张总表

| ID | 论文主张 | 代码与调用链 | 实验证据 | 判定 |
| --- | --- | --- | --- | --- |
| V1 | 最近 L 个观测构成 working memory | `LongMemDDPM.sample` 保留最近 `obs_steps`；`LongMemChiTransformerPTP.forward` 接收 condition | Robomimic 配置通常 `obs_steps = 2` | 一致 |
| V2 | 越窗观测加入时间位置编码 | `QFormerMemory.forward/inference` 在压缩前加入 `mem_pos_emb(ep_step)` | 保存配置使用 `sin_pos` | 一致 |
| V3 | compressor 使用 trainable queries | `QFormer.query_tokens` 为 `nn.Parameter` | checkpoint 含 query token 参数 | 一致 |
| V4 | 每层先对历史 summary self-attention，再对观测 cross-attention | `QFormerLayer.self_attn` 查询 `query_cache`，随后 `cross_attn` 查询 observation features | 单元测试覆盖输出形状，未覆盖注意力语义 | 一致 |
| V5 | 输出固定 M 个 episodic memory tokens | `QFormer` 的 `n_queries = mem_compress_length`，输出形状为 `[B, M, D]` | checkpoint 的 query token 形状为 `[1, 2, 256]` | 一致 |
| V6 | 每层维护 summary cache，另维护 observation cache | `QFormerLayer.query_cache` 与 `QFormerMemory.condition_cache` 分开维护 | 推理 pipeline 每回合 reset | 一致 |
| V7 | 缓存大小固定，逐步递归压缩 | 两类 cache 都受 `cache_max_length` 约束；主设置为 8 | 结构上有界；无独立长 horizon 资源曲线产物 | 部分一致 |
| V8 | memory 与近期观测共同条件化动作扩散模型 | `LongMemChiTransformerPTP` 拼接 noise、memory、observation embeddings，再作为 Transformer decoder memory | 本地 checkpoint 和 rollout 均经过该调用链 | 一致 |
| V9 | compressor 与 policy 端到端用行为克隆损失联合训练 | `LongMemDDPM.model` 包含 memory；optimizer 接收全部 model parameters；loss 反传 diffusion MSE | 训练日志存在 | 一致 |
| V10 | 历史 cache detach，避免跨时间反传 | observation 和 query cache 写入时调用 `detach()` | 属于代码事实 | 一致 |
| V11 | 训练样本包含从 episode 开始的完整轨迹 | sampler 从 episode 开始分段采样，但超过 `maximum_memory_length` 后只保留最近部分，默认上限 16 | 无证明上限不会触发；MIKASA 最长设置也可能超过 16 个采样段 | 不一致 |
| V12 | memory 动态概括 full trajectory history | 在线递归 summary 理论上可传递更早信息；训练输入存在截断，且没有 state probe 验证信息保真 | 成功率不能单独证明完整历史被保留 | 部分一致 |
| V13 | 过滤无关历史并缓解 nuisance correlation | Q-Former 可学习压缩，但没有显式去相关目标 | 原论文没有因果干预或泄漏 probe；当前只有有限诊断 | 证据不足 |
| V14 | 近常数的逐步计算与存储 | 主配置的 observation 和 summary cache 都有上界；输出 token 固定 | Table II 数字仅见论文，当前没有对应原始 benchmark 文件 | 部分一致 |
| V15 | 方法实例化在 DP-VPWEM | LongMem ChiTransformer、DDPM、训练和评估 pipeline 完整 | Robomimic checkpoint 与结果存在 | 一致 |
| V16 | 方法实例化在 MaIL-VPWEM | 有 `LongMemMaIL` 网络类，`LongMemDDPM` 能识别该类型 | 未找到 long-memory MaIL pipeline、YAML、checkpoint 或 result | 无法复现 |
| V17 | 三个 benchmark 全部使用 3 个随机 seed | Robomimic 有多组三 seed；MIKASA 和 MoMaRT 当前无本地正式结果文件 | 工作区无法核验全矩阵 | 无本地证据 |
| V18 | memory 模块约 2.24M 参数 | 2-layer、M=2、D=256 的 QFormerMemory 实测为 2,239,488 参数 | 与论文约 2.24M 一致 | 一致 |
| V19 | MIKASA 平均超过 VLA 基线 20% 以上 | DP-VPWEM pipeline 存在 | 论文图报告 91% 和 82%；本地无正式 result 文件 | 无本地证据 |
| V20 | MoMaRT 平均提升约 5%，DP 和 MaIL 均提升 | DP pipeline 存在；MaIL 完整运行链缺失 | 本地无 MoMaRT result 文件 | 无本地证据 |
| V21 | Robomimic 与基线相当 | 本地标准四任务有三 seed 结果 | checkpoint 是 4-layer diffusion，并非论文 8-layer；LongSquare 严重失败 | 部分一致 |
| V22 | LongSquare 主设置约 70%，L1S1 为 65% | 代码能配置不同 token 数 | 本地 4-layer 三 seed 为 1.11%；唯一 8-layer seed 为 0%；无论文消融产物 | 当前复现不一致 |
| V23 | cache size、层数、token 数和管理方法消融趋势成立 | FIFO、Random、K-Means、AdjSim 均有实现 | 当前没有对应 Figure 7 的原始结果文件 | 无本地证据 |

## 四、原 VPWEM 公式—代码映射

论文 Eqs. (1)-(5) 的计算意图可以写成：

```text
q_1,t = learned_query
x_1,n = q_n,t + self_attention(q_n,t, summary_cache_n)
x_2,n = x_1,n + cross_attention(x_1,n, observation_cache)
q_(n+1),t = x_2,n + ffn(x_2,n)
e_t = output_projection(q_N,t)
```

逐项映射：

| 公式项 | 代码位置 | 审计 |
| --- | --- | --- |
| learned query | `cleandiffuser/nn_memory/qformer_memory.py:167-174` | 一致 |
| summary self-attention | 同文件 `81-125` | 代码采用 pre-norm 和 residual，语义一致 |
| observation cross-attention | 同文件 `126-136` | 一致 |
| FFN | 同文件 `137-141` | 一致 |
| final projection | 同文件 `225-226` | 代码先 norm 再 linear；论文简写为 MLP，属于实现细节 |
| per-layer summary cache | 同文件 `81-108` | 一致，写入 cache 前 detach |
| observation cache | 同文件 `299-323` | 一致，写入 cache 前 detach |

论文 diffusion 目标 Eq. (6) 对应：

```text
L_action = mean((eps_hat(x_t, t, working_memory, episodic_memory) - eps)^2)
```

代码位置为 `cleandiffuser/diffusion/longmem_ddpm.py:210-242`，并由 `pipelines/longmemdpptp_*` 调用 `agent.update(...)`。在 `pred_reg_coeff = 0` 且新增 PER 辅助系数为 0 时，与论文目标一致。

## 五、论文超参数—代码—checkpoint 对照

| 设置 | 论文 Table I | 发布版 HEAD | 当前工作区 | 本地 checkpoint |
| --- | --- | --- | --- | --- |
| diffusion layers | 8 | pipeline 硬编码 4 | pipeline 改为读取 8 | 标准 Robomimic 多数为 4；仅 LongSquare 有一个 8-layer seed |
| compressor layers | 2 | 多份 YAML 为 1 | MIKASA 主任务已改 2；MoMaRT/Robomimic YAML 多数仍为 1 | 用于本地复现汇总的 checkpoint 多数为 2 |
| short tokens L | 2 | 多数为 2 | 主任务为 2 | 保存配置为 2 |
| long tokens M | 2 | 部分 MIKASA YAML 为 4 | MIKASA 主任务已改 2；部分扩展仍为 4 | 复现 checkpoint 多数为 2 |
| cache size S | 8 | 8 | 8 | 8 |
| main cache method | FIFO | MoMaRT 与部分任务为 AdjSim | MIKASA 主任务为 FIFO；MoMaRT 仍为 AdjSim | 标准 Robomimic 复现多为 FIFO |
| subsample ratio | MIKASA 5，其余 20 | RememberColor3 为 20 | MIKASA 两个主任务已改 5 | Robomimic 为 20 |
| batch size | 64 | ShellGame 和非 embedding MoMaRT 为 16 | 仍有 16 与 64 混用 | Robomimic embedding 运行多为 64 |
| gradient steps | MIKASA 600k，其余 1M | MIKASA 主任务为 200k | MIKASA 主任务已改 600k | Robomimic 为 1M |

结论：论文超参数不能仅靠仓库默认 YAML 复现。必须提供冻结后的论文配置文件，并用 checkpoint state dict 核对实际层数。

## 六、本地 Robomimic 结果审计

以下结果都来自原始 `.npy` 文件。为保持 compressor 层数一致，Square/ph 和 Transport/ph 的 seed 100 选用明确标记为 Q-Former 2-layer 的运行。所有四个标准任务的 checkpoint 仍是 4-layer diffusion，因此标记为“本地变体”，不是严格论文复现。

| 任务 | 三 seed success | checkpoint 结构 | 证据结论 |
| --- | --- | --- | --- |
| Square/ph | `94.44% +/- 1.92%` | diffusion 4，Q-Former 2 | 有效本地变体结果 |
| Square/mh | `76.67% +/- 3.33%` | diffusion 4，Q-Former 2 | 有效本地变体结果 |
| Transport/ph | `90.00% +/- 5.77%` | diffusion 4，Q-Former 2 | 有效本地变体结果；旧汇总误混入 Q-Former 1-layer seed 100 |
| Transport/mh | `36.67% +/- 6.67%` | diffusion 4，Q-Former 2 | 有效本地变体结果 |
| 四个标准任务平均 | `74.44%` | 非论文 8-layer 设置 | 只能作本地参考 |
| LongSquare/ph | `1.11% +/- 1.92%` | diffusion 4，Q-Former 2 | 复现失败 |
| LongSquare/ph 8-layer | seed 100 为 `0%` | diffusion 8，Q-Former 2 | 只有一个 seed，仍失败 |

主要文件：

- `outputs/ours/robomimic_emb/square_ph/`
- `outputs/ours/robomimic_emb/square_mh/vpwem_reproduction/`
- `outputs/ours/robomimic_emb/transport_ph/`
- `outputs/ours/robomimic_emb/transport_mh/vpwem_reproduction/`
- `outputs/ours/robomimic_emb/longsquare_ph/`

## 七、数据采样实现风险

`LongMemSequenceSampler` 的实际过程为：

```text
history before current sequence
    -> split into segments of subsample_ratio frames
    -> randomly select one frame per segment
    -> if selected count > maximum_memory_length:
           retain only the latest maximum_memory_length frames
```

对应代码为 `cleandiffuser/dataset/dataset_utils.py:235-278`。

必须处理的两个问题：

1. `memory_type` 被保存为成员变量，但当前采样逻辑没有根据 `random_from_full` 或 `sliding_window` 分支。配置名称不能当作已经实现的 baseline。
2. episode 起点计算使用 `np.searchsorted(episode_ends, buffer_start_idx) - 1`。当 `buffer_start_idx` 恰好等于新 episode 起点时，可能把该样本误归到前一 episode，形成首样本跨 episode memory 污染。需要增加边界单元测试并修正后再重跑关键实验。

## 八、PER 主张—代码—实验总表

| ID | PER 主张 | 代码证据 | 实验证据 | 判定 |
| --- | --- | --- | --- | --- |
| P1 | 固定 K_bank 的事件库 | `PERMemory.bank` 固定为 `bank_size` | 单元测试覆盖上界 | 一致 |
| P2 | feature delta 分数 | `1 - cosine(z_t, z_(t-1))` | Transport/mh 正式对照使用 delta | 一致 |
| P3 | predictive error 分数 | predictor 和 MSE 已实现 | 正式运行系数为 0，未验证 | 待验证 |
| P4 | hybrid 分数 | 离线归一化后平均；在线直接相加 | 无正式结果 | 训练/推理不一致 |
| P5 | skip | 阈值、最大写入间隔和空库共同决定写入 | 单元测试覆盖静态历史 | 一致 |
| P6 | similar-slot merge | 相似时直接覆盖非锚点槽 | 没有加权融合 | 应改称 update |
| P7 | replace | 满库时替换衰减分数最低的非锚点槽 | 单元测试覆盖容量，不覆盖最优性 | 机制一致，效果待验证 |
| P8 | evidence + belief tokens | 代码支持 E 个 evidence 和 M-E 个 belief tokens | Transport/mh 正式运行 `evidence_length = 0` | 实现存在，正式结果未验证 evidence |
| P9 | predictor 辅助损失 | `memory_aux_coeff * L_pred` 已接入 | 正式运行 `memory_aux_coeff = 0` | 未验证 |
| P10 | 反事实 margin loss | batch roll 错误记忆与 hinge loss 已接入 | 正式运行 `counterfactual_memory_coeff = 0` | 未验证 |
| P11 | zero、shuffle、reverse 干预 | `apply_memory_intervention` 已实现 | 单元测试通过；只有单 seed 30 episodes 诊断 | 工具一致，正式证据不足 |
| P12 | novelty 独立项 | 无 | 无 | 不可写 |
| P13 | action influence 写入项 | 无 | 无 | 不可写 |
| P14 | state probe、write-rate loss、trajectory adversary | 无 | 无 | 不可写 |
| P15 | PER 普遍优于 FIFO | 单任务调用链完整 | Transport/mh 平均 `+2.89 pp`，seed 100 为负，三 seed CI 跨 0 | 不可泛化 |

### PER 正式对照的实际设置

```text
task = Robomimic Transport/mh
diffusion_layers = 4
qformer_layers = 2
memory_tokens = 2
event_bank = 8
event_score = delta
evidence_length = 0
memory_aux_coeff = 0
counterfactual_memory_coeff = 0
finetune_steps_used_for_eval = 10000
evaluation_episodes_per_train_seed = 300
evaluation_seed = 20260806
```

| 训练 seed | FIFO | PER-delta | 差值 |
| --- | --- | --- | --- |
| 100 | 40.67% | 39.33% | -1.33 pp |
| 200 | 37.67% | 42.00% | +4.33 pp |
| 300 | 40.67% | 46.33% | +5.67 pp |
| 均值 | 39.67% | 42.56% | +2.89 pp |

三 seed 差值的样本标准差为 `3.72 pp`；以训练 seed 为统计单位，95% t 区间约为 `[-6.34, 12.12] pp`。当前只能写“观察到正向均值差”，不能写“显著提升”。

证据路径：`artifacts/per_vpwem/formal_eval/train{100,200,300}_{fifo,per}/`。

### 记忆干预诊断

seed 200 的同一 PER checkpoint 有 30-episode 诊断：正常 memory 为 46.67%，zero 为 40.00%，shuffle_batch 为 10.00%。这是有价值的方向性证据，但只有一个训练 seed、30 episodes，且没有纳入三 seed 300-episode 正式矩阵，不能作为最终因果结论。

## 九、当前允许和禁止的论文表述

### 可以写

- VPWEM 使用有界 observation cache 和 per-layer summary cache，将历史压缩为固定数量的 memory tokens。
- memory tokens 与近期观测共同条件化 diffusion action model。
- 2-layer、M=2、D=256 的 memory module 约为 2.24M 参数。
- 本地 4-layer Robomimic 变体取得本报告列出的三 seed 结果。
- PER-delta 在 Transport/mh 当前设置下观察到 `+2.89 pp` 的三 seed 平均差，但不具统计确定性。
- zero 和 shuffled memory 的单 seed诊断表明策略可能使用样本匹配的 memory，仍需正式验证。

### 不可以写

- 当前训练代码对任意长度的完整历史进行无损或完整压缩。
- 当前发布配置可以直接复现论文全部数字。
- 当前仓库已经复现 MIKASA 91%、RememberColor3 82% 或 MoMaRT 平均提升 5%。
- 当前仓库已有可复现的 MaIL-VPWEM 完整实验链。
- LongSquare 已被本地复现。
- PER 已显著或普遍超过 FIFO、HAMLET、FrameSamp 或 recurrent baselines。
- predictive、hybrid、反事实训练、state probe、novelty 或 action influence 已获得正式实验验证。

## 十、投稿前 P0 修复清单

1. 冻结一个论文配置目录，明确 8-layer diffusion、2-layer Q-Former、M=2、S=8、FIFO、batch、steps 和 subsample ratio；禁止依赖临时 Hydra 覆盖。
2. 修复 sampler 的 episode 边界计算，并增加“首样本不得读取上一 episode”的测试。
3. 明确“完整历史”的定义。若仍保留 `maximum_memory_length = 16`，论文改为“有限分段历史”；若坚持 full history，修改训练数据链并重跑。
4. 用严格论文配置重跑 Robomimic 三 seed，尤其是 LongSquare；旧 4-layer结果单独标为开发结果。
5. 补齐 MIKASA 和 MoMaRT 的配置、checkpoint、三 seed result 与原始日志。
6. 若继续声称 MaIL-VPWEM，补齐可执行 pipeline、YAML、checkpoint 和评估；否则从新稿贡献中删除。
7. 把 PER/FIFO 正式对照扩展到至少两个 memory-intensive 任务，并使用独立评估 seed。
8. 将 none、zero、shuffle_batch 纳入同一 checkpoint、同一 episode 集合的三 seed 300-episode矩阵。
9. predictive 实验必须启用 `memory_aux_coeff`；hybrid 必须先统一训练和推理公式。
10. 为每张论文表生成 manifest，记录代码 commit、dirty diff hash、环境、配置、checkpoint SHA256、result 文件和统计脚本。

## 十一、测试状态

```bash
cd /data/wjr/vpwem/code_vpwem
PYTHONPATH=. /home/wjr/miniconda3/envs/vpwem5090/bin/pytest -q tests
```

结果：`21 passed, 0 failed, 6 warnings`。

测试记录见 `docs/vpwem_local_test_report_2026-08-09.md`。
