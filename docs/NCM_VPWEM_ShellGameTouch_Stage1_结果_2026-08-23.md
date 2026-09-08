# NCM-VPWEM：ShellGameTouch Stage-1 实验结果

日期：2026-08-23

## 结论

本轮预注册判定为 **NO-GO**：matched counterfactual supervision 在离线 diffusion loss 上建立了最强的 correct/wrong 可分性，但没有在因果 rollout 中优于 VPWEM、Random-CF 或 Unmatched-Branch-CF。因此目前不应继续加入 architectural necessity gate、compact belief、slots 或 revision；先保持 Stage-1 目标可归因，并解决新协议下绝对策略能力不足的问题。

Matched-CF 的关键结果是：

- correct rollout：Success 13%，BranchAcc 30%。
- matched-wrong rollout：Success 5%，BranchAcc 10%。
- correct−matched-wrong：Success +8pp，95% CI [0, 16]pp；BranchAcc +20pp，95% CI [10, 30]pp。
- matched-wrong 把动作选择吸向 donor 杯子的 uplift 为 +27pp，95% CI [17, 37]pp；第一运动方向 uplift 为 +50pp。
- 离线 ΔCF 为 0.291，是四组最高。
- 但相对 VPWEM，Matched-CF 的 correct Success 为 −4pp，95% CI [−11, 3]pp；BranchAcc 为 −9pp，95% CI [−17, −1]pp，精确 McNemar p=0.049。
- Matched-CF 的 +8pp success intervention gap 未达到预注册的 15pp；它的 intervention effect 也不是特有的，三个控制的 success gap 为 12–13pp、BranchAcc gap 为 21–30pp。

这意味着“错误历史会因果性地改变动作”成立，但“matched supervision 比随机或未匹配跨分支监督更有效”不成立。

## 为什么重做数据协议

原始 ShellGameTouch demonstrations 不满足完整 current-input alias：

- 单帧 `obs5` 的 branch probe 接近 chance，但训练使用两帧 current。
- `[obs4, obs5]` 仍含红球线索，leave-one-collector-batch-out 1NN BranchAcc 为 91.27%。
- `[obs5, obs6]` 已含首个 branch-specific action 的后果，BranchAcc 为 92.67%。
- 原评测一次执行 8-step action chunk，会在首个 chunk 中跨过 reveal→decision，策略可直接从初始可见球规划动作，不需要 memory。

因此本轮收集了新的物理 simulator rollout：t=0…5 执行统一 neutral hold `[0,0,0,0,0,0,0,+1]`，t=6 起使用官方 MIKASA state oracle 闭环控制。`+1` 保持 gripper 张开；全零不是 gripper hold。策略评测使用 `inference_action_steps=1`，每一步仍调用 policy 更新 memory，只在 t≤5 覆盖送入环境的动作。

新数据共 1000 episodes，其中 996 条物理成功；branch 数为 `[337, 360, 303]`。完整决策 current `[obs5, obs6_hold]` 的 leave-one-batch-out 1NN BranchAcc 为 33.03%，低于 majority 36.14%；首个 target action 的 branch probe 为 98.69%。因此它通过了“current 近不可辨、target action 强分支化”的监督可行性门槛。

## 固定实验设计

四组为 VPWEM、Random-CF、Unmatched-Branch-CF、Matched-CF。所有组固定：

- 同一个 600k VPWEM 初始化权重，SHA256 `d485e9ca3453645bdb1eaeddb8131e8134dbcf2e134530aa10dff08b1347046f`。
- 同一 Q-Former、decoder、2 个 memory tokens、训练 batch、学习率 `1e-5` 与 nominal 10k 预算。
- 训练循环按现有 inclusive step 语义实际执行 10,001 次 update；四组完全相同。
- 三个 CF 组均使用系数 0.5、margin 0.1、概率 1.0；只改变 negative 来源。
- pair-aware batch 固定 pair 成员，但每 epoch 打乱行顺序；Random-CF 仍为随机 `roll`，显式 matched/unmatched partner 不依赖 batch 顺序。
- correct/wrong 共用 diffusion `t`、noise、`x_t`、condition、long-memory dropout availability 与 decoder RNG；Q-Former 对 batch 只编码一次，再按显式 partner index gather。
- 评测为相同的 100 episodes、10 envs，环境 seed 与 diffusion seed 都从 `20260807` 固定；GT branch 分布 `[40, 37, 23]`。
- 每组评测 correct、zero、random、matched-wrong 四种 intervention；commit 阈值 0.12，50 个 diffusion sampling steps。

pair manifest 覆盖 896/996 条正样本，即 89.96%，共 56 个 batch。matched cosine mean/median 为 0.999633/0.999993；unmatched 为 0.957752/0.961084。manifest SHA256 为 `3f3cfc61bf8a539e9b17a878d5ceb05d01870423729c8389560521a71be91bea`。

## 主结果

| 方法 | Correct Success | Correct BranchAcc | Zero BranchAcc | Random BranchAcc | Matched-wrong Success | Matched-wrong BranchAcc | Success gap | Branch gap | Offline ΔCF | Wrong-cup uplift | First-direction uplift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VPWEM | 17% | 39% | 13% | 17% | 4% | 9% | 13pp | 30pp | 0.214 | 25pp | 47pp |
| Random-CF | 16% | 32% | 16% | 16% | 4% | 10% | 12pp | 22pp | 0.265 | 25pp | 50pp |
| Unmatched-Branch-CF | 16% | 30% | 14% | 16% | 3% | 9% | 13pp | 21pp | 0.267 | 25pp | 51pp |
| Matched-CF | 13% | 30% | 14% | 14% | 5% | 10% | 8pp | 20pp | 0.291 | 27pp | 50pp |

Matched-CF 相对控制的 paired correct-rollout 比较：

| 比较 | Success Δ (95% CI) | BranchAcc Δ (95% CI) | Branch McNemar p |
|---|---:|---:|---:|
| Matched − VPWEM | −4pp [−11, 3] | −9pp [−17, −1] | 0.0490 |
| Matched − Random-CF | −3pp [−9, 3] | −2pp [−7, 3] | 0.6875 |
| Matched − Unmatched | −3pp [−8, 1] | 0pp [−5, 5] | 1.0 |

Matched-CF 内部 correct−matched-wrong 的 BranchAcc 差异显著（精确 McNemar p=0.000325），但 Success 差异不显著（p=0.0768）。更重要的是，同样的错误历史干预在 VPWEM 和两个控制上产生相等或更大的 gap，所以这不能归因于 matched training。

zero-memory 把四组 BranchAcc 从 30–39% 降到 13–16%，说明当前协议确实要求 memory；这是 intervention diagnostic，不是已经加入 decoder 的 learned necessity gate。

## 预注册门槛

| 门槛 | 结果 |
|---|---|
| Matched-CF 相对 VPWEM BranchAcc ≥ +10pp | 未通过（−9pp） |
| Matched-CF correct−matched-wrong Success ≥ 15pp | 未通过（8pp） |
| Matched-CF Success 不比 VPWEM 低超过 5pp | 通过（−4pp） |
| Matched-CF offline ΔCF 高于三控制 | 通过 |
| 错误历史吸引最终杯选择 | 通过 |
| 错误历史吸引第一运动方向 | 通过 |

只有 4/6 门槛通过，最终判定 NO-GO。

## 复现与审计

- 四个训练 checkpoint 均为 549,074,919 bytes，包含 `model`、`model_ema`、`optimizer`；model/model_ema 各 333 tensors，四个 SHA256 均不同。
- 四组 step-0 correct action loss 精确相同，均为 `0.472758412361145`。
- 100 个 `(reset_seed, env_slot)` episode key 全部唯一；所有方法和四种 intervention 的 episode/GT 表逐元素相同。
- matched-wrong donor 表在四方法间逐元素相同，且 100/100 donor branch 都不同于 GT branch。
- 20,000 次 paired bootstrap 与精确 McNemar 分析独立重跑后，`summary.json` 的 SHA256 仍为 `f49399d2521aad75698fa93520acce6140b76762e314ff082fc2c78d952413ea`，逐字节一致。
- Matched-CF 离线评估在 GPU 6 独立重跑后，JSON SHA256 均为 `29345c363104151f793ecc222750952e158611417e0345766f28e24382ce1d2d`，逐字节一致。
- NCM 相关测试为 16 passed。
- 数据 marker、oracle checkpoint、collector、base checkpoint、pair manifest、训练权重、rollout、offline JSON 和关键源码的 SHA256 均记录在 [reproducibility.json](../artifacts/ncm_vpwem_shellgame_stage1_oracle_seed100/analysis/reproducibility.json)。
- 完整统计见 [summary.json](../artifacts/ncm_vpwem_shellgame_stage1_oracle_seed100/analysis/summary.json)，可读摘要见 [RESULTS.md](../artifacts/ncm_vpwem_shellgame_stage1_oracle_seed100/analysis/RESULTS.md)，pair 质量见 [pair_quality.md](../artifacts/ncm_vpwem_shellgame_stage1_oracle_seed100/pair_manifest/pair_quality.md)。

服务器训练和评测只使用物理 GPU 2–6，没有使用 GPU 0/1。

## 实现落点

- 配对与固定 batch：[ncm_pairing.py](../cleandiffuser/dataset/ncm_pairing.py)
- 显式 partner 数据返回：[mikasa_dataset_longmem.py](../cleandiffuser/dataset/mikasa_dataset_longmem.py)
- per-sample matched-CF loss 与共享随机性：[longmem_ddpm.py](../cleandiffuser/diffusion/longmem_ddpm.py)
- 训练入口：[longmemdpptp_mikasa_image.py](../pipelines/longmemdpptp_mikasa_image.py)
- 物理 hold 数据收集：[collect_shellgame_neutral_hold.py](../tools/collect_shellgame_neutral_hold.py)
- manifest 构建与 probe：[build_ncm_shellgame_manifest.py](../tools/build_ncm_shellgame_manifest.py)
- 因果 rollout：[ncm_shellgame_eval.py](../pipelines/ncm_shellgame_eval.py)
- 离线 matched gap：[ncm_shellgame_offline_eval.py](../pipelines/ncm_shellgame_offline_eval.py)
- 统计分析：[analyze_ncm_shellgame.py](../tools/analyze_ncm_shellgame.py)
- 一键 runner：[run_ncm_shellgame_stage1_oracle.sh](../tools/run_ncm_shellgame_stage1_oracle.sh)

Q-Former 与 decoder 架构没有改动；oracle 只用于 demonstration action 与只读评测诊断，从未进入 policy observation。

## 局限与下一步

本轮只有一个训练 seed、100 个配对 eval episodes，而且 600k 初始化来自旧的自然-prefix 数据，新物理-prefix 只微调 nominal 10k。13–17% 的绝对 Success 说明前缀分布适配不足，限制了 positive headroom；不过四组预算、episode 和噪声完全对齐，所以足以支持本轮 NO-GO，而不足以宣称 matched supervision 永久无效。

下一步优先级应是：先用同一物理-hold 数据把 VPWEM 训练到可靠的绝对 Success，再在固定协议下重复四组、多 seed 比较。只有 Matched-CF 同时取得相对控制的 rollout 增益和更强的 donor-directed causal effect，才进入 learned necessity gate；compact belief、slots、revision 继续推迟。

早期“给原 action 序列整体后移”的 hybrid 数据只有 309/1000 物理成功，不能作为主实验。其结果保存在 [diagnostic artifact](../artifacts/ncm_vpwem_shellgame_stage1_hybrid_seed100/analysis/RESULTS.md)，仅用于记录失败路径。
