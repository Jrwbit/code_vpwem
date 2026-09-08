# VPWEM 六类 Memory 机制迁移筛选计划

日期：2026-08-26

## 目标

在相同 VPWEM condition encoder、Diffusion Policy、输出 memory-token 数和训练预算下，分别筛选 GMP、AEM、DSSP、VQ-Memory、Chameleon 和 CycleManip 的最小可迁移机制。这里的 `style` 变体是机制消融，不冒充原论文完整复现。

## 六个实验臂

| 实验臂 | 保留的最小机制 | 当前实现名 | 仍未包含的原论文组件 |
|---|---|---|---|
| GMP | 当前状态与扩散时刻共同预测逐通道 memory read gate | `qformer + gated` | 原论文完整跨 trial 记忆与噪声历史动作训练 |
| AEM | 历史视觉表示与对齐动作联合递归编码，预测 action effect | `aem_style` | 独立的大规模 masked pretraining 阶段 |
| DSSP | 固定状态递归历史编码与下一视觉特征预测 | `dssp_style` | 原论文特定 SSM block 与完整对比目标 |
| VQ-Memory | 离散码本量化历史状态，再形成固定 token | `vq_style` | 仅量化 proprioception 的专用 encoder 与 RuleSafe 配置 |
| Chameleon | 当前控制状态查询 VPWEM tokens，再融合 recalled trace | `qformer + control_indexed` | event trace propagation、Control-JEPA 和完整 rectified-flow policy |
| CycleManip | 历史递归状态与训练期 progress auxiliary loss | `cycle_progress` | 真正 cycle-stage 标签和 cyclic RoboTwin 数据；RememberColor5 上只能作为 progress proxy |

## 公平筛选

第一轮统一使用 RememberColor5、seed 100、5k updates。它只能排除训练不稳定或完全无信号的变体，不能形成性能结论。

晋级条件：

1. 前向、反向、在线 reset 和 checkpoint load 全部通过；
2. 相同 paired episodes 上 success 或 BranchAcc 至少一项优于 VPWEM；
3. 正确 memory 优于 zero/wrong memory，且变化方向与隐藏分支一致；
4. 推理成本和参数增量可报告；
5. 只有通过筛选的变体进入 20k/完整训练和 3-seed 评测。

## 解释边界

- `cycle_progress` 在非循环任务上的结果不能写成 CycleManip 复现。
- 当前 `vq_style` 量化融合后的历史 embedding；正式 VQ-Memory 对照还需改成只读取对齐的 proprioception history。
- 当前 Chameleon 实验只迁移 control-indexed read；若该读出没有正信号，不实现更复杂的 trace propagation 和 Control-JEPA。
- AEM 的 action history 必须真实对齐；代码会在缺失时直接报错，不允许用视觉差分冒充 action effect。

