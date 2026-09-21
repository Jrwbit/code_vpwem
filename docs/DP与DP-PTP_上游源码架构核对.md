# DP 与 DP-PTP 网络架构源码核对

## 源码位置与版本

2026-09-21 从 GitHub 浅克隆到本机，保留各自的 Git 历史和提交号：

| 项目 | 本机目录 | 上游提交 |
| --- | --- | --- |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | `/home/wjr/code/diffusion_policy_upstream` | `5ba07ac6661db573af695b419a7947ecb704690f` |
| [Learning Long-Context Diffusion Policies via Past-Token Prediction](https://github.com/long-context-dp/ldp) | `/home/wjr/code/ldp_upstream` | `5113f461cb0e4193cab1a1be8bbceb975fdd57cc` |

这些目录是源码副本，没有下载数据集和权重。本仓库的 MoMaRT 实验采用 CleanDiffuser 版本；下面把它与两份上游源码分开说明。

## 上游 DP 的 Transformer 路径

入口是 `diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py`，去噪网络在 `diffusion_policy/model/diffusion/transformer_for_diffusion.py`。路径如下：

```text
最近 To 帧观测 ──视觉/低维编码器──> To 个观测特征 ──线性投影──┐
扩散步 t ──正弦时间嵌入─────────────────────────────────────┤ 条件 token
带噪动作 x_t ──线性投影+位置嵌入──> Transformer Decoder × 8 <──┘
                                ├─因果 action self-attention
                                └─对时间/观测条件的 cross-attention
                  LayerNorm + Linear ──> 每个动作位置的噪声预测
```

上游默认图像 Transformer 配置为 `n_obs_steps=2`、`horizon=10`、`n_action_steps=8`、`n_emb=256`、8 层、4 heads；训练用 DDPM 的 epsilon 目标。观测编码器由 Robomimic 的 `ObservationEncoder` 构造，具体视觉流随任务配置变化。时间与观测 token 进入条件侧，带噪动作 token 进入 decoder；推理反复去噪后从动作序列的索引 `To-1` 开始取执行块。见上游 `diffusion_policy/config/train_diffusion_transformer_hybrid_workspace.yaml` 和上述两个源码文件。

**容易混淆之处：**上游 DP 的 `causal_attn=True` 配置使用 `t >= s-1` 的 cross-attention mask，且其原始 `compute_loss` 对整段动作算损失。因此不能把本仓库 `ChiTransformer` 的 mask 直接当作上游官方 DP 的 mask。

## 上游 PTP 增加了什么

PTP 指 **Past-Token Prediction**。LDP 上游复用 DP 的 Transformer 主干和时间对齐 mask；关键训练开关在 `diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py`：`past_action_pred=False` 时丢掉前 `To-1` 个动作位置的 loss，`True` 时保留这些过去动作位置的 loss。其 `experiment_configs/square/transformer_square_past.yaml` 设置 `past_action_pred: true`。论文研究的是长上下文下，用过去动作 token 的预测提供额外训练信号；仅看 mask 不能概括完整 PTP 方法。

## 本仓库 MoMaRT 的 DP / DP-PTP 实际比较

构造入口是 `tools/print_momart_architectures.py`，DP-PTP 训练入口是 `pipelines/dpptp_momart_image.py`，配置在 `configs/dp/momart/chi_transformer_ptp/momart.yaml`。DP 用 `chi_transformer` 配置与 `ChiTransformer` 类重建；当前仓库没有 `pipelines/dp_momart_image.py`，尽管打印脚本把它列为入口，因此 DP 的 raw-image MoMaRT 训练入口不能由现有文件直接核实。已保存的模块实例输出见 `reports/momart_architecture_prints_20260920/{dp,dpptp}.txt`。

| 部件 | DP | DP-PTP |
| --- | --- | --- |
| 观测编码 | 相同：4 路 RGB/深度各自的 ResNet18，加低维 proprio 和 scan，经 MLP 成 `[B,2,256]` | 相同 |
| 条件序列 | `[扩散步 token, o₀, o₁]` | 相同 |
| 动作去噪 | 9 个 10 维动作 token；256 宽、8 层、4 heads 的 Transformer Decoder | 相同 |
| 动作 self-attention | 因果 mask | 相同 |
| 条件 cross-attention | `t + To >= s-1`；`To=2` 时所有动作位置可见两帧观测 | `t >= s-1`；动作位置 0 看不到 `o₁`，从位置 1 起可见两帧 |
| 训练/执行 | 若用仓库通用 DDPM，9 个位置算噪声 MSE；配置为 50 步采样、执行索引 1–8 | PTP 训练入口证实采用该 DDPM 路径与 9 个位置 loss；评测执行索引 1–8 |
| 参数量 | 54,879,350 | 54,879,350 |

其中 `t` 是从 0 开始的动作位置，`s=0,1,2` 依次对应扩散步、`o₀`、`o₁`。`0` 表示允许 attention，`−∞` 表示遮挡：

```text
                  扩散步   o₀    o₁
DP       动作 0      0      0     0
         动作 1+     0      0     0
DP-PTP   动作 0      0      0    −∞
         动作 1+     0      0     0
```

两个类的源码 `cleandiffuser/nn_diffusion/chitransformer.py` 和 `chitransformerptp.py` 逐行比较，除类名相关初始化外，唯一运算差别就是上述 mask 公式。`cleandiffuser/diffusion/ddpm.py` 的 `loss()` 对整个动作张量计算损失；PTP 的 MoMaRT 训练入口未传入位置 loss 权重。因此，这里的网络对照主要隔离了 **cross-attention 的时间对齐**，并非 LDP 上游 `past_action_pred` 开关的完整消融。现有 DP 数字属于脚本重建的网络结构和参数量，不能当作训练过的 DP 模型结果。
