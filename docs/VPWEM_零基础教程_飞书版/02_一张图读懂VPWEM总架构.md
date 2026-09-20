# 02 一张图读懂 VPWEM 总架构

## 1. VPWEM 为什么存在

普通短上下文策略近似认为：“只看最近几帧，就足以决定现在该做什么。”但 RememberColor 或 ShellGame 任务的关键线索早已离开画面：

```text
相同的当前画面 + 不同的早期线索 = 不同的正确动作
```

把整个 episode 所有帧直接塞进 Transformer 又会越来越慢、越来越占显存。VPWEM 的折中是：近期保真，远期压缩。

## 2. 两条时间通路

```text
episode:  o0  o1  o2  ...  o(t-2)  o(t-1)  ot
          └────── 更早历史 ──────┘  └─最近 To帧─┘
                    │                    │
                    ▼                    ▼
             Q-Former 压缩         直接保留
             [B,Th,D]              [B,To,D]
                    │                    │
                    ▼                    │
             [B,Tm,D]                   │
                    └────────┬───────────┘
                             ▼
                       动作 Transformer
```

- Working memory：最近 `To` 帧，不压缩，保留局部细节。
- Episodic memory：更早的单回合历史，压成固定 `Tm` 个 token。
- 新 episode 必须清空 cache；它不是跨任务的终身记忆。

## 3. 用主配置代入真实数字

配置：`configs/dp/mikasa/chi_transformer_ptp_longmem/remember_color_5_emb.yaml`

```yaml
action_dim: 8
obs_steps: 2
action_steps: 8
horizon: 9
diffusion_num_layers: 8
mem_compress_length: 4
qformer_emb_dim: 256
qformer_num_layers: 1
qformer_cache_max_length: 8
dataset_memory_subsample_ratio: 20
```

假设 `B=64, Th=12`，一次训练的主形状是：

```text
近期观测 condition       [64, 2, 256]
更早历史 memory          [64,12, 256]
压缩历史 compressed      [64, 4, 256]
真实/带噪动作             [64, 9,   8]
动作 token                [64, 9, 256]
完整条件序列              [64, 7, 256]  # 1 timestep + 4 memory + 2 obs
预测噪声                  [64, 9,   8]
```

## 4. 四个核心对象各管什么

| 对象 | 文件 | 责任 | 不负责什么 |
| --- | --- | --- | --- |
| `MultiImageObsConditionRobomimic` | `nn_condition/multi_image_condition.py` | 图像/低维状态 → D 维 token | 不记历史、不生成动作 |
| `QFormerMemory` | `nn_memory/qformer_memory.py` | 可变历史 → 固定 Tm token | 不直接看动作 loss 的定义 |
| `LongMemChiTransformerPTP` | `nn_diffusion/longmem_chitransformerptp.py` | 条件化带噪动作，预测噪声 | 不维护在线历史 cache |
| `LongMemDDPM` | `diffusion/longmem_ddpm.py` | 加噪、调用三网、loss、EMA、反向采样 | 它是编排器，不是第四种特征层 |

## 5. 真实构造顺序

入口 `pipelines/longmemdpptp_mikasa_image.py` 的关键逻辑：

```python
nn_condition = MultiImageObsConditionRobomimic(..., emb_dim=256)

nn_diffusion = LongMemChiTransformerPTP(
    action_dim, 256, horizon, obs_steps, mem_compress_length,
    d_model=256, nhead=4, num_layers=diffusion_num_layers)

nn_memory = QFormerMemory(
    mem_compress_length=mem_compress_length,
    emb_dim=qformer_emb_dim,
    num_layers=qformer_num_layers,
    cache_max_length=qformer_cache_max_length)

agent = LongMemDDPM(
    nn_diffusion=nn_diffusion,
    nn_condition=nn_condition,
    nn_memory=nn_memory)
```

这解释了为什么 VPWEM 不能只看一个 `model.py`：它在 pipeline 中被组装。

## 6. 三种“长度”不要混淆

| 参数 | 控制对象 | 改大后的直接后果 |
| --- | --- | --- |
| `dataset_maximum_memory_length` | 离线样本最多取多少历史帧 | 输入历史证据增多 |
| `qformer_cache_max_length` | Q-Former 内部 cache 最多保留多少时刻 | 内部记忆容量/开销增大 |
| `mem_compress_length = Tm` | 最终给动作策略多少历史 token | decoder 条件长度与参数接口变化 |

`cache_max_length=8` 不代表最终输出 8 个 token；主配置最终输出是 `Tm=4`。

## 7. 原始基线与当前扩展

最小 VPWEM 路径：

```text
QFormerMemory + memory_read_mode=context + action diffusion MSE
```

当前仓库还能选择：

- `memory_read_mode=gated`：对 memory token 加读取门；
- `control_indexed`：用当前控制状态再次检索 memory；
- `PERMemory`、`*_style`：替换记忆模块；
- `memory_aux_coeff`、`counterfactual_memory_coeff`：添加训练目标；
- Direct Spatial History：直接保留历史空间 token。

学习时先掌握最小基线，再看扩展，否则容易把论文方法和新实验混为一谈。

## 8. 本章自测

1. 为什么近期观测不压缩？保留当前控制所需的精细信息。
2. 为什么历史要压成固定 Tm？使下游条件长度和计算预算有界。
3. 谁维护训练/采样流程？`LongMemDDPM`。
4. 谁真正预测动作噪声？`LongMemChiTransformerPTP`。
5. `Th` 改变时，`Tm` 必须改变吗？不必，压缩器就是做 `Th -> Tm`。

---

下一篇：`03_感知器与QFormer记忆层.md`
