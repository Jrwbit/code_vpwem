# 04 动作 Transformer 与 Diffusion

## 1. 为什么动作也要“扩散”

训练时从真实动作序列 `x0` 开始，随机加噪得到 `xt`：

```text
xt = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
```

网络看到 `xt`、噪声等级 `t`、近期观测和历史记忆，预测加入的噪声 `epsilon`：

```text
epsilon_hat = network(xt, t, recent_obs, memory)
loss = mean((epsilon_hat - epsilon)^2)
```

推理时没有真实动作：从随机噪声动作开始，重复去噪若干步，最后得到可执行动作序列。

## 2. 输入不是四种东西直接相加

类：`LongMemChiTransformerPTP`。

四类输入先各自变成 D 维 token：

```python
t_emb   = self.map_noise(noise).unsqueeze(1)  # [B,1,D]
act_emb = self.act_emb(x)                     # [B,Ta,D]
obs_emb = self.obs_emb(condition)             # [B,To,D]
mem_emb = self.mem_emb(memory)                 # [B,Tm,D]
```

条件序列按固定顺序拼接：

```python
cond_emb = torch.cat([t_emb, mem_emb, obs_emb], dim=1)
# [B, 1+Tm+To, D]
```

主例即 `[B,7,256] = 1 + 4 + 2`。带噪动作是 decoder 的另一组目标 token `[B,9,256]`。

## 3. 条件 Encoder 做什么

`cond_emb` 加上可学习位置编码后，经过条件 encoder：

```python
final_emb = self.drop(cond_emb + cond_pos_emb)
final_emb = self.encoder(final_emb)
```

主 pipeline 没显式传 `n_cond_layers`，默认 0。此时所谓 encoder 实际是逐 token 的 MLP，而不是 Transformer Encoder：

```python
Linear(D,4D) -> Mish -> Linear(4D,D)
```

如果设 `n_cond_layers > 0`，才会构建真正的 `nn.TransformerEncoder`，让 timestep、memory、obs 条件 token 先互相注意。

## 4. 一个动作 Decoder layer 内部

PyTorch 的 `TransformerDecoderLayer` 可概括为：

```text
动作 token
  ├─ masked self-attention：动作位置之间交流
  ├─ residual + norm
  ├─ cross-attention：动作 Q 去读条件 K/V
  ├─ residual + norm
  ├─ FFN: D -> 4D -> D
  └─ residual + norm
```

主例 `diffusion_num_layers=8`，即这个 decoder layer 堆叠 8 次；每层 `nhead=4`，每头宽度 `256/4=64`。

## 5. 两个 mask 各防什么

### 5.1 `tgt_mask`：动作不能偷看未来动作 token

9 个动作位置的因果可见性：

```text
       a0 a1 a2 a3 ...
a0      ✓  ×  ×  ×
a1      ✓  ✓  ×  ×
a2      ✓  ✓  ✓  ×
```

这约束动作 self-attention。

### 5.2 `memory_mask`：早期动作不能偷看未来观测

条件列顺序是 `[timestep, mem0...mem3, obs0, obs1]`：

```text
所有动作位置：都可看 timestep 和全部 memory token
动作位置 0：只可看 obs0
动作位置 1 以后：可看 obs0、obs1
```

这就是 PTP 的时间对齐逻辑。这里变量名 `memory_mask` 是 PyTorch decoder 对 cross-attention 条件的叫法，它同时覆盖 timestep、长期 memory 和近期 obs。

## 6. PTP 是什么

PTP = Past Token Prediction。主配置：

```text
To=2, action_steps=8
Ta=horizon=To+action_steps-1=9
```

训练 9 个动作位置；rollout 通常从 `To-1` 对应位置开始取连续 8 步。PTP 是动作/观测时间对齐机制，不是长期记忆模块。

## 7. 从 decoder 输出回到动作空间

```python
x = self.decoder(...)
x = self.ln_f(x)  # [B,Ta,D]
x = self.head(x)  # [B,Ta,A]
```

输出在训练时是预测噪声，不是直接执行动作。DDPM 还要用它完成反向更新；最后一次去噪完成的 `x0` 才是动作序列。

## 8. 训练层与采样步不要混淆

| 名称 | 主例 | 含义 |
| --- | ---: | --- |
| `diffusion_num_layers` | 8 | 一次网络前向中 decoder 堆几层 |
| `sample_steps` | 50 | 推理从噪声到动作重复调用网络几次 |
| `qformer_num_layers` | 1 | 记忆压缩网络堆几层 |

50 个 diffusion step 每一步都会调用同一个 8 层 decoder；不是构建 50 个不同网络。

## 9. 三种 memory 读取方式

- `context`：原始基线，memory token 直接作为条件。
- `gated`：当前 obs 与 timestep 产生 `[B,D]` 门值，再逐通道缩放所有 memory token。
- `control_indexed`：当前 obs 作为 Q，再对 memory K/V 做一次检索并融合。

后两者是仓库扩展。若要公平比较，只改变 `memory_read_mode`，其余数据、seed、Tm、训练步数应一致。

## 10. 本章自测

1. diffusion 网络训练时预测什么？默认预测加到真实动作上的噪声。
2. 动作 decoder cross-attention 的 Q/K/V 是谁？动作 token / 条件 token / 条件 token。
3. `memory_mask` 只管长期 memory 吗？不是，它管整个条件序列可见性。
4. `n_cond_layers=0` 时有没有 Transformer 条件 encoder？没有，是 MLP。
5. 8 层 decoder 与 50 步采样是什么关系？同一个 8 层网络被重复调用 50 次。

---

下一篇：`05_训练与在线推理调用链.md`
