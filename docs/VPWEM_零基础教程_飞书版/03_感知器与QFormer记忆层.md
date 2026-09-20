# 03 感知器与 Q-Former 记忆层

## 1. 感知器：把不同观测翻译成同一种 token

真实类：`cleandiffuser/nn_condition/multi_image_condition.py`。

图像分支先经过 ResNet，低维状态直接保留，随后拼接并用 MLP 投影：

```text
RGB 图像 [B,T,3,H,W] -> ResNet -> [B,T,F_rgb] ┐
手腕图像                 -> ResNet -> [B,T,F_wrist] ├─ concat -> MLP -> [B,T,256]
机器人状态 [B,T,P]       ────────────────────────┘
```

对应代码骨架：

```python
feature = self.key_model_map[key](img)  # 每个 RGB key 的 ResNet
features.append(feature)
features.append(data)                   # low_dim
features = torch.cat(features, dim=-1)
result = self.mlp(features)             # -> emb_dim
```

主配置中的 `dp_emb` 已经是预计算的 256 维观测 embedding，所以概念上可以直接进入后续模块；原始图像 pipeline 则会在线执行感知器。

## 2. QFormerMemory 的输入输出契约

```text
输入：memory [B,Th,D] + ep_step [B,Th]
输出：compressed_memory [B,Tm,D]
```

`ep_step` 是每帧在 episode 中的真实时间位置。模块先加位置编码，让相同画面在“很早以前”和“刚刚发生”时可被区别：

```python
memory = memory + self.mem_pos_emb(ep_step)
```

## 3. 三层对象关系

```text
QFormerMemory             管时间重放、位置编码、observation cache、在线接口
└── QFormer               管 learned queries、输入/输出投影
    └── QFormerLayer × N  管 self-attn、cross-attn、FFN、query cache
```

主例中 `qformer_num_layers=1`，所以只有一个 `QFormerLayer`；不是 8 层。8 层属于后面的动作 decoder。

## 4. learned queries 是什么

```python
self.query_tokens = nn.Parameter(torch.randn(1, n_queries, d_model))
```

主例形状 `[1,4,256]`，对 batch 展开后为 `[B,4,256]`。它们：

- 是训练参数，初始为随机数；
- 不是某四帧，也没有人工规定“颜色 query”“位置 query”；
- 像 4 个可学习的信息槽位，各自向历史提问；
- 经过所有 Q-Former 层后，直接成为 4 个 memory token。

## 5. 两类 cache

### 5.1 condition cache：保留历史证据

每次把历史观测加进来：

```python
condition_to_add = condition.detach()
self.condition_cache = torch.cat(
    [self.condition_cache, condition_to_add], dim=1)
```

其逻辑形状近似 `[B,cache_time,1,D]`，使用前展平为 `[B,cache_time,D]`。

### 5.2 query cache：保留过去的摘要状态

每个 `QFormerLayer` 都有自己的 `query_cache`，逻辑形状：

```text
[B, cache_time, Tm, D]
```

它不是原始图像，而是过去每次压缩后留下的 query 状态。

### 5.3 detach 的含义

写 cache 前调用 `.detach()`：后续仍能读这些数，但梯度不会无限穿过所有历史时刻。它限制反向传播图，不代表从数值上删除历史。

## 6. 单个 QFormerLayer 的真实顺序

```text
learned query
    │
    ├─ LayerNorm
    ├─ Self-Attention：读 query cache（过去已经总结了什么）
    ├─ 残差相加
    │
    ├─ LayerNorm
    ├─ Cross-Attention：读 condition cache（过去实际看到了什么）
    ├─ 残差相加
    │
    ├─ LayerNorm
    ├─ FFN: Linear(D,4D) -> GELU -> Linear(4D,D)
    └─ 残差相加 -> 下一层或输出
```

真实代码：

```python
attn_output, _ = self.self_attn(q, all_q, all_q)
query = query + self.dropout1(attn_output)

attn_output, _ = self.cross_attn(q, key, value)
query = query + self.dropout2(attn_output)

ffn_output = self.linear2(self.dropout(
    self.activation(self.linear1(q))))
query = query + self.dropout3(ffn_output)
```

这里两次注意力的 Q/K/V：

| 子层 | Q | K/V | 目的 |
| --- | --- | --- | --- |
| self-attn | 当前 Tm 个 query | 历史 query cache + 当前 query | 继承过去摘要 |
| cross-attn | 已更新 query | observation/condition cache | 回看历史证据 |

## 7. 离线训练 forward 为什么有时间循环

```python
self.reset()
for t_idx in range(Th):
    self._update_condition_cache(memory[:, t_idx:t_idx+1])
    cp_memory = self.qformer(self.condition_cache_view)
return cp_memory
```

训练样本一次给出一段历史，模块先 reset，再按时间顺序“重演”，最终只返回最后一次的 `[B,Tm,D]` 摘要。

## 8. 在线 inference 为什么不能每次 reset

在线环境每次只收到新离开 working-memory 窗口的旧观测：

```python
compressed = memory_module.inference(new_memory)
```

它保留 cache 到下一次调用，并按 `subsample_ratio` 稀疏写入。只有 episode 结束/重置时才应：

```python
agent.model["memory"].reset()
agent.model_ema["memory"].reset()
agent.reset_cache()
```

忘记 reset 会把上一个 episode 的信息泄漏给下一个 episode。

## 9. cache 压缩策略

`cache_compress` 可选：

- `fifo`：满了删除最早项；
- `random`：随机淘汰；
- `kmeans`：用聚类中心替代；
- `adjsim`：合并相邻且相似的项；
- `none`：不压缩，长度可能无界。

注意两级压缩：cache 策略控制内部保留哪些证据；learned queries 再把这些证据读成固定 `Tm` 个输出。

## 10. 容易踩的代码边界

- `num_heads` 必须整除 `D`，如 `256/8=32`。
- `num_layers=0` 时不再使用 Q-Former，代码要求 `cache_max_length == mem_compress_length`。
- `use_pos_emb` 若既不是 `sin_pos` 也不是 `learned_pos`，当前实现不会创建 `mem_pos_emb`，之后会报错。
- 变长历史需要正确 padding/mask；当前 `QFormerMemory.forward` 接收 `mask`，但主体没有把它传给注意力，分析 padding 时必须留意这一实现事实。

## 11. 本章自测

1. learned query 是历史帧吗？不是，是可学习的信息槽位。
2. condition cache 与 query cache 分别保存什么？历史观测证据与过去摘要。
3. `Tm=4` 表示 Q-Former 有 4 层吗？不是，表示 4 个 query/输出 token。
4. 为什么 `.detach()` 后还能记住？数值保留，只切断跨时间梯度。
5. 修改 `num_heads` 前先检查什么？`D % num_heads == 0`。

---

下一篇：`04_动作Transformer与Diffusion.md`
