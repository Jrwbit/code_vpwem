# 01 先补地基：从张量形状到 Q/K/V

## 1. Tensor 就是有很多轴的数字盒子

一个 RGB 图像可写成 `[C,H,W] = [3,128,128]`：

- `C=3`：红、绿、蓝三个通道；
- `H=128`：128 行像素；
- `W=128`：128 列像素。

多一批图像变成 `[B,C,H,W]`；每个样本再有多帧，就变成 `[B,T,C,H,W]`。

```python
# 64 个样本，每个样本 2 帧 RGB 图像
x.shape == (64, 2, 3, 128, 128)  # [B,T,C,H,W]
```

最重要的区别：`C` 是原始图像通道，`D` 是神经网络提取后的特征宽度。图像经过编码器后常变成：

```text
[B,T,C,H,W] -> [B,T,D]
[64,2,3,128,128] -> [64,2,256]
```

这里每个 `[256]` 向量就是一帧的 embedding，也可叫一个 token。

## 2. token 不是一个字，而是一份固定宽度的信息卡

语言模型里 token 可能代表字词；VPWEM 里 token 可以代表：

- 一帧近期观测；
- 一份压缩后的历史摘要；
- 一个 diffusion timestep；
- 一步带噪动作。

它们含义不同，但进入 Transformer 前都被投影成相同宽度 `D=256`，这样才能做相似度、相加和注意力。

## 3. Linear 层是在“重新解释特征”

```python
layer = nn.Linear(8, 256)
y = layer(x)  # [...,8] -> [...,256]
```

数学上是 `y = xW^T + b`。它不会改变前面的 `B,T` 轴，只改变最后一维。

在动作网络中：

```python
self.act_emb = nn.Linear(act_dim, d_model)  # A -> D
self.obs_emb = nn.Linear(obs_dim, d_model)  # 观测 D -> 模型 D
self.head = nn.Linear(d_model, act_dim)     # D -> A
```

可把它理解成三个翻译器：动作语言 → Transformer 语言，观测语言 → Transformer 语言，Transformer 语言 → 动作语言。

## 4. 激活、归一化、dropout 各做什么

| 层 | 初学者直觉 | 是否改变形状 |
| --- | --- | --- |
| `GELU/Mish/LeakyReLU` | 给网络非线性，不再只是大矩阵乘法 | 否 |
| `LayerNorm(D)` | 把每个 token 的 D 个数拉回稳定尺度 | 否 |
| `Dropout(p)` | 训练时随机遮一部分数，防止过度依赖 | 否 |

注意：形状不变不等于层没作用，它改变的是数值与学习行为。

## 5. Q/K/V 到底是什么

把注意力想成在档案室查资料：

- `Q = Query`：我现在想找什么？
- `K = Key`：每份资料的目录标签是什么？
- `V = Value`：每份资料真正携带的内容是什么？

核心计算：

```text
score = Q @ K^T / sqrt(d)
weight = softmax(score)
output = weight @ V
```

第一步用 Q 与每个 K 算匹配分；第二步变成总和为 1 的权重；第三步按权重混合 V。

### 一个只有数字的例子

```text
Q 对三个 K 的分数 = [1, 5, 0]
softmax 后约为     = [0.018, 0.976, 0.007]
```

输出几乎就是第二份 `V`。所以注意力不是“复制最像的一个”，而是对所有 V 做可学习的加权汇总。

## 6. self-attention 与 cross-attention

区别只看 Q 与 K/V 来自哪里：

```text
self-attention:  Q、K、V 来自同一组 token
cross-attention: Q 来自 A；K、V 来自 B
```

VPWEM Q-Former 里：

```python
# 当前 query 去读“历史 query + 当前 query”
self.self_attn(q, all_q, all_q)

# 更新后的 query 去读历史观测证据
self.cross_attn(q, key, value)
```

因此第二行中的具体角色是：

```text
Q = learned query（我想从历史中提取什么）
K = 历史观测 token（每帧可匹配的索引）
V = 同一批历史观测 token（真正被取回的内容）
```

动作 Transformer decoder 里则是：

```text
Q = 每一步带噪动作 token
K/V = [diffusion step, memory tokens, recent observation tokens]
```

## 7. multi-head 为什么叫“多头”

若 `D=256, heads=8`，每个头通常处理 `256/8=32` 维。不同头可以学习不同匹配关系，例如物体颜色、位置、任务阶段，但代码没有强制每个头具体负责什么。

多头结果拼回 `D=256`，所以输入输出形状仍是 `[B,T,256]`。

## 8. 残差与 Pre-Norm

Q-Former 的结构类似：

```python
q = norm(query)
attn_output = attention(q, ...)
query = query + dropout(attn_output)  # 残差
```

`query + 新信息` 是残差连接：保留旧信息，只学习“应该补充什么”。先 `norm` 再进入子层叫 Pre-Norm，通常更利于深层训练。

## 9. Encoder 与 Decoder 不等于“压缩”和“解压”

在 PyTorch Transformer 语境中：

- Encoder：让一组条件 token 互相交流；
- Decoder：目标 token 先彼此做 self-attention，再 cross-attention 读取 Encoder 输出。

VPWEM 中 decoder 的目标 token 是带噪动作，不是文字；它输出的也是每一步动作的噪声预测。

## 10. 本章自测

1. `[32,9,8]` 中 9 是动作维度吗？不是，通常是 `Ta`；8 才是 `A`。
2. Q 与 K 决定什么？匹配权重。V 决定什么？取回的内容。
3. Cross-attention 的 K/V 一定是长期记忆吗？不一定，要看调用处。
4. `nn.Linear(8,256)` 会改变 batch size 吗？不会。
5. `C=3` 和 `D=256` 分别是什么？RGB 通道数与 token 特征宽度。

---

下一篇：`02_一张图读懂VPWEM总架构.md`
