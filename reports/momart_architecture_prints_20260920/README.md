# MoMaRT 六种网络的实际 PyTorch 输出

以下文件由实验类直接实例化后执行 `print(module)` 生成，不是手绘结构图。每份文件包含：实验入口、协议、Condition、Memory、Diffusion、参数量以及实际 cross-attention mask。

| 方法 | Condition | Memory | Diffusion | 总参数量 |
|---|---|---|---|---:|
| DP | `MultiImageObsConditionMomart` | `None` | `ChiTransformer` | 54,879,350 |
| DP-PTP | `MultiImageObsConditionMomart` | `None` | `ChiTransformerPTP` | 54,879,350 |
| No-History | `TaskConditionedObsEncoder` | `None` | `LongMemChiTransformerPTP(Tm=0)` | 54,946,422 |
| Direct | `TaskConditionedDirectObsEncoder` | `None` | `LongMemChiTransformerPTP(Tm=1024)` | 55,748,214 |
| Direct-token2 | `TaskConditionedDirectObsEncoder` | `None` | `LongMemChiTransformerPTP(Tm=2)` | 55,477,384 |
| VPWEM | `TaskConditionedEmbeddingEncoder` | `QFormerMemory` | `LongMemChiTransformerPTP(Tm=2)` | 11,860,616 |

VPWEM 的参数量不含产生 `dp_emb` 的外部 context encoder，因为当前 1M 实验入口读入的是预计算 embedding；其余三个 History-4 raw-image 方法的视觉编码器包含在 Condition 参数量内。

输出文件：`dp.txt`、`dpptp.txt`、`no_history.txt`、`direct.txt`、`direct_token2.txt`、`vpwem.txt`。

重新生成：

```bash
PYTHONPATH=. /home/wjr/miniconda3/envs/vpwem/bin/python \
  tools/print_momart_architectures.py \
  dp dpptp no_history direct direct_token2 vpwem
```
