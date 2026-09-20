# DP、DP-PTP 与 History-4 网络结构图

这些图依据当前仓库实现绘制，口径是 MoMaRT 的 `obs_steps=2`、`horizon=9`、`action_steps=8`、`d_model=256`、8 层 Transformer decoder 与 50 次 DDPM 采样。

## 图片索引

- `history6_overview.png/.svg`：六种方法总览。
- `dp_vs_dpptp_masks.png/.svg`：原版 DP 与 DP-PTP 的 cross-attention mask 对照。
- `dp.png/.svg`：原版 DP 详细结构。
- `dpptp.png/.svg`：原版 DP-PTP 详细结构。
- `no_history.png/.svg`：No-History 详细结构。
- `direct.png/.svg`：Direct 详细结构。
- `direct_token2.png/.svg`：Direct-token2 详细结构。
- `vpwem.png/.svg`：当前 MoMaRT 1M VPWEM 详细结构。

## 核心区别

| 方法 | 长期历史 | 历史输出 token | Decoder |
|---|---|---:|---|
| 原版 DP | 无 | 0 | `ChiTransformer`；所有动作位置可读取全部近期观测 |
| 原版 DP-PTP | 无 | 0 | `ChiTransformerPTP`；近期观测按动作位置时间对齐 |
| No-History | 丢弃 | 0 | `LongMemChiTransformerPTP(Tm=0)` |
| Direct | 四路视觉的 4×4 空间网格，固定 16 帧 | 1024 | `LongMemChiTransformerPTP` |
| Direct-token2 | 与 Direct 相同，末端无参数平均池化 | 2 | `LongMemChiTransformerPTP` |
| VPWEM | `dp_emb` + episode 位置编码 + FIFO/Q-Former | 2 | `LongMemChiTransformerPTP` |

## 公平性边界

当前 MoMaRT 1M 对照中，Direct、Direct-token2、No-History 使用 raw-image policy 和 effective batch 64；VPWEM 使用预计算 `dp_emb`、任务对应 context encoder 和 effective batch 256。因此结构图准确反映当前已评测系统，但四者并非只改变记忆模块的严格单变量实验。

可重复生成：

```bash
python3 tools/draw_history6_architectures.py
```
