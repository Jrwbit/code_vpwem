# MoMaRT：论文 Figure 4 与本次 1M 多任务实验

| 任务配置 | 论文 DP-VPWEM | 论文 MaIL-VPWEM | 本次 VPWEM 1M | 本次 direct-token2 1M |
|---|---:|---:|---:|---:|
| dwclean / expert | ≈65% | ≈71% | 63.33% | 60.00% |
| dwclean / suboptimal | ≈37% | ≈35% | 13.33% | 30.00% |
| skclean / expert | ≈82% | ≈81% | 70.00% | 60.00% |
| skclean / suboptimal | ≈27% | ≈25% | 33.33% | 70.00% |
| dwsetup / expert | ≈88% | ≈87% | 83.33% | 90.00% |
| dwsetup / suboptimal | ≈80% | ≈87% | 80.00% | 96.67% |
| dssetup / expert | ≈95% | ≈96% | 100.00% | 96.67% |
| dssetup / suboptimal | ≈72% | ≈76% | 76.67% | 80.00% |
| unload / expert | ≈77% | ≈79% | 80.00% | 63.33% |
| unload / suboptimal | ≈64% | ≈77% | 53.33% | 70.00% |
| **Average** | **69%** | **71%** | **65.33%** | **71.67%** |

## 口径说明

- 论文逐任务点没有打印数值，上表为 Figure 4 柱高约读数；论文 Average 的 69% 和 71% 是图中明确打印的数值。
- 论文 Figure 4 报告三次随机种子的均值及标准差。
- 本次结果来自单个多任务混训 seed（seed 100）的 1M checkpoint；每个任务配置评测 30 episodes，共 300 episodes/方法。
- 本次 direct-token2 不是 MaIL-VPWEM，二者只在同图中展示，不应视为同架构复现。
- 论文为逐任务实验，本次为多任务混训后逐任务评测，因此该对比用于定位性能，不是严格同协议复现。
