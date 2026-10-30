# 已运行的 MoMaRT 带框 No-History 基线

归档日期：2026-10-30（北京时间）。来源为 BHR `/home/jrw/code_vpwem_boxed_bs64_1m_20261029` 实际源码；该目录无 Git 元数据，因此本分支首次完整保存此前仅在服务器上的三个 Python 包及必要数据采样修订。

这是可回退的 WORKING BASELINE；working 的证据边界为训练持续运行、回归检查及现有权重推理通过，不表示新 BS64 1M 训练已完成或已得到其最终成功率。

- 分支：`baseline/momart-boxed-working-20261030`，标签：`momart-boxed-working-20261030`。
- 主体：ResNet18＋LongMemChiTransformerPTP，最近2帧，多模态256维融合及task embedding；长期memory为0；仅主RGB画固定红碗框。
- 配置：`experiments/momart_boxed_nohistory_bs64_1m_20261029/config.yaml`，seed100、batch64、预测9步、执行8步、50次DDPM、目标1M更新。
- 启动器：同目录 `run.sh`；绝对路径是原服务器复现入口，换机器时需修改数据与输出根目录。
- 10月30日本轮只读核查日志最新为533000步，loss=0.0693003，已存在500k checkpoint；没有停止、重启或更改该任务。
- 原服务器原代码检查：21 passed，覆盖画框、真实十数据集采样/episode边界、cache、检测选择、原tracker、任务encoder及cosine延长。
- 原500k权重在CPU严格加载model/EMA成功，50步EMA采样得到有限 `[1,9,10]` 动作；使用合成观测做接口验证，不能当真实rollout成功。
- `source_manifest.json` 记录70份源文件SHA256及服务器依赖版本，已逐文件核对本地归档与服务器一致；不含权重、数据、缓存或凭据。
- 既有完整评测及限制沿用 `docs/momart_total_success_20261029.md`，不把历史250k结果当本次500k/1M结果。

## 已知限制保留

离线独立检测和在线combined tracker并非完全同协议；reset补帧重复推进问题尚保留在此原基线，不能将历史结果标为已修复。后续A1/A2/A3分支将统一观测流和reset合同，并重跑共同对照。该标签保存原始工作状态，后续修订不得移动本标签。

## 验证命令（原服务器环境）

```bash
/home/jrw/anaconda3/envs/vpwem/bin/python -m pytest -q \
  momart_boxed_no_history/test_painting.py \
  momart_boxed_no_history/test_dataset.py \
  momart_no_history_baseline/test_cache.py \
  momart_bbox_memory/test_detector.py \
  momart_bbox_memory/test_precision_sequence_detector.py \
  tests/test_momart_no_history_baseline.py \
  tests/test_boxed_cosine_extension.py
```

后续开发从此归档提交创建 `feat/momart-object-memory-a123`，独立工作区 `/home/wjr/code/code_vpwem_momart_memory`。不修改原服务器工作目录或当前运行任务。
