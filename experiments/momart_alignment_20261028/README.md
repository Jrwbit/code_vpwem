# MoMaRT 输入对齐审计与 reset 修复

本目录针对 BHR `code_vpwem_boxed_eval_20261026` 的 `dbf6779` 快照。
历史训练权重、数据、原始评测工作树和成功率不修改。

`precision_reset.patch` 修复 `PrecisionSequenceDetector` 在 episode reset 时，
将 `[frame0, frame0]` 的补帧作为两次真实时刻处理的问题。
仅复用首次调用中连续相同的前缀；后续调用里，即使两张图像相同，
最新帧仍推进 tracker。环境之间缓存独立，reset 清除前回合缓存。

回归验证（脚本只提取原类 AST，替换昂贵的 detector，类方法原样执行）：

```bash
python tools/check_momart_precision_reset.py /path/to/momart_bbox_memory/precision_detector.py
```

原版本 5 项中 3 项失败；应用补丁后 5 项通过。补丁应应用于新工作树：

```bash
git -C /path/to/new-checkout apply --check /path/to/precision_reset.patch
git -C /path/to/new-checkout apply /path/to/precision_reset.patch
```

完整只读审计，使用具有 MoMaRT 依赖的 BHR Python：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jrw/anaconda3/envs/vpwem/bin/python \
  /path/to/tools/audit_momart_boxed_alignment.py \
  --source-root /home/jrw/code_vpwem_boxed_eval_20261026 \
  --training-root /home/jrw/code_vpwem_momart_bbox_memory \
  --work-dir /home/jrw/vpwem_outputs/momart_boxed_current_tokens_bs256_150k/seed100 \
  --boxed-work-dir /home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_150k/seed100 \
  --output /tmp/momart_alignment_audit_20261028.json
```

可追加 `--detector-bundle /path/to/detector_bundle.json` 在 CPU 对一个真实 episode
的小片段比较离线缓存、同源实时检测和在线跟踪，不运行机器人 rollout。
默认不重新运行检测器。探针、报告不包含图像或权重，生成 JSON 不提交 Git。

此修复不解决原图坐标/随机裁剪参考系、RGB/深度独立增强、训练/评测框来源差异。
这些是单独的训练协议问题，不能通过给旧 checkpoint 临时换坐标语义解决。
