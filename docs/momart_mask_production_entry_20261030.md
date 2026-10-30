# Mask 正式生产入口已接入并通过真实样本协议验收

2026-10-30（北京时间）。任务1已完成：正式默认配置的 `FrozenPerception / cache / prepare_train / train / evaluate` 共用冻结Mask前端。代码提交 `34f337c`，工作树 `/home/wjr/code/momart-memory-association`；BHR隔离执行目录 `/home/jrw/code_vpwem_momart_association_20261030`。

本记录更新此前关联消融报告中的“正式入口仍是GDINO”状态：该阻塞现已解除。**这只完成Gate A的感知/缓存协议项；没有运行策略smoke训练、全数据缓存或闭环，不表示Gate B通过。** 关联默认仍为原IoU对照，未启用上轮尚未达标的外观运动候选。

## 接入范围

- `mask_detector.py` 提供生产Mask R-CNN ResNet50-FPN，7类含背景，严格加载原权重；不依赖实验脚本。保持诊断时的320/448缩放、FP32、六类阈值、class0红色支持过滤及NMS。
- `FrozenPerception` 按 `detector_backend` 显式分派；正式 `config.yaml` 默认Mask。对象ROI仍是冻结500k EMA ResNet18 layer2、原始未画框90×120 RGB、[-1,1]、stride8、ROIAlign 3×3/aligned/sampling2均值128维。
- 固定8槽及最多8个class-balanced检测；离线和在线共享 `make_bank`、关联、历史、淘汰与真实传感器时间。读取周期0.4秒、每次使用真实最近两帧，reset重复帧只更新一次。
- Mask权重SHA256、ROI权重SHA256、类别顺序、六类阈值、ROI参数、库版本、前端源码和红色过滤源码写入感知签名；关联/容量/时序另写stream签名。`ObjectDataset` 和训练入口校验完整Mask元数据，评测入口校验保存的stream及感知签名。
- `prepare_train` 增加 `--prepare-only`，可只生成和核验缓存；正式默认使用独立新缓存目录 `momart_objects_mask8_production_v1`、新运行目录 `momart_object_memory_mask8_v1`。未重启旧预处理，未复用旧GDINO对象缓存。
- 原主RGB沿用独立归档红碗路径。Mask不会将对象框画入主图；原主视觉代码、权重、bundle和基线目录未更改。显式旧GDINO配置保留用于历史复现，不能与Mask对象缓存混用。

冻结Mask权重：`/home/jrw/vpwem_outputs/momart_detector_drawer_augmented_20261030/training/mask_rcnn/best.pt`，SHA256 `6403f0154dbab64bed0708e01d270a3497272bf44740ce2a2dc0787079d5fd36`。ROI源checkpoint SHA256 `418ac07eba671fcc615b4953abd3bea06c8810570e676a8cc9ca4d38bc812239`。六类阈值沿用0.05、0.60、0.50、0.90、0.80、0.35（配置保留原浮点值），没有训练或重新选阈值。

## 实际验收

从真实训练视图提取两条训练episode的各25帧前缀，合计50帧：`unload_dishwasher_to_dresser_suboptimal/demo_1`、`table_setup_from_dresser_expert/demo_0`，均为物理帧0–24。使用原始RGB、动作及其他模态，并保存源路径、物理帧映射与逐数组SHA256。生成独立、明确标为训练前缀的验收数据；不是完整episode或独立效果测试集。

BHR物理GPU1、现有 `detector_compare` 环境执行 `tools/verify_momart_mask_production.py`。离线走生产 `prepare_dataset`（SQLite→所有相位Zarr）及 `ObjectDataset / train.split_batch`；在线通过生产Mask重新推理，不读取离线SQLite结果，再走 `OnlineObjects / evaluate.normalize_observations`。所有训练anchor均另按因果前缀重放，覆盖8种训练相位；部署phase0另执行真实主视觉路径。

验收前固定浮点 `atol=rtol=1e-5`、布尔/类别精确相等。本次8种数组的最大绝对差均为0；没有要求跨GPU或不同运行库逐比特一致。Token编码器为固定seed100的未训练共享GRU，检查协议计算一致性，不声称学会使用Memory。

| 验收项 | 实测结果 |
|---|---|
| 全部真实训练anchor | 50/50，覆盖所有训练相位 |
| A1/A2/A3逐字段数组比较 | 1392项通过，8字段最大绝对差均为0 |
| A1/A2/A3 Token比较 | 174次通过 |
| 训练batch对象字段 | 400项通过（50样本×8字段） |
| 评测归一化对象字段 | 64项通过（8次读取×8字段） |
| 非空Memory覆盖 | 部署query累计20个有效槽位，非全padding验收 |
| 与旧Mask诊断前端对照 | 4张真实训练图，类别/框/分数/ROI通过容差比较 |
| 离线主RGB、动作及其他模态 | 50个样本与原基线逐项相同 |
| 在线主RGB | 8次读取与独立重放的原归档boxer逐值相同 |
| 旧缓存或变更协议 | 10项实际拒绝检查通过 |
| 相关单元/回归检查 | 40 passed，6条现有依赖弃用警告 |

10项实际拒绝包括：两任务各自的GDINO对象元数据、变更阈值、变更关联配置、旧GDINO SQLite，以及评测缺失stream或改变容量签名。额外单元检查覆盖权重校验失败、类别顺序/容量/阈值无效、缓存writer和在线adapter提前拒绝旧前端。不能以修改配置后直接读旧对象缓存继续训练。

另实际执行正式CLI `python -m momart_object_memory.prepare_train --config .../config.yaml --device cuda:0 --prepare-only`；正常结束，`work/pipeline_status.json` 为 `stage=cache_complete`、`policy_training_started=false`。只对上述验收前缀生成和resume缓存，没有启动原全数据预处理或训练。

## 证据与复现

最终验收目录：`/home/jrw/vpwem_outputs/momart_mask_production_acceptance_v4_20261030/`，包含 `result.json`、`config.yaml`、`prepare_cli.log`、`work/object_protocol.json`、`work/pipeline_status.json`、独立source及objects缓存；完整日志为同路径加 `.log`。最终 `result.json` SHA256为 `e0bed8139c412f15b1d87595b0d7f777a642d9f1f621d211003f071b83ed9947`。模型、数据、SQLite、缓存及运行输出未加入Git。

```bash
# 只读原数据，在新的验收目录复现，不启动训练或仿真动作。
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 \
/home/jrw/vpwem_envs/detector_compare/bin/python \
  tools/verify_momart_mask_production.py --device cuda:0 --output NEW_AUDIT_DIR

# 正式新缓存入口；全量执行前按资源与数据范围选择配置。
python -m momart_object_memory.prepare_train \
  --config momart_object_memory/config.yaml --device cuda:0 --prepare-only
```

本轮验收仅支持“Mask生产接入及真实样本协议一致性通过”。全数据缓存、10k–30k策略smoke、身份/新鲜度独立验收与完整DP时延仍是后续工作。本地Git提交已保留；当前环境GitHub凭据助手失效的问题未变，不宣称已推送。

飞书追加并回读核验：[任务1验收记录](https://ccnuoe6sry36.feishu.cn/docx/RfXudCiBOoeG3Bx0D34cbU1rn0g#doxcn4ZGP14cR4mYZLgsPRwDXSb)。
