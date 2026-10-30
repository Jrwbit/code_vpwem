# A1/A2/A3：带框 No-History 上的对象记忆原型

起点是已推送的 `baseline/momart-boxed-working-20261030` / `momart-boxed-working-20261030`，提交 `ebbc1b7`。基线及验证边界见 [归档说明](../baselines/momart_boxed_working_20261030/README.md)。开发在 `feat/momart-object-memory-a123`，不覆盖原服务器训练目录。

三组共用全部网络参数、初始化、检测/特征管线、槽位和训练循环，`object_memory.mode` 只控制输入证据：

| 模式 | 输入 |
|---|---|
| a1 | 仅当前可靠可见对象，空槽屏蔽 |
| a2 | 每对象最新可靠状态，离屏后保留外观及age |
| a3 | 每对象最近3次可靠观测，含相对时间，GRU压成一个256维token |
| a3_latest_control | 相同网络/容量，3个历史位置都重复最新状态 |

A1/A2同样使用GRU，只将更早位置标为无效。对象数上限8，超容量直接报错以暴露跟踪碎片，禁止静默覆盖身份。实例ID是内部槽位关联键，不作为整数语义特征。第一版为类别约束IoU匹配，尚无运动预测或可靠的遮挡后重识别；没有实现关系边、事件触发关键帧或额外动态reader。A3保存的是可靠观测序列，不声称已经学习到交互成功事件。

## 特征与时序

- 采用冻结GroundingDINO tiny，同一权重逐类别提示，不使用VLM Head。红色支持过滤只用于class0红碗；部件不继承红碗颜色约束。
- 对象特征从归档500k模型的EMA主RGB ResNet18权重严格加载；独立冻结，取layer2，使用未画框90×120原图、[-1,1]归一化和stride8 ROIAlign，输出128维。不会冻结随机网络或随主策略更新缓存特征。
- 保持原带框多模态主路径：两帧、256维、task embedding。离线直接保留 `BoxedNoHistoryDataset` 的画框 RGB；在线独立复用旧 `PrecisionSequenceDetector` / `OnlineBoxedRGB`，bundle SHA256 固定。对象分支的 class0 框也不重画主 RGB。原基线离线用红碗检测缓存、在线用 combined_blend 的已有差异被保留并显式记录，不宣称二者本来完全相同。
- 新对象token和validity随场景条件打包，DDPM只编码一次，去噪循环中解包只读。mask在decoder中真正屏蔽空槽，并禁止位置0的过去动作读取最新bank；默认条件encoder是逐token MLP。
- 空槽不可读；离屏对象仍可读但当前box归零。对象特征、可见性、置信度和age都按同一因果观测流更新。
- phase0部署查询为0、8、16…，每次只处理最近两帧，reset的[0,0]仅更新一次。训练支持所有8个相位的同节奏前缀增强；逐相位从reset回放，而不是让tracker看完整视频。相位定义在缓存manifest中显式记录。
- `period_seconds=0.05`取自本服务器robomimic `env_ig_momart.py`默认 `action_timestep=1/20`；若环境覆盖这个参数，必须更改配置并重新生成缓存。
- 原基线的RGB/深度/相邻帧随机裁剪规则继续保留；对象ROI使用独立原图坐标，当前实现不声称将原基线所有模态的裁剪窗口统一。

## 入口

以下为正式缓存和训练命令，尚未执行全数据缓存或正式A123训练。先完成对象覆盖、实例碎片、累计容量和感知时延审计；当前六类提示为初始配置，真实抽样已出现跨类别大框重叠，不能当检测可靠性验证通过。

```bash
# 生成三组共用缓存；拒绝覆盖已有目录，失败缓存不会标记complete。
python -m momart_object_memory.cache --config momart_object_memory/config.yaml --device cpu

# 相同配置只切换mode；各组独立输出目录。
python -m momart_object_memory.train --config momart_object_memory/config.yaml --mode a1
python -m momart_object_memory.train --config momart_object_memory/config.yaml --mode a2
python -m momart_object_memory.train --config momart_object_memory/config.yaml --mode a3

# 在三组初始化完成后，核对配置、感知、源文件和初始权重哈希。
python -m momart_object_memory.compare RUN_A1 RUN_A2 RUN_A3

# 单任务评测沿用原score/动作执行入口，读取训练保存的config和normalizers。
python -m momart_object_memory.evaluate \
  --checkpoint RUN_A1/models/model_100000.pt \
  --train-config RUN_A1/train_config.yaml \
  --dataset-path DATA_ROOT/table_cleanup_to_sink_expert.zarr \
  --env-name table_cleanup_to_sink --env-type expert \
  --work-dir EVAL_DIR --result-path EVAL_DIR/result.npy --device cuda:0
```

修改seed、预算或目录应生成一个共同配置文件，再给三组使用。`object_protocol.json`包含共同配置指纹、感知/特征版本、源文件哈希；`initialization.json`记录模型初始权重哈希及参数数。三组fingerprint允许差异仅为mode和输出目录。模型/优化器checkpoint与原baseline保存方式相同，但新入口拒绝自动续训：尚未实现完整的RNG与loader位置恢复，不把重新随机采样称为精确续训。评测不会覆盖训练初始化记录。

已有单碗bbox sidecar仍用于原数据集验证和normalizer构建；实际给模型的主RGB红框使用新共用感知缓存重绘。历史A0成绩不直接与新感知协议宣称因果比较；后续A0应在统一感知协议下重评。

## 检查

新增测试验证A123逐参数同初始化、padding和PTP不泄漏、遮挡/reset/多环境隔离、未来帧不影响前缀、所有训练相位缓存与在线重放一致、动作/其他模态/episode末尾补帧保留、GRU反向梯度和EMA去噪，以及原图坐标到stride8特征图的映射。运行环境为BHR现有vpwem，不依赖开发机旧torch环境。

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests/test_momart_object_memory.py \
  momart_boxed_no_history/test_painting.py momart_boxed_no_history/test_dataset.py \
  momart_no_history_baseline/test_cache.py momart_bbox_memory/test_detector.py \
  momart_bbox_memory/test_precision_sequence_detector.py \
  tests/test_momart_no_history_baseline.py tests/test_boxed_cosine_extension.py \
  tests/test_no_history_zarr_metadata.py
```

没有新闭环成功率，也没有新增在线实时频率结论。完整源报告、飞书链接及最终验证结果见 `docs/momart_object_memory_baseline_and_implementation_20261030.md`。


## 2026-10-30：GPU5 A1 BS256 / 250k 实验配置

`experiments/momart_a1_bs256_250k_20261030/config.yaml` 固定BS256、250k更新、seed100，保存间隔25k，共10个checkpoint。原型实际前缀在第8帧出现槽位超限，因此该实验显式使用 `max_detections=8`（每类别先保留最高置信实例、剩余额度按置信度填充）及 `overflow=evict_oldest_invisible`，新对象替换当前未匹配且最久未见槽位时清空旧历史；不覆盖当前已匹配实例。该规则必须同样用于后续A2/A3，不能混用首版error策略缓存。

新增 `prepare_train.py`，逐帧感知结果持久写SQLite，每64帧提交；数据与模型/类别/阈值版本匹配才恢复。随后按回合写128帧Zarr块并允许从完整回合边界续做，避免逐anchor写大量小文件。完成全部十数据集缓存后自动启动训练。训练本身仍拒绝未验证的自动续训。真实GPU5单次BS256反向更新已通过，峰值分配约9229MiB；这只是合成观测smoke，不计入正式250k。

## 感知与 memory 的比较约束（2026-10-30 修正）

`object_protocol.json` 使用 `independent_main_visual_v2`，分别保存主视觉约定、旧框检测元数据和对象感知元数据。旧协议不能直接用于新版评估。所有 arm 使用同一配置（只允许 mode / 输出目录变化）、源代码哈希、对象感知与冻结 ROI 权重、实例管理、缺失 mask、物理时刻、容量和初始化。A1/A2/A3 只改变当前状态 / 最新可靠状态 / 最近三次可靠观测的可读证据。

下一轮对象检测候选为领域 YOLO640，Mask R-CNN 保留作部件对照。当前默认对象检测代码仍为此前暂停的 GDINO 管线；未通过质量复核前，不把它当作可用的新检测器、不自动重启策略训练，也不把旧感知缓存复用为 YOLO 缓存。正式切换要同时记录新权重、类别顺序、dev 阈值、输入尺寸、后处理哈希，并为三组共建新缓存。原先出现的“新对象 class0 框替换主视觉框”的实现已删除，17 项测试包含离线与在线主 RGB 相等性检查。
