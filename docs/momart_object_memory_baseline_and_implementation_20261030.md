# MoMaRT工作基线Git归档与A1/A2/A3首版实现

日期：2026-10-30（北京时间）。飞书：https://feishu.cn/docx/UZGYdy4BQoPZYPxuwERcLr8Lnqg

按用户“先保存work基线，再在该基线上改”的要求，已先将服务器实际工作的带框No-History代码推送到GitHub并创建固定标签，随后在独立分支实现A1/A2/A3共用路径。没有修改原服务器训练目录或停止原训练。首版代码和配置已上传；全数据对象缓存与正式A123长训练尚未启动。

| 交付 | Git位置 | 证据边界 |
|---|---|---|
| 已运行基线 | baseline/momart-boxed-working-20261030，提交ebbc1b7 | 实际服务器源码、70文件哈希、21项检查、原500k权重严格加载及50步EMA采样 |
| 固定回退标签 | momart-boxed-working-20261030 | 已推送；标签不随开发移动 |
| A123首版 | feat/momart-object-memory-a123，代码提交0a0846d | 统一训练/评测入口、36项检查通过；尚无正式新闭环成功率 |

Git入口：[基线提交](https://github.com/Jrwbit/code_vpwem/commit/ebbc1b755e423a636f0fc0efc682a8a432ed766e)、[固定标签](https://github.com/Jrwbit/code_vpwem/tree/momart-boxed-working-20261030)、[开发代码提交](https://github.com/Jrwbit/code_vpwem/commit/0a0846d)、[开发分支](https://github.com/Jrwbit/code_vpwem/tree/feat/momart-object-memory-a123)。

**基线确实保存了什么**

原服务器目录 `/home/jrw/code_vpwem_boxed_bs64_1m_20261029` 没有Git元数据。本轮从该目录取回 `momart_boxed_no_history`、`momart_no_history_baseline`、`momart_bbox_memory` 三个包及必要采样修订和回归脚本，保存到独立工作区 `/home/wjr/code/code_vpwem_momart_memory`。

本轮只读检查时原训练日志为533000步，loss约0.0693003；该日志是本轮检查时的快照，不代表后续实时状态。已有500k checkpoint；不能把“working”解释成1M已经训练完或最终成功率已经验收。

`baselines/momart_boxed_working_20261030/source_manifest.json`保存70份源文件SHA256、Python及主要依赖版本，逐文件与服务器核对一致。原21项检查通过，包括真实十数据集的动作/其他模态不变、episode边界和画框处理。CPU严格加载原500k model及EMA全部参数成功；合成观测下50次DDPM得到有限 `[1,9,10]` 动作。权重、数据、缓存、日志输出及凭据没有加入Git。

原代码的离线检测/在线tracker差异，以及reset重复补帧的问题，作为历史基线限制保留；不会在固定标签上暗改，再把旧成绩说成新协议成绩。首轮[可行性与创新性报告](https://feishu.cn/docx/OWlqdkloNoj0Avx6U0scFCymnjf)继续保留。

**A1/A2/A3如何保持可比较**

| 项目 | 实施方式 |
|---|---|
| 同一代码 | 三组只切换object_memory.mode；共享感知、bank、数据、编码器、优化器循环、DDPM和评测计分 |
| 相同模型结构 | 每对象1个256维token，上限8对象；三组均含同一GRU，历史窗口固定3个位置 |
| A1 | 只让当前可见对象有效；不可见对象的历史特征、age和其他状态清空并屏蔽 |
| A2 | 仅最后可靠状态有效，离屏对象仍能读取；当前2D框无效，age继续增加 |
| A3 | 每实例最近3次可靠观测及相对时间经GRU聚合，最终token数量和维度不变 |
| 容量控制 | a3_latest_control把最新状态重复3次，经同一个网络读取 |
| 参数核对 | 测试三组state_dict键、形状和每个初始参数逐值相同；训练时另写initialization.json哈希 |
| 配置核对 | object_protocol.json记录共同配置、感知和源码哈希；compare入口拒绝mode/输出目录以外的差异 |

最近三次可靠观测是首版时序基线，尚不是交互成功事件、任务阶段监督或长期关键帧选择。当前不添加额外reader。可读槽位数量随真实对象可见性/历史存在性不同是实验自变量；网络容量和最大token预算相同。

**感知与物理观测时序**

使用同一冻结GroundingDINO tiny及版本化类别词表。原红碗颜色约束仅保留在class0，其余门、把手、篮架、抽屉不继承红色过滤。主RGB仍只画最高置信红碗框；对象分支使用未画框的原图。

对象特征从固定500k EMA主RGB ResNet18严格加载，独立冻结，取layer2的128维ROIAlign特征。冻结特征源checkpoint SHA256为 `418ac07eba671fcc615b4953abd3bea06c8810570e676a8cc9ca4d38bc812239`。原始90×120坐标按stride8映射，不按带padding的特征图尺寸拉伸；包含独立坐标回归检查。缓存可以固定保存这些特征，因为该独立对象编码器不会随新策略训练更新。

采用按类别约束的确定性IoU关联，回合内稳定槽位，尚无运动预测或外观重识别。容量不足时报错，避免静默将别的对象历史覆盖到同一ID。此关联器主要提供可审计的原型，不构成真实遮挡、多同类目标、移动相机下身份保持可靠性的验收。

离线缓存逐回合按观测前缀重放，phase0查询0、8、16…，处理帧序列0、7、8、15、16…；reset的[0,0]只更新一次。训练采样任意anchor时，以8种同节奏相位增强得到对应前缀；manifest明确记录该增强，不把它称为所有训练样本都从部署phase0取样。测试中的缓存/回放查询覆盖全部回合anchor，而不是仅检查一个片段。

age使用模拟传感器时间。当前period_seconds=0.05依据本服务器robomimic `env_ig_momart.py`默认action_timestep=1/20；若环境构造覆盖该值，需改共同配置并重建缓存。对象更新仅在新sensor step发生；像素相同但时间不同仍是新观测。去噪期间不更新bank。

对象validity传到decoder attention层，空槽不会仅因填零就被当成可读条件。过去动作位置0无法读取当前bank。原条件MLP逐token处理，防止先混合信息后绕过时间mask。默认基线RGB/深度/相邻帧裁剪规则保留；独立对象ROI明确使用原图参考系，不声称这次已经重构并统一全部旧模态裁剪。

**验证结果**

最终相关检查：36 passed，6条依赖弃用警告；无测试失败或跳过。包含13项新对象记忆检查以及23项基线/缓存相关回归。

- 三组逐参数同初始化、同模型结构。
- 当前对象与持久对象边界、不可见框清零、同类多实例检测重排后槽位保留。
- reset补帧只更新一次、相同像素的新时刻正常更新、跨环境隔离、容量超限显式失败。
- 离线所有相位缓存与在线前缀逐字段相同；未来帧修改不影响已生成前缀。
- 空槽padding真实阻断信息；当前bank变化不影响过去动作位置0。
- 动作标签、其他模态及episode末帧补齐保留；错误执行频率的缓存被拒绝。
- 完整新agent完成CPU反向更新，memory参数实际改变，EMA采样输出有限值且不改变外部bank。
- 原图box到stride8 ROIAlign坐标计算独立验证。

最终源码还对真实 `table_cleanup_to_sink_expert` 第100训练帧运行冻结感知：产生7个检测，全部128维特征有限，ROI stride8；单次CPU感知前向约34.152秒，不含首次模型加载，不是GPU实时性能。该单帧用于接口smoke，不是检测准确率或闭环任务成功率。

这次真实帧抽样出现门/篮架/抽屉等不同提示给出高度重叠的大框；没有人工真值，不能量化其误检率，也不能宣称部件覆盖已通过。正式长训练前需要检查六类提示的覆盖、小把手尺寸、身份碎片与累计对象容量，并测部署设备端到端延迟。首版IoU跟踪和上限8不能据此当作已足够。

过程中补回了原服务器的 `tools/convert_no_history_hdf5_to_zarr.py`，以支持此前已归档但依赖该工具的缓存元数据回归；隔离环境首次缺tools时的测试收集错误已解决，最终36项完整通过。开发机旧torch环境没有pytest，本次实际检查在服务器现有vpwem Python3.9/torch2.6环境进行。

**文件与复现入口**

- `momart_object_memory/README.md`：模式定义、限制、缓存/训练/评测/比较命令。
- `state.py`、`online.py`、`cache.py`：同一套实例bank与因果流；缓存不覆盖已存在目录，未完整生成时不会标记complete。
- `perception.py`：固定感知与EMA对象特征版本。
- `model.py`：共享GRU对象编码、条件打包、padding/PTP及旧位置embedding显式映射。
- `dataset.py`、`train.py`、`evaluate.py`、`compare.py`：共同训练评测入口及配置/感知/初始参数哈希核验。
- `tests/test_momart_object_memory.py`：必要机制验证。

基线训练循环只增加可注入的suite/model/batch接口，默认参数仍走原实现；原评测只增加可注入的归一化入口与p50/p95/p99端到端推理耗时输出。新评测不会重写训练初始化记录。首版拒绝自动续训，因为尚未实现完整RNG和loader位置恢复，不能将重新随机采样称作精确续训。

本轮没有生成全数据正式对象缓存、启动A123长训练或获得新成功率；原基线训练继续运行。下一步先处理感知覆盖与跟踪容量，再使用同一配置生成三组共用缓存和小规模训练验证，最后执行正式对照。
