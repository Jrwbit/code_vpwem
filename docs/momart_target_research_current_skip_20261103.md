# MoMaRT 当前观测直接通路，2026-11-03

## 开发切换错误定位

固定时钟GRU混合漏掉的一个开发切换为 `unload_dishwasher_to_dresser_suboptimal/demo_2` 的288步（growth_95_288），弱标签从碗架切换到31:bowl。正确碗候选存在，检测分数.994，slot0；slot1为碗架。

原冻结候选碗/架概率.622166/.375186，阈值.6接受碗。GRU uniform混合25%碗/架.503962/.492785，原始排序仍选碗，但margin .2拒选；GRU transition4混合25%为.479116/.518778，原始排序已转向碗架，同样被margin .2拒选。两者均不算正确切换。仅改变该帧阈值不是可用修复，不修改既定标签或分母。

审计脚本 `audit_development_switches.py --root OUTPUT_ROOT --screen PREVIOUS_SCREEN --names blend_frozen_configuration current_gru_uniform_new0.25 current_gru_transition4_new0.25 --output NEW_AUDIT.json` 保留全部16个开发切换；标签只用于离线评价，不能作为选择器输入。该结果不能证明真实人类意图或仅凭单图确认弱标签正确。

## 搜索前协议

假设：只以64维GRU隐藏状态给候选打分，可能压制当前证据。新增直接通路，将294维当前公开context与64维隐藏状态拼接，358维输入pointer。比较同参数量、同初始化的reset控制，区分额外参数与持久记忆的贡献。没有基于标签重置、GT阶段输入或额外相机。

在前一轮完全相同的主视角固定时钟归档、108训练/24开发、训练设置和完整门槛下，重复reset/GRU × uniform/transition4 × seed100/300共8次拟合；每次原始F1/固定门槛两真实保存状态，offset0/4及原33点校准网格；四个家族各取最佳与当前冻结候选混合25/50/75%，共46条记录含2固定控制。仍须保留正确框、抓碗/携带错误、类别稳定性和及时切换，且三个开发组分别通过。只登记新架构，旧结果不重写。

运行：`benchmark_growth_recurrent.py --root OUTPUT_ROOT --main-history MAIN_HISTORY.npz --output NEW_SCREEN --device cuda:0 --holdout-prefix confirm8 --current-skip`。本轮物理GPU1；确认8无标签/候选/特征，SHA仍要求0206d3011b46833572c23da835d6595bb3c75e8950c16bc67dc204a4ab2f6acd。若开发未通过，不能消耗确认8来挑配置。

57项相关测试通过；首次测试收集失败于新增pytest装饰器漏全局import，已修正后完整重跑通过。覆盖直接通路/旧架构、reset/GRU、临时查询、因果前缀及实际训练保存。尚无本轮模型结果或闭环收益结论。

## 切换监督的时间含义

源文件 `growth_phases.json` SHA efb9a8c42f4a68cd9f9ae7e8d5243ef379716a9b13c4de595b76579df0994d4e，四帧离线旁路核验如下。

| 步 | 当前稳定持碗 | 未来8/16步稳定持碗 | 实测grasped | 特权表面距离 | 旧代理目标 |
|---|---|---|---:|---:|---|
| 280 | 否 | 否 | −1 | 4.477 cm | 碗架 |
| 288 | 否 | 是 | −1 | 2.394 cm | 预抓碗 |
| 296 | 否 | 是 | 1 | 2.340 cm | 预抓碗 |
| 304 | 是 | 否 | 1 | 2.931 cm | 持碗 |

`export_phases.py` 的held_soon检查未来8/16步，held_now要求当前和过去8步grasped均正、距离均≤3.5cm、相对位移≤2.5cm。288步的切换由预抓代理的未来稳定持碗条件触发；原先运动事件仍指向碗架。这是提前预测目标的弱监督，并非当下真实意图或接触力真值。距离为离线特权审计证据，不作为新模型输入。保留全部标签与分母，不以代理歧义豁免错误，也不把匹配旧标签直接称为人类意图正确。

## 跨结构融合的后续预登记

直接通路研究的46条记录完成后，所有新方案均失败。GRU uniform混合25%恢复16/16切换，正确1146，但类别抖动16/1103；先前无直接通路GRU uniform混合25%正确1138、切换15/16、抖动7/1073。这仅提出互补误差假设。

在下一步评分前登记：对uniform/transition4两族各取已固定的 `current_gru_*_new0.25` 概率，无直接通路/直接通路按后者25/50/75%融合，6个新配置+2个完整冻结控制=8条记录、0拟合、200校准点。全部概率源及注册表、主视角历史、数据审计SHA登记；仍要求原双V4+当前保留的三个开发组全部通过。开发集重复探索如实计数，确认8仍不访问。不能把两个研究失败转写为已证实互补。

源码 `benchmark_architecture_blend.py --root OUTPUT_ROOT --output NEW_DIRECTORY`。如开发通过，仍须冻结模型/比较器和权重SHA、严格数值/流式核验，再做独立确认；局部分数不触发部署或晋级。
