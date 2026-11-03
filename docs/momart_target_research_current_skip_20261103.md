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
