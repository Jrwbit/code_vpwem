"""Render verified MoMaRT results as checkpoint tables and percentage-only charts."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/'artifacts/momart_reorganized_20261029/evidence.json').read_text())
assets = ROOT/'docs/assets/momart_reorganized_20261029'
assets.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),
                     'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
groups = defaultdict(dict)
for source, block in data['roots'].items():
    for row in block['results']:
        assert 'error' not in row, row
        v = row['data']; success = np.asarray(v['episode_success'])
        assert len(success) == 30 and np.isin(success, (0, 1)).all()
        assert np.isclose(success.mean(), v['mean_success'])
        prefix = 'all' if source == 'nohistory_250k' else row['path'].split('/')[0]
        key = (source, prefix)
        task = v['env_name']+'_'+v['env_type']
        assert task not in groups[key]
        groups[key][task] = v
assert all(len(g) == 10 for g in groups.values())
tasks = sorted(groups[('250k','step_250000')])
names = {t: t.replace('table_cleanup_to_', '清理→').replace('table_setup_from_', '摆台←').replace('unload_dishwasher_to_dresser','洗碗机→柜台').replace('dishwasher','洗碗机').replace('dresser','柜台').replace('sink','水槽').replace('_suboptimal',' / 次优').replace('_expert',' / 专家') for t in tasks}
def series(key): return np.array([groups[key][t]['mean_success']*100 for t in tasks])
def total(key): return float(series(key).mean())
nodekeys = [('150k','boxed_no_history')]+[('intermediate',f'step_{s}000') for s in [175,200,225]]+[('250k','step_250000')]
sections = {
    '150k': [('带框·8步·s20260917',nodekeys[0]), ('带框·4步·s20260917',('150k_action4','step_150000')), ('带框+坐标/ROI·8步·s20260917',('150k','boxed_current_object_tokens'))],
    '175k': [('带框·8步·s20260917',nodekeys[1])],
    '200k': [('带框·8步·s20260917',nodekeys[2])],
    '225k': [('带框·8步·s20260917',nodekeys[3])],
    '250k': [('无框·8步·s20260917',('nohistory_250k','all')),('带框·8步·s20260917',nodekeys[4])]
        +[(f'带框·8步·s100·R{i}',('repeats',f'seed100_repeat{i}')) for i in [1,2,3]]
        +[(f'带框·8步·s{s}',('repeats',f'seed{s}')) for s in [200,300]]
        +[(f'带框·4步·s{s}',('250k_action4',f'seed{s}')) for s in [100,200,300]],
    '1M': [(f'{label}·8步·s20260917',('1m',method)) for method,label in [('no_history','无框No-History'),('vpwem','VPWEM'),('direct','Direct'),('direct_token2','Direct2')]],
}
assert {key for configs in sections.values() for _, key in configs} == set(groups), 'Unlisted experiment configuration'
def save(fig,name):
    fig.savefig(assets/f'{name}.png',dpi=150,bbox_inches='tight');fig.savefig(assets/f'{name}.svg',bbox_inches='tight');plt.close(fig)
def bars(ax, labels, values, title, colors=None):
    rects=ax.bar(np.arange(len(values)),values,color=colors or '#3977ad',width=.65)
    ax.set_xticks(np.arange(len(labels)),labels,rotation=15,ha='right');ax.set_ylim(0,100)
    ax.set_ylabel('总成功率（%）');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.bar_label(rects,labels=[f'{v:.2f}%' for v in values],padding=3,fontsize=9)
fig,ax=plt.subplots(figsize=(9,4))
bars(ax,['150k','175k','200k','225k','250k'],[total(k) for k in nodekeys],'带框 No-History：不同checkpoint（8步，eval seed20260917）')
save(fig,'checkpoint_totals')
fig,axes=plt.subplots(1,3,figsize=(17,4.7))
for ax,step in zip(axes,['150k','250k','1M']):
    selected=sections[step][:3] if step=='150k' else sections[step][:2] if step=='250k' else sections[step]
    labels=[s.split('·')[0] if step!='150k' else ['带框8步','带框4步','坐标/ROI8步'][i] for i,(s,k) in enumerate(selected)]
    bars(ax,labels,[total(k) for _,k in selected],f'{step} 同训练步数对照 / seed20260917')
fig.suptitle('不同方法/训练预算分组展示；同训练步数不代表所有配置已控制',fontsize=14)
fig.tight_layout();save(fig,'method_totals')
fig,ax=plt.subplots(figsize=(8,4.5));pos=np.arange(3)
for offset,steps,color in [(-.19,8,'#3977ad'),(.19,4,'#dc8d36')]:
    keys=[('repeats','seed100_repeat1') if s==100 else ('repeats',f'seed{s}') for s in [100,200,300]] if steps==8 else [('250k_action4',f'seed{s}') for s in [100,200,300]]
    vals=[total(k) for k in keys];r=ax.bar(pos+offset,vals,.36,label=f'执行{steps}步',color=color)
    ax.bar_label(r,labels=[f'{v:.2f}%' for v in vals],padding=3)
ax.set_xticks(pos,['seed100','seed200','seed300']);ax.set_ylim(0,100);ax.set_ylabel('总成功率（%）');ax.set_title('同一250k带框权重：4步与8步');ax.legend();save(fig,'action_steps')
fig,axes=plt.subplots(5,2,figsize=(13,17),sharey=True)
for i,(ax,task) in enumerate(zip(axes.flat,tasks)):
    vals=[series(k)[i] for k in nodekeys];bars(ax,['150k','175k','200k','225k','250k'],vals,names[task]);ax.set_ylabel('任务成功率（%）')
fig.suptitle('每个任务跨checkpoint对照：带框No-History / 8步 / seed20260917',fontsize=15)
fig.tight_layout(rect=[0,0,1,.98]);save(fig,'task_checkpoints')
lines=['# MoMaRT 实验结果重整：按checkpoint归并','',f'原始结果核验时间：{data["time"]}。全部表格仅显示成功率（%），不列成功次数、奖励、方差或GPU日志。每个配置包含相同十个任务配置；总成功率按全部回合汇总，各任务等量，因此也等于十项均值。',
'','同一训练步数的结果归入一张表；不同权重仍单列方法标签，不能把“同为250k”当作同一个权重。s表示评测seed；带框模型训练seed均为100。所有历史快照和失败过程保留在原审计报告，本页只展示当前完整成绩。',
'','## 总体柱状图','', '![不同checkpoint总成功率](assets/momart_reorganized_20261029/checkpoint_totals.png)','',
'![同训练步数的方法对照](assets/momart_reorganized_20261029/method_totals.png)','',
'![250k四步八步对照](assets/momart_reorganized_20261029/action_steps.png)','',
'四步三seed总体60.11%，八步三seed总体58.89%，差1.22个百分点。八步seed100取R1一次参与跨seed平均，R2/R3不重复加权；三个R的成功标志完全一致。这个结果不证明显著稳定优势。',
'','## 每个checkpoint的统一表格','',
'每表行是任务，列是该训练步数下的全部方法/执行步数/评测seed，最后一行是总成功率。250k的R3含原GPU6八项与GPU0补齐两项。']
def table(headers,rows):
    lines.append('| '+' | '.join(headers)+' |');lines.append('|'+'|'.join(['---']*len(headers))+'|')
    lines.extend('| '+' | '.join(row)+' |' for row in rows);lines.append('')
for step,configs in sections.items():
    lines+=['',f'### {step}','']
    table(['任务 / 成功率%']+[label for label,key in configs],[[names[t]]+[f'{groups[key][t]["mean_success"]*100:.2f}' for label,key in configs] for t in tasks]+[['**总成功率**']+[f'**{total(key):.2f}**' for label,key in configs]])
lines+=['## 各任务跨checkpoint对照','','固定带框No-History、执行8步、eval seed20260917；每行可横向比较同一个任务的不同checkpoint，每列可纵向比较不同任务的对应checkpoint。', '']
table(['任务 / 成功率%','150k','175k','200k','225k','250k'],[[names[t]]+[f'{groups[k][t]["mean_success"]*100:.2f}' for k in nodekeys] for t in tasks]+[['**总成功率**']+[f'**{total(k):.2f}**' for k in nodekeys]])
lines+=['![各任务跨checkpoint柱状图](assets/momart_reorganized_20261029/task_checkpoints.png)','',
'## 阅读口径','',
'相同checkpoint表中的不同方法可能有不同数据处理、batch和输入结构；尤其1M VPWEM使用预计算embedding，其他基线使用原始图像。250k无框与带框是否完全公平仍需对照训练配置，不能仅凭步数相同归因于画框。不同seed只在各自标注的列比较；不要将1M无框与150k带框当作同预算对照。',
'','250k四步/八步使用同一带框权重、旧combined_blend检测器、相同计分规则；都未应用reset补帧修复。所有恢复结果现已完整，旧OOM及未启动状态不再作为当前结果展示。',
'','数据来源：本次重新读取所有已列原始result.npy。生成脚本：`tools/reorganize_momart_results_20261029.py`。原始证据：`artifacts/momart_reorganized_20261029/evidence.json`（不提交Git）。']
lines += ['', '飞书重整版：https://feishu.cn/docx/A4mVdzwCZovWOlxKxXNci1fynze']
(ROOT/'docs/momart_results_reorganized_20261029.md').write_text('\n'.join(lines)+'\n')
print('Verified',sum(len(g) for g in groups.values()),'results;',len(groups),'complete configurations; rendered 4 charts.')
