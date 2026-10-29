"""Only ten-task aggregate success rates; never emit task-level tables or plots."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--baselines', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('docs/momart_total_success_20261029.md'))
    args = parser.parse_args()
    data = json.loads(args.evidence.read_text()); baseline = json.loads(args.baselines.read_text())
    groups = defaultdict(list)
    for name, root in data['roots'].items():
        for row in root['results']:
            assert 'error' not in row
            value = row['data']; flags = value['episode_success']
            assert len(flags) == 30 and all(x in (0, 1) for x in flags)
            assert abs(sum(flags)/30 - value['mean_success']) < 1e-8
            prefix = 'all' if name == 'nohistory_250k' else row['path'].split('/')[0]
            groups[name, prefix].append(sum(flags))
    assert all(len(rows) == 10 for rows in groups.values())
    def rate(name, prefix): return sum(groups[name, prefix])/3
    plt.rcParams.update({'font.family': FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(), 'axes.unicode_minus':False, 'font.size':11})
    assets = args.output.parent/'assets/momart_total_success_20261029'; assets.mkdir(parents=True,exist_ok=True)
    def chart(name, labels, values, title):
        fig, ax = plt.subplots(figsize=(10,5)); bars=ax.bar(range(len(values)),values,color=['#677f99','#db943e','#3b86a8','#73a082'][:len(values)])
        ax.set_xticks(range(len(labels)), labels); ax.set_ylim(0,100); ax.set_ylabel('十任务总成功率（%）'); ax.set_title(title)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=4)
        fig.tight_layout();fig.savefig(assets/(name+'.png'),dpi=160);fig.savefig(assets/(name+'.svg'));plt.close(fig)
    a150=[baseline['150000']['percent'],rate('150k_action4','step_150000'),rate('150k','boxed_no_history'),rate('150k','boxed_current_object_tokens')]
    a250=[baseline['250000']['percent'],rate('250k','step_250000')]
    chart('total_150k',['No-History\n8步','带框\n4步','带框\n8步','带框＋坐标/ROI\n8步'],a150,'150k：十任务总成功率（评测seed20260917）')
    chart('total_250k_reference',['No-History\n8步','带框\n8步'],a250,'250k：十任务总成功率（评测seed20260917）')
    nh=[baseline['150000']['percent'],baseline['250000']['percent'],rate('1m','no_history')]
    chart('total_nohistory_budgets',['150k / 8步','250k / 8步','1M / 8步'],nh,'No-History：不同训练步数的十任务总成功率')
    eight=[rate('repeats','seed100_repeat1')]+[rate('repeats',f'seed{s}') for s in [200,300]]
    four=[rate('250k_action4',f'seed{s}') for s in [100,200,300]]
    fig,ax=plt.subplots(figsize=(9,5));p=np.arange(3)
    for offset,values,label,color in [(-.19,eight,'带框8步','#3b86a8'),(.19,four,'带框4步','#db943e')]:
        bars=ax.bar(p+offset,values,.36,label=label,color=color);ax.bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=4)
    ax.set_xticks(p,['seed100','seed200','seed300']);ax.set_ylim(0,100);ax.set_ylabel('十任务总成功率（%）');ax.set_title('同一250k带框权重：不同评测seed');ax.legend();fig.tight_layout()
    fig.savefig(assets/'total_250k_seeds.png',dpi=160);fig.savefig(assets/'total_250k_seeds.svg');plt.close(fig)
    lines=['# MoMaRT：仅汇总十任务总成功率','', '每个数值均为全部10个任务配置的总成功率；不展示任何单任务结果。下列表格仅列百分比，不列成功次数、奖励、方差。', '']
    def table(headers,rows):
        lines.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
        lines.extend('| '+' | '.join(row)+' |' for row in rows);lines.append('')
    def image(name): lines.extend([f'![十任务总成功率](assets/momart_total_success_20261029/{name}.png)',''])
    lines+=['## 150k：各方案总成功率','','评测seed统一为20260917。你所指的“带框带历史”按本轮确认，对应带框＋坐标/ROI版本；它没有长期历史记忆，以下使用真实名称。','']
    table(['方案','执行步数','总成功率'],[[label,step,f'{v:.2f}%'] for label,step,v in zip(['No-History（无框）','带框No-History','带框No-History','带框＋坐标/ROI'],['8','4','8','8'],a150)])
    image('total_150k')
    lines+=['## 250k：各方案与不同seed总成功率','','“—”表示没有该配置的完整评测，不是0%。不同seed不混在同一比较中。','']
    table(['方案','执行步数','seed20260917','seed100','seed200','seed300'],[
        ['No-History（无框）','8',f'{a250[0]:.2f}%','—','—','—'],
        ['带框No-History','8',f'{a250[1]:.2f}%']+[f'{v:.2f}%' for v in eight],
        ['带框No-History','4','—']+[f'{v:.2f}%' for v in four]])
    image('total_250k_reference');image('total_250k_seeds')
    lines += [f'三个不同seed等权汇总：带框8步 **{np.mean(eight):.2f}%**，带框4步 **{np.mean(four):.2f}%**。八步seed100的三次重复均为59.00%，汇总时只计一次，不重复加权。','',
        '250k带框＋坐标/ROI未找到完整评测，因此不补数、不画零值柱。','',
        '## 无框No-History：150k、250k和1M总成功率','','评测seed统一为20260917，执行8步。150k/250k来自250k训练链，1M来自另一条历史训练链；不将它们冒充同一个模型文件。','']
    table(['训练步数','总成功率'],[[label,f'{v:.2f}%'] for label,v in zip(['150k','250k','1M'],nh)]);image('total_nohistory_budgets')
    lines += ['## 必要口径说明','',
        '这里把“1M和250k”解释为两个训练步数的No-History对照。若指的是1M训练链在250k时保存的那个checkpoint，其完整评测尚未找到，不能拿另一条250k训练链替代。', '',
        '带框150k与250k方案还存在续训及学习率计划差异；历史1M无框的有效batch与250k不同。上述是实际总成绩对照，不单独证明画框或执行步长的因果效果。带框4/8步沿用同一权重和旧检测协议，均未混入reset修复。','',
        '本报告只重新汇总已完成结果，没有新启动实验。原始结果、单任务证据及审计记录仍留在历史报告中，但本页不展示分项。','',
        '飞书汇总：https://feishu.cn/docx/A4mVdzwCZovWOlxKxXNci1fynze']
    args.output.write_text('\n'.join(lines)+'\n')
    print('Only aggregate tables generated:',a150,a250,nh,eight,four)

if __name__=='__main__':main()
