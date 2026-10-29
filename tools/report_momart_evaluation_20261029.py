"""Render the read-only 2026-10-29 snapshot; no remote mutations or rollouts."""
import json, statistics, sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text()); roots=x['roots']; groups={}
for name,g in roots.items():
 for r in g['results']:
  assert 'error' not in r, r
  a=r['data']; assert a['verified_n']==30
  assert a['episode_id']==list(range(30))
  assert a['checkpoint_step']==(1000000 if name=='1m' else 150000 if name.startswith('150k') else int(r['path'].split('step_')[1].split('/')[0]))
  assert a['eval_seed']==(int(r['path'].split('/')[0].replace('seed','').split('_')[0]) if name in ('repeats','250k_action4') else 20260917)
  if 'action4' in name:
   assert (a['execution_action_steps'],a['training_action_steps'],a['horizon'],a['obs_steps'])==(4,8,9,2)
  prefix=r['path'].split('/')[0]
  groups.setdefault((name,prefix),{})[a['env_name']+'_'+a['env_type']]=a
manifest=json.loads(roots['250k_action4']['status']['manifest.json'])
assert manifest['tracker_reset_fix'] is False
assert manifest['assignments']=={'100':0,'200':1,'300':7}
for key,g in groups.items():
 assert len(g)==(8 if key==('repeats','seed100_repeat3') else 10)
for name in ['150k_action4','250k_action4']:
 summary=json.loads(roots[name]['status']['summary.json']); assert summary['complete']
 for row in summary.get('rows',[]):
  for kind,key in [('action4',('150k_action4','step_150000')),('action8',('150k','boxed_no_history'))]:
   a=groups[key][row['task']]; assert a['verified_success']==row[kind]['successes']
 for seed,run in summary.get('runs',{}).items():
  gg=groups[(name,'seed'+seed)]
  assert sum(a['verified_success'] for a in gg.values())==run['successes']
  for row in run['rows']:
   assert gg[row['task']]['verified_success']==row['successes']
   r=next(r for r in roots[name]['results'] if r['path']==f"seed{seed}/step_250000/{row['task']}/result.npy")
   assert r['sha256']==row['result_sha256']
tasks=sorted(groups[('250k','step_250000')]); lines=[]
def p(s=''):
 lines.append(s + ('\n' if s.startswith('#') else ''))
def table(headers,rows):
 p('| '+' | '.join(headers)+' |');p('|'+'|'.join(['---']*len(headers))+'|')
 for row in rows:p('| '+' | '.join(map(str,row))+' |')
 p()
def total(g):return sum(a['verified_success'] for a in g.values()),sum(a['verified_n'] for a in g.values())
def score(g):
 s,n=total(g);return f'{s}/{n}，{100*s/n:.2f}%' if n else '—'
def counts(keys):
 return [[t]+[f"{groups[k][t]['verified_success']}/30" if t in groups[k] else '缺失' for k in keys] for t in tasks]
rkeys=[('repeats',f'seed100_repeat{i}') for i in (1,2,3)]; rr=[groups[k] for k in rkeys]; common=sorted(set.intersection(*(set(g) for g in rr)))
p('# MoMaRT 评测总对比与250k重复性验证｜2026-10-29')
p(f'\n北京时间08:00开始只读核查；原始结果快照时间：{x["time"]}。所有下列已有成绩均重新加载远端 result.npy 复算，未启动、修改或终止训练/评测，未补跑实验。\n')
p('## 结果与状态')
p('250k八步 seed100 第1、2轮均为 **177/300（59.00%）**，且逐episode成功标志完全一致。第3轮两项评测发生CUDA显存不足，只完成 **125/240（52.08%）**；该部分比例不能与完整300回合直接排名。supervisor已在00:03:07标记第一阶段失败，因此八步seed200/300没有启动。\n')
p('250k四步三seed均完成：**100：177/300（59.00%）；200：188/300（62.67%）；300：176/300（58.67%）**。目前唯一完整的同eval seed执行长度对照为seed100：四步和两轮完整八步总分相同，尚无稳定提分证据。既有eval seed20260917的250k八步 **194/300（64.67%）** 已重新核验；不能拿它直接替代seed100/200/300的八步对照。\n')
p('## 实验协议与可比性')
table(['实验组','训练/评测seed','训练步/执行块','输入与可比性'],[
['带框训练节点','100 / 20260917','150k、175k、200k、225k、250k / 8','同离线训练框、在线combined_blend；150k后保留Adam/EMA续训并延长cosine至250k'],
['250k八步复测','100 / 100重复3次，200、300','250k / 8','同权重，物理GPU3/4/6；200/300原定GPU3/4，未启动'],
['250k四步','100 / 100、200、300','250k / 4','物理GPU0/1/7；旧tracker，reset修复未应用，与八步一致'],
['150k四步 / 坐标ROI','100 / 20260917','150k / 4或8','四步是执行长度消融；ROI是另一结构，不能与250k混作同预算'],
['普通No-History、Direct、Direct2','100 / 20260917','1M / 8','无红框，raw-image，有效batch64；1M与带框150k/250k预算不同'],
['VPWEM','100 / 20260917','1M / 8','预计算embedding、任务context encoder、有效batch256，与raw-image存在混杂']])
p('带框组预测horizon=9、最近观测2帧、DDPM50、EMA、10并行环境、10配置各30回合，上限1500环境步。主RGB画1像素红框；训练为离线Grounding DINO框，评测为冻结combined_blend（max_age=3、refine_alpha=0.5）。四步仅执行动作索引1:5，training_action_steps仍为8；八步执行1:9。旧计分wrapper只回传动作块末两个info，缩短执行块同时增加观测、tracker更新和成功检查频率，不能把变化单独归因于预测质量。八步episode_steps=1504为外层计数，环境实际1500截断。\n')
p('## 逐轮完成情况')
rows=[]
for i,g in enumerate(rr,1):rows.append([f'250k八步100重复{i}',[3,4,6][i-1],score(g),f'{len(g)}/10；'+('完成' if i<3 else '2项OOM失败')])
rows += [['250k八步200',3,'0/0；无成功率','未启动，前序失败阻断'],['250k八步300',4,'0/0；无成功率','未启动，前序失败阻断']]
for seed,gpu in [(100,0),(200,1),(300,7)]:rows.append([f'250k四步{seed}',gpu,score(groups[('250k_action4',f'seed{seed}')]),'10/10完成，退出码0'])
table(['轮次/评测seed','物理GPU','成功/实际回合','状态'],rows)
p('四步queue_status、summary、completion均确认完成，最后完成时间05:34:50。八步重复1于00:00:31、重复2于00:03:07结束，均failures=0；重复3于23:13:32结束，failures=2。重复3的missing=0是队列结束字段，不能据此认定10项都有结果：原始文件实际仅8份。\n')
p('## 250k十任务对照')
p('每格为成功/实际回合；R1/R2/R3均为eval seed100的八步重复。八步200/300所有任务均未运行，因此没有可填的分项成绩。\n')
table(['任务配置','八步R1','八步R2','八步R3','四步100','四步200','四步300'],counts(rkeys+[('250k_action4',f'seed{s}') for s in [100,200,300]]))
p('## 同seed重复性与逐episode一致性')
p('三轮完整300回合成功率的均值、标准差、极差：**不可计算**，因为R3缺60回合。仅R1/R2完整轮：均值59.00%，样本标准差0.00个百分点，极差0.00个百分点；这不是三个独立seed统计。\n')
vals=[sum(g[t]['verified_success'] for t in common)/240*100 for g in rr]
p(f'三轮共同8配置、各240回合：成功数分别为{[sum(g[t]["verified_success"] for t in common) for g in rr]}；成功率均值{statistics.mean(vals):.2f}%，样本标准差{statistics.stdev(vals):.2f}个百分点，极差{max(vals)-min(vals):.2f}个百分点。只描述共同子集，不能当完整基准。\n')
rows=[]
for i,j in [(0,1),(0,2),(1,2)]:
 ts=sorted(set(rr[i])&set(rr[j])); n=len(ts)*30; eq=sum(a==b for t in ts for a,b in zip(rr[i][t]['episode_success'],rr[j][t]['episode_success']))
 samecount=sum(rr[i][t]['verified_success']==rr[j][t]['verified_success'] for t in ts)
 rows.append([f'R{i+1} vs R{j+1}',f'{samecount}/{len(ts)}',f'{eq}/{n}（{eq/n*100:.2f}%）',n-eq])
table(['对比','任务成功数相同','存储序号成功标志一致','不一致episode数'],rows)
rows=[]
for t in tasks:
 if t in common:
  v=[g[t]['verified_success']/30*100 for g in rr]; eq=sum(len(set(z))==1 for z in zip(*(g[t]['episode_success'] for g in rr)))
  rows.append([t,f'{statistics.mean(v):.2f}',f'{statistics.stdev(v):.2f}',f'{max(v)-min(v):.2f}',f'{eq}/30'])
 else: rows.append([t,'缺R3','—','—','—'])
table(['任务','三轮均值%','样本SD百分点','极差百分点','三轮标志全一致'],rows)
p('以上按各文件episode_success存储顺序逐项比较；episode_id已核对为0–29。相同序号和seed不证明模拟器初始状态、扩散噪声、并行调度或GPU浮点轨迹完全一致；没有状态快照可作严格配对认证。总成功数相等也不等于逐episode复现。统计单位是重复执行，不是独立训练或独立评测seed；不据此构造独立样本置信区间或显著性结论。\n')
p('## 训练节点与150k执行长度、ROI对照')
keys=[('150k','boxed_no_history')]+[('intermediate',f'step_{s}') for s in [175000,200000,225000]]+[('250k','step_250000'),('150k_action4','step_150000'),('150k','boxed_current_object_tokens')]
table(['版本','成功/回合/百分比','状态'],[[label,score(groups[k]),'10/10完成，原始复核'] for label,k in zip(['150k八步','175k八步','200k八步','225k八步','250k八步','150k四步','150k坐标/ROI八步'],keys)])
table(['任务','150k/8','175k/8','200k/8','225k/8','250k/8','150k/4','150k ROI'],counts(keys))
p('同eval seed20260917的曲线是59.33%→55.00%→57.00%→62.67%→64.67%，并非单调提高；250k相对150k多16次成功、+5.33个百分点。它描述已有续训路径，不是架构改动效果。150k四步相对八步多5次、+1.67个百分点；150k坐标/ROI比纯画框少18次、−6.00个百分点，均只覆盖单训练seed。\n')
p('## 跨seed与执行块对比')
a4=[total(groups[('250k_action4',f'seed{s}')])[0]/3 for s in [100,200,300]]
p(f'250k四步三独立评测seed等权均值{statistics.mean(a4):.2f}%，样本标准差{statistics.stdev(a4):.2f}个百分点，极差{max(a4)-min(a4):.2f}个百分点；总计541/900（60.11%），仅为该三seed汇总，不是900个独立训练样本。训练seed始终只有100。\n')
table(['eval seed','八步','四步','可解释差值'],[['100','R1/R2各177/300；R3不完整','177/300','相对两完整轮均0.00个百分点'],['200','未运行','188/300','无法计算'],['300','未运行','176/300','无法计算']])
p('seed100四步与R3共同8配置的比较仅作子集诊断：'+score({t:groups[('250k_action4','seed100')][t] for t in common})+'，R3为125/240；缺失两配置不补零。八步不存在完整100/200/300三seed均值；禁止把R1/R2/R3当三个seed，也不把旧取消目录中的结果借来补齐。\n')
p('## 历史1M方法：独立预算分组')
hkeys=[('1m',s) for s in ['no_history','vpwem','direct','direct_token2']]
table(['方法','原始复算','说明'],[[s,score(groups[k]),'10/10完整，checkpoint=1M，eval seed20260917'] for s,k in zip(['普通No-History','VPWEM','Direct','Direct2（Direct-token2）'],hkeys)])
table(['任务','No-History 1M','VPWEM 1M','Direct 1M','Direct2 1M'],counts(hkeys))
p('四种方法原始文件本次均可读取，分项与既有report_feishu.md一致。普通No-History 217/300（72.33%）是1M无框成绩，不能当作150k或250k画框成绩；预算与batch不等，不能据其高于64.67%推断画框有害。Direct和Direct2仅相差1次成功；VPWEM输入编码与batch不同，无法把分差归因于长期记忆机制。历史No-History后五项曾重复执行并覆盖，当前按最终每配置30回合计，未合并成60。\n')
p('## 失败、缺失与证据边界')
p('- R3缺失table_setup_from_dishwasher_expert及suboptimal，各30回合；两份日志均为torch.OutOfMemoryError，任务退出码1。错误发生在在线Grounding DINO/BERT推理路径。日志写GPU 0是进程可见设备编号，调度物理卡为6；不能据日志编号断言物理GPU0故障，也未进一步证明其他进程归属或根因。\n- 八步seed200/300因阶段屏障未启动，各缺300回合；不会自动因本报告补跑。\n- 取消的eval seed20260918/20260919以及旧seed100/200/300目录全部排除。\n- 本次表列168份可用result.npy均验证二值成功标志、30回合、均值、checkpoint、eval seed和顺序ID；共168份（28+30+10+20+10+30+40）、5040个已存储回合，不合并为独立统计样本。\n')
p('## 证据与复算方法')
p('服务器：ssh bitbot421；读取Python：/home/jrw/anaconda3/envs/vpwem/bin/python。对各result.npy的episode_success求和，以实际数组长度作分母；按env_name和env_type对齐，不依赖目录遍历顺序。150k/250k四步summary逐任务计数与原始文件一致；250k四步summary中SHA256也逐项匹配。中间节点、既有250k与重复轮读取全部gpu*_status.tsv并与原始文件对照。\n')
for name,g in roots.items():p(f'- {name}：`{g["path"]}`')
p('\n复算脚本：tools/collect_momart_evaluation_20261029.py（SSH stdin只读执行），tools/report_momart_evaluation_20261029.py（本地离线生成）。完整数组、结果哈希、状态快照保存在被Git忽略的artifacts/momart_evaluation_20261029/evidence.json；原始模型/数据不复制、不提交。\n')
p('历史解释来源：docs/momart_boxed_250k_repeats_20261028.md末节、docs/momart_boxed_250k_action4_queue_20261028.md、docs/momart_boxed_intermediate_eval_20261028.md、docs/momart_boxed_nohistory_150k_improvement_20261027.md、docs/assets/momart_main_figures_20261028/evidence.md、reports/momart_1m_four_methods/report_feishu.md及reports/momart_history4_sh2_transfer_20260920.md。本报告为2026-10-29实时结果，旧文档的“运行中”仅保留历史含义。\n')
if len(sys.argv)>2: p('新飞书文档：'+sys.argv[2]+'\n')
Path('docs/momart_evaluation_comparison_20261029.md').write_text('\n'.join(lines)+'\n')
print(f'Validated {sum(len(g["results"]) for g in roots.values())} files; report generated')
