"""Read-only remote collector. Run with SSH stdin; write stdout to ignored artifacts.

ssh bitbot421 /home/jrw/anaconda3/envs/vpwem/bin/python - < this_file.py
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime,argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--include-nohistory-250k', action='store_true')
args=parser.parse_args()
base=Path('/home/jrw/vpwem_outputs'); train=base/'momart_boxed_no_history_bs256_250k_continued_20261027/seed100'
roots={
'repeats':train/'eval_online_combined_blend_250k_seed100_triplicate_20261028',
'intermediate':train/'eval_online_combined_blend_175k_200k_225k_20261028',
'250k':train/'eval_online_combined_blend_250k',
'150k':base/'momart_boxed_online_150k_combined_blend_20261026_gpu1',
'150k_action4':base/'momart_boxed_no_history_150k_action4_20261027_gpu356',
'250k_action4':train/'eval_online_combined_blend_250k_action4_seeds100_200_300_20261028',
'1m':base/'fair_history_momart_multitask_eval_1m'}
if args.include_nohistory_250k:
 roots['nohistory_250k']=base/'fair_history_momart_no_history_bs256_250k/seed100/eval_curve_20260922/step_250000'
def conv(o):
 if isinstance(o,np.ndarray): return o.tolist()
 if isinstance(o,np.generic): return o.item()
 raise TypeError(type(o).__name__)
out={'time':datetime.datetime.now().astimezone().isoformat(),'roots':{}}
for name,root in roots.items():
 group={'path':str(root),'results':[],'status':{}}
 for p in sorted(root.rglob('result.npy')):
  try:
   a=np.load(p,allow_pickle=True).item(); s=np.asarray(a['episode_success']); assert s.ndim==1 and np.isin(s,[0,1]).all()
   assert np.isclose(s.mean(),a['mean_success']); a['verified_n']=len(s);a['verified_success']=int(s.sum())
   group['results'].append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':a})
  except Exception as e: group['results'].append({'path':str(p),'error':str(e)})
 for p in root.rglob('*'):
  if p.is_file() and (p.name in ['summary.json','completion.json','queue_status.json','status.json','manifest.json','supervisor.log'] or p.name.endswith('_status.tsv')):
   group['status'][str(p.relative_to(root))]=p.read_text(errors='replace')
 out['roots'][name]=group
print(json.dumps(out,default=conv,ensure_ascii=False))
