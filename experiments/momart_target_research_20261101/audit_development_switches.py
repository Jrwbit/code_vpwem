"""Offline proxy-label switch audit. Labels never enter inference or relabelling."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from calibration import apply_calibration
from growth_data import load_growth_data,growth_partitions


def audit_switches(data,index,probability,calibrations):
    known=set(map(int,index));pred={k:apply_calibration(v,calibrations[k]) for k,v in probability.items()};events=[]
    for ep in sorted({data['rows'][i]['episode_id'] for i in known}):
        ids=sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==ep],key=lambda i:data['rows'][i]['step'])
        for left,right in zip(ids,ids[1:]):
            a,b=data['rows'][left],data['rows'][right]
            if left not in known or right not in known or not 0<b['step']-a['step']<=16:continue
            if a['label']['target']['instance_id']==b['label']['target']['instance_id']:continue
            models={}
            for key,values in probability.items():
                chosen=int(pred[key][right]);models[key]=dict(accepted=chosen,raw=int(values[right].argmax()),probability=values[right].tolist(),
                    correct=bool(chosen<8 and data['positive'][right,chosen]),calibration={q:calibrations[key].get(q,0.) for q in ['threshold','margin']})
            events.append(dict(episode=ep,left=a['id'],right=b['id'],step=b['step'],source=b['source'],demo=b['demo'],role=int(data['roles'][right]),
                target=b['label']['target'],objects=b['objects'],positive=np.flatnonzero(data['positive'][right]).tolist(),models=models))
    return events


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    data=load_growth_data(a.root);index=growth_partitions(data)['dev'];registry=json.loads((a.screen/'registry.json').read_text())
    probability={k:np.load(a.screen/(k+'_probabilities.npz'))['probability'] for k in a.names}
    events=audit_switches(data,index,probability,{k:registry[k]['calibration'] for k in a.names})
    result=dict(events=events,known_dev=len(index),registry_sha256=hashlib.sha256((a.screen/'registry.json').read_bytes()).hexdigest(),
        caveat='Offline development proxy-label audit only, no relabel or inference inputs; cannot prove human intention')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps([e for e in events if not all(x['correct'] for x in e['models'].values())],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','screen','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--names',nargs='+',required=True);main(p.parse_args())
