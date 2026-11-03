"""Offline border/sliver diagnostics with exhaustive, unchanged denominators."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from data import load_group
from growth_data import load_growth_data,growth_partitions
from calibration import apply_calibration


def border_flags(objects,positive,image_hw,limit):
    height,width=image_hw
    flags=[]
    for slot in np.flatnonzero(positive):
        box=np.asarray(objects[slot]['box'],float)
        assert np.isfinite(box).all()
        thin=min(box[2]-box[0],box[3]-box[1])<=limit
        edge=box[0]<=1 or box[1]<=1 or box[2]>=width-1 or box[3]>=height-1
        flags.append(bool(thin and edge))
    # All alternate positive proposals must be edge slivers; one full proposal defeats the flag.
    return bool(flags and all(flags))


def summarize(data,index,pred,limit):
    buckets={k:dict(known=0,visible=0,available=0,correct=0,selected=0,acquire_wrong=0) for k in ['edge_sliver','other_available','no_matching_proposal']}
    examples=[]
    for i in index:
        row=data['rows'][i];positive=data['positive'][i];flag=border_flags(row['objects'],positive,[90,120],limit)
        bucket='edge_sliver' if flag else 'other_available' if positive.any() else 'no_matching_proposal'
        selected=pred[i]<8;correct=bool(selected and positive[pred[i]])
        wrong=bool(selected and data['roles'][i]==0 and row['objects'][pred[i]]['class_id']!=0)
        b=buckets[bucket];b['known']+=1;b['visible']+=int(data['visible'][i]);b['available']+=int(positive.any());b['correct']+=correct;b['selected']+=selected;b['acquire_wrong']+=wrong
        if flag:examples.append(dict(id=row['id'],episode=row['episode_id'],step=row['step'],correct=correct,slot=int(pred[i])))
    assert sum(x['known'] for x in buckets.values())==len(index)
    return dict(buckets=buckets,examples=examples)


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    r=a.root/'momart_target_research_20261101';data=load_growth_data(a.root);splits=growth_partitions(data)
    selected=json.loads((a.screen/'selected.json').read_text());registry=json.loads((a.screen/'registry.json').read_text());names=[selected['name'],'blend_frozen_configuration'];result={}
    for name in names:
        p=np.load(a.screen/(name+'_probabilities.npz'))['probability'];pred=apply_calibration(p,registry[name]['calibration'])
        result[name]={split:{str(t):summarize(data,index,pred,t) for t in [2,4,8]} for split,index in splits.items() if split in ['train','dev']}
    corpus=r/'confirm9_corpus';extra=load_group(corpus,r/'confirm9_features.npz',r/'confirm9_phases.json','examined');index=np.flatnonzero(extra['y']>=0)
    for name in ['primary','retention']:
        z=np.load(r/'confirm9_frozen_confirmation'/(name+'_predictions.npz'));assert z['ids'].tolist()==[row['id'] for row in extra['rows']]
        result['examined_confirm9_'+name]={str(t):summarize(extra,index,z['prediction'],t) for t in [2,4,8]}
    payload=dict(protocol='border_sliver_diagnostic_v1',results=result,thresholds_pixels=[2,4,8],border_distance_pixels=1,image_hw=[90,120],
        condition='All matching positive detector boxes thin and touch image boundary; no-positive queries separately counted; every known query retained',
        registry_sha256=hashlib.sha256((a.screen/'registry.json').read_bytes()).hexdigest(),
        caveat='Offline label-conditioned stratification, not an online target-visible detector, not a reason to drop errors or relabel invisible targets; no model fitting or confirmation10 input')
    a.output.write_text(json.dumps(payload,indent=2))
    print(json.dumps({k:v if k.startswith('examined') else {s:{t:x['buckets'] for t,x in vals.items()} for s,vals in v.items()} for k,v in result.items() if not k.startswith('examined')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','screen','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
