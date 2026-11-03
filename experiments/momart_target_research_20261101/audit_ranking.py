"""Paired control reproducibility and unchanged-dev error taxonomy after ranking."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from benchmark import write_json
from calibration import apply_calibration
from growth_data import load_growth_data,growth_partitions


def error_counts(pred,indices,data):
    counts=dict(correct=0,wrong_class=0,same_class_nonmatching_box=0,available_rejected=0,unavailable_rejected=0,bowl_to_rack=0)
    for i in indices:
        slot=int(pred[i])
        if slot==8:
            counts['available_rejected' if data['positive'][i].any() else 'unavailable_rejected']+=1
        elif data['positive'][i,slot]:counts['correct']+=1
        else:
            gold=data['rows'][i]['label']['target']['class_id'];observed=data['rows'][i]['objects'][slot]['class_id']
            counts['wrong_class' if observed!=gold else 'same_class_nonmatching_box']+=1
            counts['bowl_to_rack']+=int(gold==0 and observed==3)
    return counts


def main(a):
    r=a.root/'momart_target_research_20261101';source=r/'growth_ranking_v4_seed100_300'
    if json.loads((source/'status.json').read_text())['stage']!='complete':raise ValueError('Search incomplete')
    data=load_growth_data(a.root);splits=growth_partitions(data);controls=[]
    for loss in ['uniform','transition4']:
        for seed in [100,300]:
            new=source/f'geometry_ranking0.0_{loss}_seed{seed}.pt'
            old=r/'growth_history_v4_seed100_300'/f'geometry_history_none_{loss}_seed{seed}.pt'
            n=torch.load(new,map_location='cpu',weights_only=False);o=torch.load(old,map_location='cpu',weights_only=False)
            delta=max(float((n['state'][key]-o['state'][key]).abs().max()) for key in n['state'])
            controls.append(dict(loss=loss,seed=seed,max_parameter_difference=delta,best_epoch=n['best_epoch'],reference_best_epoch=o['best_epoch']))
    if any(c['max_parameter_difference']!=0 or c['best_epoch']!=c['reference_best_epoch'] for c in controls):raise ValueError('Paired CE control failed to reproduce')
    records={}
    for directory in [source,r/'growth_ranking_blends_v4']:
        registry=json.loads((directory/'registry.json').read_text())
        for name,value in registry.items():
            probability=np.load(directory/(name+'_probabilities.npz'))['probability'];pred=apply_calibration(probability,value['calibration'])
            records[directory.name+'/'+name]={key:error_counts(pred,splits[key],data) for key in ['dev','old_dev','new_dev']}
    report=dict(controls=controls,records=records,caveat='Known unchanged proxy labels only. Same-class nonmatching includes poor localization and cannot alone establish an identity error. All class errors include unavailable-target rows; bowl_to_rack includes all roles, not acquisition alone. No confirmation labels loaded.')
    write_json(a.output,report);print(json.dumps(dict(controls=controls,records=len(records)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);main(p.parse_args())
