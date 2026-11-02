"""Mask-restricted observed depth; train/dev selection with a fixed three-seed grid."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from benchmark import train_torch,train_tree,write_json
from benchmark_geometry import load_geometry_data
from data import partitions,feature_view,metrics
from refine import calibrate_v2
from confirm import apply_calibration


def attach_masks(data,paths):
    by_id={}
    for path in paths:
        meta=json.loads(path.with_suffix('.json').read_text());array=np.load(path,allow_pickle=False)['candidates']
        assert len(meta['ids'])==len(array)
        for key,value in zip(meta['ids'],array):
            if key in by_id:raise ValueError('Duplicate mask feature id: '+key)
            by_id[key]=value
    data['mask_candidates']=np.stack([by_id[r['id']] for r in data['rows']]);return data


def mask_view(data,name):
    if name=='mask_full':
        c,x=feature_view(data,True,True);x=x[:,:-c.shape[-1]]
    else:
        c=data['candidates'][:,:,128:];x=np.c_[data['context'][:,256:],data['geometry_context']]
        if name=='mask_spatial':c=np.concatenate([c,data['geometry_candidates']],-1)
        elif name!='mask_small':raise ValueError(name)
    c=np.concatenate([c,data['mask_candidates']],-1);m=data['valid'][:,:,None]
    return c.astype(np.float32),np.c_[x,(c*m).sum(1)/m.sum(1).clip(1)].astype(np.float32)


def main(a):
    torch.set_num_threads(2);research=a.root/'momart_target_research_20261101'
    data=attach_masks(load_geometry_data(a.root),[research/'mask_old.npz',research/'mask_new.npz']);splits=partitions(data);train=splits['train'];a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'frozen_search_protocol.json',dict(views=['mask_small','mask_spatial','mask_full'],seeds=[100,200,300],
        kinds=['mlp64','mlp128','query64','query128','boost7','extra2','extra5'],selection='original train/dev only; first confirmation already examined, diagnostic regression only'))
    baseline=json.loads((research/'screen_seed100/registry.json').read_text())['previous_pointer']['raw']['dev']['recall']
    registry={};probabilities={};groups={}
    def record(name,prob,component):
        cal=calibrate_v2(prob,splits['dev'],data,baseline);pred=apply_calibration(prob,cal)
        registry[name]=dict(name=name,component=component,calibration=cal,raw={k:metrics(prob.argmax(-1),v,data) for k,v in splits.items() if len(v)},reject={k:metrics(pred,v,data) for k,v in splits.items() if len(v)})
        probabilities[name]=prob;np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob);write_json(a.output/'registry.json',registry)
    for view_name in ['mask_small','mask_spatial','mask_full']:
        raw,context=mask_view(data,view_name);mask=data['valid'];cm=raw[train][mask[train]].mean(0);cs=raw[train][mask[train]].std(0).clip(.05)
        xm=context[train].mean(0);xs=context[train].std(0).clip(.05)
        view=dict(raw=raw,context=context,candidates=torch.tensor((raw-cm)/cs),contexts=torch.tensor((context-xm)/xs),cm=cm,cs=cs,xm=xm,xs=xs)
        specs=[dict(kind=kind,width=width) for kind in ['mlp','query'] for width in [64,128]]+[dict(kind='boost',leaves=7),dict(kind='extra',leaf=2),dict(kind='extra',leaf=5)]
        for spec in specs:
            family=f"{view_name}_{spec['kind']}{spec.get('width',spec.get('leaf',spec.get('leaves')))}";groups[family]=[]
            for seed in [100,200,300]:
                name=family+'_seed'+str(seed);s=dict(spec,name=name,view=view_name,weighted=True)
                if s['kind'] in ['extra','boost']:prob,ckpt,info=train_tree(s,data,view,splits,seed)
                else:prob,ckpt,info=train_torch(s,data,view,splits,seed,150)
                ckpt['subject_protocol']='active_subject_v1';torch.save(ckpt,a.output/(name+'.pt'))
                record(name,prob,dict(kind='parent',name=name,screen=str(a.output)));groups[family].append(name)
            print(family,len(registry),flush=True)
    for family,names in groups.items():record(family+'_ensemble3',np.mean([probabilities[n] for n in names],0),dict(kind='blend',members=[registry[n]['component'] for n in names],weights=[1/3]*3))
    def merit(name):
        r=registry[name];c=r['calibration'];return (c['joint_goal_met'],c['metrics']['recall'] if c['joint_goal_met'] else r['raw']['dev']['f1'],-r['raw']['dev']['wrong_acquire'])
    winner=max(registry,key=merit);write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),selection_partition='old/dev only',confirmation_evaluated=False))
    print(json.dumps(dict(winner=winner,calibration=registry[winner]['calibration']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
