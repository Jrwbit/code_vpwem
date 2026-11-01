"""Add robot/depth geometry and compare views on the same audited subject protocol."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from data import load_group,combine,feature_view,partitions,metrics,calibrate,attach_geometry
from benchmark import train_torch,train_tree,write_json,subject_memory


def load_geometry_data(root):
    research=root/'momart_target_research_20261101';previous=root/'momart_target_selector_20261101';stage=root/'momart_stage_target_20261101'
    old=attach_geometry(load_group(root/'momart_detector_drawer_augmented_20261030',previous/'features.npz',stage/'old_phases.json','old',research/'extra_old.npz'),research/'geometry_old.npz')
    regression=attach_geometry(load_group(stage/'new_corpus',stage/'new_features.npz',stage/'new_phases.json','regression',research/'extra_new.npz'),research/'geometry_new.npz')
    return combine([old,regression])


def main(a):
    torch.set_num_threads(2);data=load_geometry_data(a.root);splits=partitions(data);index=splits['train'];a.output.mkdir(parents=True,exist_ok=False)
    views={}
    for name in ('geometry','rich_geometry','spatial'):
        raw,context=feature_view(data,name=='rich_geometry',geometry=True,spatial=name=='spatial');valid=data['valid']
        cm=raw[index][valid[index]].mean(0);cs=raw[index][valid[index]].std(0).clip(.05)
        xm=context[index].mean(0);xs=context[index].std(0).clip(.05)
        views[name]=dict(raw=raw,context=context,candidates=torch.tensor((raw-cm)/cs),contexts=torch.tensor((context-xm)/xs),cm=cm,cs=cs,xm=xm,xs=xs)
    specifications=[]
    for view in views:
        for kind in ('mlp','query','extra','boost'):
            specifications.append(dict(name=view+'_'+kind,view=view,kind=kind,width=64,weighted=True,leaf=2,leaves=7))
    specifications.extend([dict(name='geometry_soft',view='geometry',kind='mlp',width=128,weighted=True,soft_role=.5),
                           dict(name='rich_geometry_gru',view='rich_geometry',kind='gru',width=64,weighted=True)])
    registry={};probabilities={}
    def record(name,prob,info=None,checkpoint=None):
        assert np.isfinite(prob).all() and np.allclose(prob.sum(-1),1.,atol=1e-4)
        raw=prob.argmax(-1);cal=calibrate(prob,splits['dev'],data);rejected=np.where(prob.max(-1)>=cal['threshold'],raw,8)
        registry[name]=dict(name=name,info=info or {},calibration=cal,
           raw={k:metrics(raw,v,data) for k,v in splits.items() if len(v)},reject={k:metrics(rejected,v,data) for k,v in splits.items() if len(v)})
        probabilities[name]=prob;np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob)
        if checkpoint is not None:
            checkpoint['calibration']=cal;checkpoint['subject_protocol']='active_subject_v1';torch.save(checkpoint,a.output/(name+'.pt'))
        write_json(a.output/'registry.json',registry);write_json(a.output/'status.json',dict(stage='searching',latest=name,completed=len(registry)))
        print(json.dumps(dict(name=name,dev=registry[name]['raw']['dev'],regression=registry[name]['raw']['regression'])),flush=True)
    for column,name in [(0,'surface_min'),(1,'surface_q10'),(2,'surface_median')]:
        for temperature in (.05,.2):
            d=data['geometry_candidates'][:,:,column];valid=data['valid']&(data['geometry_candidates'][:,:,-1]>0)
            logits=np.where(valid,-d/temperature,-1e6);prob=np.zeros((len(d),9),np.float32)
            for i in range(len(d)):
                if valid[i].any():
                    scores=np.where(valid[i],np.exp(logits[i]-logits[i].max()),0);prob[i,:8]=scores/scores.sum()
                else:prob[i,8]=1.
            record(name+str(temperature),prob,dict(observed_depth_and_measured_robot_fk=True))
    for spec in specifications:
        write_json(a.output/'status.json',dict(stage='training',method=spec['name'],completed=len(registry)))
        if spec['kind'] in ('extra','boost'):prob,ckpt,info=train_tree(spec,data,views[spec['view']],splits,a.seed)
        else:prob,ckpt,info=train_torch(spec,data,views[spec['view']],splits,a.seed,a.epochs)
        record(spec['name'],prob,info,ckpt)
    ranked=sorted(registry,key=lambda name:registry[name]['raw']['dev']['f1'],reverse=True)
    names=ranked[:3];record('geometry_ensemble3',np.mean([probabilities[name] for name in names],0),dict(members=names))
    for name in ranked[:2]:
        for strength in (.5,1.):record(name+'_memory'+str(strength),subject_memory(probabilities[name],data,strength),dict(parent=name,strength=strength))
    baseline=json.loads((a.previous/'registry.json').read_text())['previous_pointer']['raw']['dev']['recall']
    def merit(name):
        r=registry[name];c=r['calibration'];met=c['precision_90_feasible'] and (c['metrics']['recall'] or 0)>=baseline
        return (met,c['metrics']['recall'] if met else r['raw']['dev']['f1'],-r['raw']['dev']['wrong_acquire'])
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],development_joint_goal_met=bool(merit(winner)[0]),
        confirmation_evaluated=False,method_count=len(registry),criterion='old/dev only, same as first screen'))
    write_json(a.output/'status.json',dict(stage='screening_complete',winner=winner,completed=len(registry),confirmation_evaluated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--seed',type=int,default=100)
    p.add_argument('--epochs',type=int,default=150);main(p.parse_args())
