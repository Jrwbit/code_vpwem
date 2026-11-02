"""Soft causal role filtering with train-only transition rates and public emissions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import expm
from benchmark_geometry import load_geometry_data
from benchmark import write_json
from data import partitions,feature_view,metrics
from refine import calibrate_v2,role_breakdown

CLASS_ROLE={0:[0,4],2:[1],3:[2],5:[3]}


def role_features(data,view):
    _,context=feature_view(data,rich=view=='rich_geometry',geometry=True,spatial=view=='spatial')
    return context


def transition_rates(data,index):
    rates=np.full((5,5),.01);np.fill_diagonal(rates,0)
    counts=np.zeros((5,5));duration=np.ones(5)
    allowed=set(index);priors=np.ones((len(data['tasks']),5))
    for i in index:priors[data['tasks'].index(data['rows'][i]['task']),data['roles'][i]]+=1
    for ep in sorted({data['rows'][i]['episode_id'] for i in index}):
        # Unknown labels break transition supervision; no bridging unlabelled intervals.
        ids=sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==ep],key=lambda i:data['rows'][i]['step'])
        for left,right in zip(ids,ids[1:]):
            gap=(data['rows'][right]['step']-data['rows'][left]['step'])*.05
            if left not in allowed or right not in allowed or gap>1.2:continue
            a,b=data['roles'][left],data['roles'][right];duration[a]+=gap
            if a!=b:counts[a,b]+=1
    rates+=counts/duration[:,None];np.fill_diagonal(rates,-rates.sum(-1))
    return rates,priors/priors.sum(-1,keepdims=True)


def soft_role_filter(probability,emissions,data,rates,priors,strength=1.,transition_scale=1.,temporal=True):
    result=probability.copy();valid=data['valid'];classes=data['candidates'][:,:,133:].argmax(-1)
    cache={}
    for ep in sorted({r['episode_id'] for r in data['rows']}):
        indices=sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==ep],key=lambda i:data['rows'][i]['step'])
        belief=priors[data['tasks'].index(data['rows'][indices[0]]['task'])].copy();last=None
        for i in indices:
            step=data['rows'][i]['step']
            if temporal:
                if last is not None:
                    dt=min(5.,(step-last)*.05)*transition_scale
                    if dt not in cache:cache[dt]=expm(rates*dt)
                    belief=belief@cache[dt]
                belief=belief*np.clip(emissions[i],.01,1);belief/=belief.sum()
            else:belief=emissions[i].copy()
            role_prior=np.full(6,.01)
            for cls,roles in CLASS_ROLE.items():role_prior[cls]=belief[roles].sum()
            result[i,:8]*=np.where(valid[i],np.clip(role_prior[classes[i]],.01,1)**strength,0)
            # Keep original null mass: role ambiguity does not manufacture another candidate.
            result[i]/=result[i].sum();last=step
    return result


def train_role(data,index,kind,view,seed):
    from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
    if kind=='extra':net=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=3,max_features=.7,n_jobs=2,random_state=seed)
    else:net=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=10,l2_regularization=1.,early_stopping=False,random_state=seed)
    x=role_features(data,view);net.fit(x[index],data['roles'][index],sample_weight=data['weights'][index])
    emission=np.full((len(x),5),.001,np.float32);emission[:,net.classes_]=net.predict_proba(x);emission/=emission.sum(-1,keepdims=True)
    return net,emission


def main(a):
    torch.set_num_threads(2);data=load_geometry_data(a.root);splits=partitions(data);a.output.mkdir(parents=True,exist_ok=False)
    research=a.root/'momart_target_research_20261101';rates,priors=transition_rates(data,splits['train'])
    write_json(a.output/'frozen_search_protocol.json',dict(method='causal soft role filter',train='original16 train episodes',selection='original4 dev episodes only',
        previous_confirmation='now examined regression; never used to choose checkpoints or thresholds',views=['spatial','geometry','rich_geometry'],
        kinds=['extra','boost'],strengths=[.25,.5,1.,2.],temporal=[False,True],parents=['previous_pointer','spatial_query']))
    parent_paths={'previous_pointer':research/'screen_seed100','spatial_query':research/'geometry_screen_seed100'}
    probabilities={name:np.load(path/(name+'_probabilities.npz'))['probability'] for name,path in parent_paths.items()}
    baseline=json.loads((research/'screen_seed100/registry.json').read_text())['previous_pointer']['raw']['dev']['recall'];registry={}
    for view in ['spatial','geometry','rich_geometry']:
        for kind in ['extra','boost']:
            net,emissions=train_role(data,splits['train'],kind,view,a.seed);checkpoint=a.output/(view+'_'+kind+'_role.pt')
            torch.save(dict(model=net,view=view,rates=rates,priors=priors,seed=a.seed),checkpoint)
            for parent,prob in probabilities.items():
                for temporal in [False,True]:
                    for strength in [.25,.5,1.,2.]:
                        name=f'{parent}_{view}_{kind}_time{int(temporal)}_s{strength}'
                        prediction=soft_role_filter(prob,emissions,data,rates,priors,strength,temporal=temporal)
                        cal=calibrate_v2(prediction,splits['dev'],data,baseline)
                        order=np.sort(prediction,axis=-1);accepted=np.where((prediction.max(-1)>=cal['threshold'])&((order[:,-1]-order[:,-2])>=cal['margin']),prediction.argmax(-1),8)
                        component=dict(kind='role_filter',parent=dict(kind='parent',name=parent,screen=str(parent_paths[parent])),model=str(checkpoint),strength=strength,temporal=temporal)
                        registry[name]=dict(name=name,component=component,calibration=cal,
                            raw={k:metrics(prediction.argmax(-1),v,data) for k,v in splits.items() if len(v)},
                            reject={k:metrics(accepted,v,data) for k,v in splits.items() if len(v)},dev_roles=role_breakdown(accepted,splits['dev'],data))
                        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prediction)
            write_json(a.output/'registry.json',registry);print(view,kind,len(registry),flush=True)
    def merit(name):
        r=registry[name];c=r['calibration']
        return (c['joint_goal_met'],c['metrics']['recall'] if c['joint_goal_met'] else r['raw']['dev']['f1'],-r['raw']['dev']['wrong_acquire'])
    winner=max(registry,key=merit);write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),selection_partition='old/dev only',confirmation_evaluated=False))
    print(json.dumps(dict(winner=winner,calibration=registry[winner]['calibration']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=100);main(p.parse_args())
