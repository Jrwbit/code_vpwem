"""Autonomous intent query before same-class instance grounding.

Reuses train-only role heads, compares hard queries on the registered dev set.
No contact audit, label, object GT, or future action enters the query predictor.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import torch
from benchmark import write_json
from confirm import predict_component,apply_calibration
from data import partitions,metrics
from expanded import load_expanded_data,calibrate_v4,baseline_for
from query_grounding import ground_query
from temporal import CLASS_ROLE,role_features


def query_beliefs(emissions,rows,tasks,rates,priors,temporal):
    belief=np.zeros_like(emissions)
    for ep in sorted({r['episode_id'] for r in rows}):
        indices=sorted([i for i,r in enumerate(rows) if r['episode_id']==ep],key=lambda i:rows[i]['step'])
        state=priors[tasks.index(rows[indices[0]]['task'])].copy();last=None
        for i in indices:
            if temporal:
                if last is not None:state=state@expm(rates*min(5.,(rows[i]['step']-last)*.05))
                state=state*np.clip(emissions[i],.01,1);state/=state.sum()
            else:state=emissions[i].copy()
            belief[i]=state;last=rows[i]['step']
    result=np.zeros((len(rows),6),np.float32)
    for cls,roles in CLASS_ROLE.items():result[:,cls]=belief[:,roles].sum(-1)
    return result


def hard_query_distribution(probability,posterior,classes,valid,availability):
    query=posterior.argmax(-1)
    ranked=ground_query(probability,classes,valid,query,keep_null=False)
    q=probability[:,:8].sum(-1)
    if availability=='product':q=q*posterior.max(-1)
    elif availability!='parent':raise ValueError(availability)
    missing=ranked[:,8]>0;q=np.where(missing,0,q)
    result=ranked.copy();result[:,:8]*=q[:,None];result[:,8]=1-q
    return result


def predict_query(component,data,root):
    ckpt=torch.load(component['model'],map_location='cpu',weights_only=False)
    model=ckpt['model'];emission=np.full((len(data['rows']),5),.001,np.float32)
    emission[:,model.classes_]=model.predict_proba(role_features(data,ckpt['view']));emission/=emission.sum(-1,keepdims=True)
    posterior=query_beliefs(emission,data['rows'],data['tasks'],ckpt['rates'],ckpt['priors'],component['temporal'])
    parent=predict_component(component['parent'],data,root)
    return hard_query_distribution(parent,posterior,data['candidates'][:,:,133:].argmax(-1),data['valid'],component['availability'])


def main(a):
    torch.set_num_threads(2);r=a.root/'momart_target_research_20261101'
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'frozen_search_protocol.json',dict(method='hard autonomous intent class query then same-class grounding',
        heads='8 frozen expanded-train-only role heads',views=['spatial','rich_geometry','mask_spatial','mask_full'],kinds=['extra','boost'],
        temporal=[False,True],availability=['parent','product'],configurations=32,train='unchanged28 episodes; no fitting',dev='unchanged8 episodes only',
        labels='historical weak labels unchanged; carrier conflicts remain included; not a supervision repair or true intent benchmark',
        confirm5_manifest_sha256=__import__('hashlib').sha256((r/'confirm5_corpus/manifest.json').read_bytes()).hexdigest(),
        acceptance_revision=4,confirmation5_loaded=False))
    data=load_expanded_data(a.root);splits=partitions(data);registry={}
    parent=json.loads((r/'frozen_for_confirm4/selected.json').read_text())['result']['component']
    baseline_component=dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100'))
    baseline=baseline_for(predict_component(baseline_component,data,a.root),splits['dev'],data)
    # Compare all results with the unchanged available-parent control as well.
    parentprob=predict_component(parent,data,a.root)
    write_json(a.output/'parent_control.json',dict(calibration=calibrate_v4(parentprob,splits['dev'],data,baseline)))
    for view in ['spatial','rich_geometry','mask_spatial','mask_full']:
        for kind in ['extra','boost']:
            path=r/'temporal_expanded_v4_seed100'/(view+'_'+kind+'_role.pt')
            for temporal in [False,True]:
                for availability in ['parent','product']:
                    name=f'{view}_{kind}_time{int(temporal)}_{availability}'
                    component=dict(kind='hard_query',parent=parent,model=str(path),temporal=temporal,availability=availability)
                    prob=predict_query(component,data,a.root)
                    assert np.isfinite(prob).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
                    cal=calibrate_v4(prob,splits['dev'],data,baseline);pred=apply_calibration(prob,cal)
                    registry[name]=dict(name=name,component=component,calibration=cal,
                        raw={key:metrics(prob.argmax(-1),index,data) for key,index in splits.items() if len(index)},
                        reject={key:metrics(pred,index,data) for key,index in splits.items() if len(index)})
                    np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob)
                    write_json(a.output/'registry.json',registry)
                    print(name,cal['joint_goal_met'],cal['metrics']['correct'],cal['metrics']['wrong_acquire'],flush=True)
    def merit(name):
        c=registry[name]['calibration'];return c['joint_goal_met'],c['metrics']['recall'] if c['joint_goal_met'] else c['metrics']['f1'],c['metrics']['precision']
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),selection_partition='expanded/dev only',
        acceptance_revision=4,confirmation5_evaluated=False))
    print(json.dumps(dict(winner=winner,calibration=registry[winner]['calibration']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
