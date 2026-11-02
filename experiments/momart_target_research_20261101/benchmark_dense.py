"""Dense original train/dev trajectories, with per-role recall floors and seed controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from benchmark import train_torch,train_tree,write_json
from benchmark_geometry import load_geometry_data
from data import load_group,attach_geometry,combine,partitions,feature_view,metrics
from confirm import predict_component,apply_calibration
from refine import role_breakdown


def subset(data,index):
    return {key:([value[i] for i in index] if key=='rows' else value if key=='tasks' else value[index]) for key,value in data.items()}


def load_dense_data(root):
    r=root/'momart_target_research_20261101';old=load_geometry_data(root)
    keep=np.array([i for i,row in enumerate(old['rows']) if not(row['tag']=='old' and row['split'] in ['train','dev'])],int)
    dense=attach_geometry(load_group(r/'dense_corpus',r/'dense_features.npz',r/'dense_phases.json','old',r/'extra_dense.npz'),r/'geometry_dense.npz')
    return combine([dense,subset(old,keep)])


def calibrate_v3(prob,index,data,baseline):
    raw=prob.argmax(-1);confidence=prob.max(-1);order=np.sort(prob,axis=-1);margin=order[:,-1]-order[:,-2];points=[]
    floors={role:max(0.,(value['recall'] or 0)-.05) for role,value in baseline['roles'].items() if value['visible']>=10}
    for threshold in [0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98]:
        for gap in [0.,.1,.2]:
            pred=np.where((confidence>=threshold)&(margin>=gap),raw,8);m=metrics(pred,index,data);roles=role_breakdown(pred,index,data)
            covered=all(value['correct']>=1 for value in roles.values() if value['visible']>=3)
            retained=all((roles[role]['recall'] or 0)>=floor for role,floor in floors.items())
            if m['selected']>=10:points.append(dict(threshold=threshold,margin=gap,metrics=m,roles=roles,critical_roles_covered=covered,role_recall_retained=retained))
    feasible=[v for v in points if (v['metrics']['precision'] or 0)>=.9 and (v['metrics']['recall'] or 0)>=baseline['metrics']['recall'] and v['critical_roles_covered'] and v['role_recall_retained']]
    if feasible:chosen=max(feasible,key=lambda v:(v['metrics']['recall'],v['metrics']['precision'],-v['threshold'],-v['margin']))
    else:
        covered=[v for v in points if v['critical_roles_covered'] and v['role_recall_retained']]
        chosen=max(covered or points,key=lambda v:(v['metrics']['f1'],v['metrics']['recall'] or 0,-v['threshold'],-v['margin']))
    return dict(chosen,joint_goal_met=bool(feasible),baseline_recall=baseline['metrics']['recall'],role_recall_floors=floors,acceptance_revision=3)


def main(a):
    torch.set_num_threads(2)
    if a.expanded:
        from expanded import load_expanded_data,calibrate_v4,baseline_for
        data=load_expanded_data(a.root)
    else:data=load_dense_data(a.root)
    splits=partitions(data);train=splits['train'];a.output.mkdir(parents=True,exist_ok=False)
    research=a.root/'momart_target_research_20261101'
    write_json(a.output/'frozen_search_protocol.json',dict(protocol='expanded_examined_cohorts_v1' if a.expanded else 'dense_train_dev_v1',acceptance_revision=4 if a.expanded else 3,
        former_confirmations='1: one episode/task train, one dev; 2: all train; historical reports archived, no longer holdout' if a.expanded else 'not used for fitting',
        priority='V4: strict acquisition wrong-class reduction if baseline >0; carry errors no worse; class chatter and stale/correct subject changes no worse' if a.expanded else 'V3 overall and role recall',
        precision=.9,recall='at least frozen previous pointer on same dense dev',roles='>=1 correct if >=3 visible; per-role recall >= baseline minus .05 if >=10 visible',
        seeds=[100,200,300],views=['base','rich','geometry','rich_geometry','spatial','mask_small','mask_spatial','mask_full'] if a.expanded else ['base','rich','geometry','rich_geometry','spatial'],kinds=['mlp64','query64','extra2','boost7','extra5 for mask views','spatial/rich_geometry/mask_spatial GRU64'] if a.expanded else ['mlp64','query64','extra2','boost7','spatial/rich_geometry GRU64'],
        holdout='confirmation3 excluded; cohorts 1/2 explicitly promoted to train/dev' if a.expanded else 'confirmation2 prepared but not loaded; original train/dev membership',
        counts={k:len(v) for k,v in splits.items()},episodes={k:sorted({data['rows'][i]['episode_id'] for i in v}) for k,v in splits.items()}))
    parent=dict(kind='parent',name='previous_pointer',screen=str(research/'screen_seed100'));p=predict_component(parent,data,a.root)
    baseline=baseline_for(p,splits['dev'],data) if a.expanded else dict(metrics=metrics(p.argmax(-1),splits['dev'],data),roles=role_breakdown(p.argmax(-1),splits['dev'],data))
    registry={};probabilities={};groups={}
    def record(name,prob,component):
        cal=calibrate_v4(prob,splits['dev'],data,baseline) if a.expanded else calibrate_v3(prob,splits['dev'],data,baseline);pred=apply_calibration(prob,cal)
        registry[name]=dict(name=name,component=component,calibration=cal,raw={k:metrics(prob.argmax(-1),v,data) for k,v in splits.items() if len(v)},reject={k:metrics(pred,v,data) for k,v in splits.items() if len(v)})
        probabilities[name]=prob;np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob);write_json(a.output/'registry.json',registry)
    record('previous_pointer',p,parent)
    for screen in ['refine_seed100','temporal_seed100']:
        selection=json.loads((research/screen/'selected.json').read_text());item=selection['result'];record('frozen_'+screen,predict_component(item['component'],data,a.root),item['component'])
    for name in (['base','rich','geometry','rich_geometry','spatial','mask_small','mask_spatial','mask_full'] if a.expanded else ['base','rich','geometry','rich_geometry','spatial']):
        if name.startswith('mask_'):
            from mask_benchmark import mask_view
            raw,context=mask_view(data,name)
        else:raw,context=feature_view(data,name in ['rich','rich_geometry'],name in ['geometry','rich_geometry','spatial'],name=='spatial');mask=data['valid']
        cm=raw[train][mask[train]].mean(0);cs=raw[train][mask[train]].std(0).clip(.05);xm=context[train].mean(0);xs=context[train].std(0).clip(.05)
        view=dict(raw=raw,context=context,candidates=torch.tensor((raw-cm)/cs),contexts=torch.tensor((context-xm)/xs),cm=cm,cs=cs,xm=xm,xs=xs)
        specs=[dict(kind='mlp',width=64),dict(kind='query',width=64),dict(kind='extra',leaf=2),dict(kind='boost',leaves=7)]
        if name.startswith('mask_'):specs.append(dict(kind='extra',leaf=5))
        if name in ['spatial','rich_geometry','mask_spatial']:specs.append(dict(kind='gru',width=64))
        for spec in specs:
            family=name+'_'+spec['kind']+(str(spec['leaf']) if 'leaf' in spec else '');groups[family]=[]
            for seed in [100,200,300]:
                label=family+'_seed'+str(seed);s=dict(spec,name=label,view=name,weighted=True)
                if spec['kind'] in ['extra','boost']:prob,ckpt,info=train_tree(s,data,view,splits,seed)
                else:prob,ckpt,info=train_torch(s,data,view,splits,seed,150)
                ckpt['subject_protocol']='active_subject_v1';torch.save(ckpt,a.output/(label+'.pt'))
                record(label,prob,dict(kind='parent',name=label,screen=str(a.output)));groups[family].append(label)
            print(family,len(registry),flush=True)
    for family,names in groups.items():record(family+'_ensemble3',np.mean([probabilities[n] for n in names],0),dict(kind='blend',members=[registry[n]['component'] for n in names],weights=[1/3]*3))
    def merit(name):
        r=registry[name];c=r['calibration'];return (c['joint_goal_met'],c['metrics']['recall'] if c['joint_goal_met'] else c['metrics']['f1'],-r['raw']['dev']['wrong_acquire'])
    winner=max(registry,key=merit);write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),selection_partition='expanded/dev only; examined cohorts 1/2 promoted' if a.expanded else 'dense old/dev only',confirmation_evaluated=False,acceptance_revision=4 if a.expanded else 3))
    print(json.dumps(dict(winner=winner,calibration=registry[winner]['calibration']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expanded',action='store_true');main(p.parse_args())
