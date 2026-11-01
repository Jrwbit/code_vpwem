"""Diverse ensembles and target-independent box quality; selection uses dev only."""
import argparse
import json
from pathlib import Path
import itertools
import numpy as np
import torch
from data import partitions,metrics
from benchmark import write_json
from benchmark_geometry import load_geometry_data


def quality_features(data):
    return np.concatenate([data['candidates'],data['extra_candidates'],data['geometry_candidates']],-1)


def multiply_quality(probability,quality,strength=1.):
    result=probability.copy();factor=np.clip(quality,0,1)**strength
    before=result[:,:8].copy();result[:,:8]*=factor
    result[:,8]+=(before-result[:,:8]).sum(-1)
    return result


def role_breakdown(pred,index,data):
    result={}
    for role in range(5):
        sub=np.asarray([i for i in index if data['roles'][i]==role],dtype=int)
        value=metrics(pred,sub,data)
        result[str(role)]={key:value[key] for key in ('samples','visible','selected','correct','precision','recall')}
    return result


def calibrate_v2(probability,index,data,baseline_recall):
    raw=probability.argmax(-1);confidence=probability.max(-1)
    ordered=np.sort(probability,axis=-1);margin=ordered[:,-1]-ordered[:,-2];points=[]
    for threshold in [0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98]:
        for gap in [0.,.1,.2]:
            pred=np.where((confidence>=threshold)&(margin>=gap),raw,8)
            value=metrics(pred,index,data);roles=role_breakdown(pred,index,data)
            coverage=all(v['correct']>=1 for v in roles.values() if v['visible']>=3)
            if value['selected']>=10:points.append(dict(threshold=threshold,margin=gap,metrics=value,roles=roles,critical_roles_covered=coverage))
    overall=[p for p in points if (p['metrics']['precision'] or 0)>=.9]
    feasible=[p for p in overall if (p['metrics']['recall'] or 0)>=baseline_recall and p['critical_roles_covered']]
    if feasible:chosen=max(feasible,key=lambda p:(p['metrics']['recall'],p['metrics']['precision'],-p['threshold'],-p['margin']))
    else:
        covered=[p for p in points if p['critical_roles_covered']]
        chosen=max(covered or points,key=lambda p:(p['metrics']['f1'],p['metrics']['recall'] or 0,-p['threshold'],-p['margin'])) if points else dict(threshold=0.,margin=0.,metrics=metrics(raw,index,data),roles=role_breakdown(raw,index,data),critical_roles_covered=False)
    return dict(chosen,joint_goal_met=bool(feasible),overall_precision90_feasible=bool(overall),baseline_recall=baseline_recall,acceptance_revision=2)


def quality_labels(data,root):
    refs={}
    for corpus in [root/'momart_detector_drawer_augmented_20261030',root/'momart_stage_target_20261101/new_corpus']:
        for r in json.loads((corpus/'annotations_all.json').read_text())['annotations']:refs[r['id']]=r['objects']
    labels=np.zeros(data['valid'].shape,bool)
    for i,row in enumerate(data['rows']):
        for slot,obj in enumerate(row['objects']):
            a=np.asarray(obj['box'],float)
            for ref in refs[row['id']]:
                if ref['class_id']!=obj['class_id']:continue
                b=np.asarray(ref['box'],float);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
                overlap=inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter,1e-8)
                if overlap>=.5:labels[i,slot]=True;break
    return labels


def main(a):
    from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
    data=load_geometry_data(a.root);splits=partitions(data);a.output.mkdir(parents=True,exist_ok=False)
    screens=[a.first,a.geometry];all_registry={};paths={}
    for screen in screens:
        registry=json.loads((screen/'registry.json').read_text());all_registry.update(registry)
        paths.update({name:screen/(name+'_probabilities.npz') for name in registry})
    parents=['previous_pointer']
    for family,condition in [('query',lambda n:'query' in n and 'memory' not in n),('boost',lambda n:'boost' in n),('extra',lambda n:'extra' in n)]:
        names=[n for n in all_registry if condition(n)]
        parents.append(max(names,key=lambda n:all_registry[n]['raw']['dev']['f1']))
    write_json(a.output/'frozen_parent_selection.json',dict(parents=parents,source='old/dev raw F1 within distinct model families; no confirmation results'))
    probabilities={n:np.load(paths[n],allow_pickle=False)['probability'] for n in parents}
    components={n:dict(kind='parent',name=n,screen=str(paths[n].parent)) for n in parents}
    for left,right in itertools.combinations(parents,2):
        name='blend_'+left+'__'+right;probabilities[name]=(probabilities[left]+probabilities[right])/2
        components[name]=dict(kind='blend',members=[components[left],components[right]],weights=[.5,.5])
    qlabels=quality_labels(data,a.root);x=quality_features(data);train_rows=np.array([r['tag']=='old' and r['split']=='train' for r in data['rows']])
    mask=data['valid'];train_mask=mask&train_rows[:,None]
    weights=np.repeat((1/mask.sum(-1).clip(1))[:,None],8,axis=1)[train_mask]
    qualities={}
    for name in ['quality_extra','quality_boost']:
        if name.endswith('extra'):model=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=3,max_features=.5,n_jobs=2,random_state=a.seed)
        else:model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=15,l2_regularization=1.,early_stopping=False,max_bins=63,random_state=a.seed)
        model.fit(x[train_mask],qlabels[train_mask],sample_weight=weights)
        q=np.zeros(mask.shape,np.float32);q[mask]=model.predict_proba(x[mask])[:,list(model.classes_).index(True)];qualities[name]=q
        torch.save(dict(model=model,feature_spec='base candidate139 + wrist/depth extra11 + robot/depth geometry16',seed=a.seed),a.output/(name+'.pt'))
        np.savez_compressed(a.output/(name+'.npz'),quality=q)
    baseline=all_registry['previous_pointer']['raw']['dev']['recall'];registry={}
    def record(name,prob,component):
        cal=calibrate_v2(prob,splits['dev'],data,baseline);order=np.sort(prob,axis=-1)
        pred=np.where((prob.max(-1)>=cal['threshold'])&((order[:,-1]-order[:,-2])>=cal['margin']),prob.argmax(-1),8)
        registry[name]=dict(name=name,component=component,calibration=cal,
            raw={k:metrics(prob.argmax(-1),v,data) for k,v in splits.items() if len(v)},
            reject={k:metrics(pred,v,data) for k,v in splits.items() if len(v)})
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob)
        write_json(a.output/'registry.json',registry)
    for name,prob in probabilities.items():
        record(name,prob,components[name])
        for quality,q in qualities.items():
            for strength in (.5,1.):
                label=name+'_'+quality+'_'+str(strength)
                component=dict(kind='quality',parent=components[name],model=str(a.output/(quality+'.pt')),strength=strength)
                record(label,multiply_quality(prob,q,strength),component)
    def merit(name):
        r=registry[name];c=r['calibration']
        return (c['joint_goal_met'],c['metrics']['recall'] if c['joint_goal_met'] else r['raw']['dev']['f1'],-r['raw']['dev']['wrong_acquire'])
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],acceptance_revision=2,
        development_joint_goal_met=registry[winner]['calibration']['joint_goal_met'],method_count=len(registry),confirmation_evaluated=False))
    print(json.dumps(dict(winner=winner,calibration=registry[winner]['calibration'],methods=len(registry)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','first','geometry','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,default=100);main(p.parse_args())
