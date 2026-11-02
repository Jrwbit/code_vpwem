"""Probe frozen action-policy fused observation features for subject selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from availability_features import replace_availability
from benchmark import train_torch,train_tree,write_json
from confirm import predict_component,apply_calibration
from data import partitions,metrics
from expanded import load_expanded_data,calibrate_v4,baseline_for
from policy_features import attach_policy,policy_view


def main(a):
    torch.set_num_threads(2);r=a.root/'momart_target_research_20261101'
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(method='frozen raw-action-policy observation representation',train='unchanged expanded28 episodes',dev='unchanged expanded8 episodes only',
        views=['geometry','rich_geometry'],kinds=['mlp64','extra leaf5'],seeds=[100,200,300],
        fit_count=12,probability_controls_per_fit=[0.,4.],frozen_parent_control=True,acceptance_revision=4,
        feature_sha256=hashlib.sha256(a.features.read_bytes()).hexdigest(),feature_metadata_sha256=hashlib.sha256(a.features.with_suffix('.json').read_bytes()).hexdigest(),
        confirm5_manifest_sha256=hashlib.sha256((r/'confirm5_corpus/manifest.json').read_bytes()).hexdigest(),confirmation5_loaded=False,
        caveat='Frozen policy observation features only, not predicted actions/plans. Weak labels unchanged. Upstream source-nonvalid episodes may have been seen by frozen policy.')
    write_json(a.output/'frozen_search_protocol.json',protocol)
    data=attach_policy(load_expanded_data(a.root),a.features);splits=partitions(data);train=splits['train'];registry={}
    previous=dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100'))
    baseline=baseline_for(predict_component(previous,data,a.root),splits['dev'],data)
    def record(name,prob,component):
        assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
        cal=calibrate_v4(prob,splits['dev'],data,baseline);pred=apply_calibration(prob,cal)
        registry[name]=dict(name=name,component=component,calibration=cal,
            raw={key:metrics(prob.argmax(-1),index,data) for key,index in splits.items() if len(index)},
            reject={key:metrics(pred,index,data) for key,index in splits.items() if len(index)})
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob)
        write_json(a.output/'registry.json',registry)
        write_json(a.output/'status.json',dict(stage='fitting',completed=len(registry),latest=name))
        print(name,cal['joint_goal_met'],cal['metrics']['correct'],cal['metrics']['wrong_acquire'],flush=True)
    parent=json.loads((r/'frozen_for_confirm4/selected.json').read_text())['result']['component']
    record('frozen_parent_control',predict_component(parent,data,a.root),parent)
    for name in protocol['views']:
        raw,context=policy_view(data,name);valid=data['valid']
        cm=raw[train][valid[train]].mean(0);cs=raw[train][valid[train]].std(0).clip(.05)
        xm=context[train].mean(0);xs=context[train].std(0).clip(.05)
        view=dict(raw=raw,context=context,candidates=torch.tensor((raw-cm)/cs),contexts=torch.tensor((context-xm)/xs),cm=cm,cs=cs,xm=xm,xs=xs)
        for kind in ['mlp','extra']:
            for seed in protocol['seeds']:
                label=f'{name}_policy_{kind}_seed{seed}'
                spec=dict(name=label,view=name,policy_features_path=str(a.features),kind=kind,width=64,leaf=5,weighted=True)
                if kind=='extra':prob,ckpt,info=train_tree(spec,data,view,splits,seed)
                else:prob,ckpt,info=train_torch(spec,data,view,splits,seed,150)
                torch.save(ckpt,a.output/(label+'.pt'));write_json(a.output/(label+'_training.json'),info)
                component=dict(kind='parent',name=label,screen=str(a.output))
                for offset in protocol['probability_controls_per_fit']:
                    adjusted=replace_availability(prob,prob[:,:8].sum(-1),0.,offset)
                    wrapped=dict(kind='availability',parent=component,model=None,mix=0.,offset=offset)
                    record(label+'_bias'+str(offset),adjusted,wrapped)
    def merit(name):
        cal=registry[name]['calibration'];return cal['joint_goal_met'],cal['metrics']['recall'] if cal['joint_goal_met'] else cal['metrics']['f1'],cal['metrics']['precision']
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=12,
        selection_partition='expanded/dev only',acceptance_revision=4,confirmation5_evaluated=False,
        parent_control_retained=winner=='frozen_parent_control'))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner))
    print(json.dumps(dict(winner=winner,calibration=registry[winner]['calibration']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','features','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
