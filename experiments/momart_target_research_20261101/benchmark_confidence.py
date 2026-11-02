"""Audit-guided weak-label confidence ablation, keeping full evaluation denominators."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from availability_features import replace_availability
from benchmark import train_torch,write_json
from confirm import predict_component,apply_calibration
from data import partitions,feature_view,metrics
from expanded import load_expanded_data,calibrate_v4,baseline_for
from supervision_confidence import conflict_mask,downweight_training


def main(a):
    torch.set_num_threads(2);r=a.root/'momart_target_research_20261101';a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'frozen_search_protocol.json',dict(method='uniform audited carrier-conflict training confidence',
        hypothesis='Co-motion weak holding labels can be ambiguous; reduce supervision confidence without claiming another target is correct',
        factors=[1.,.5,.25],seeds=[100,200,300],kind='geometry mlp64',fit_count=9,bias_controls=[0.,4.],frozen_parent_control=True,
        train='unchanged28 episodes; only known train weights can change',dev='unchanged8 episodes, full original labels and weights',
        acceptance_revision=4,audit_sha256=hashlib.sha256(a.audit.read_bytes()).hexdigest(),
        confirmation5_loaded=False,confirm5_manifest_sha256=hashlib.sha256((r/'confirm5_corpus/manifest.json').read_bytes()).hexdigest(),
        caveat='No target relabel or frame deletion. Old proxy metrics remain; physical review flags are not human intention ground truth.'))
    data=load_expanded_data(a.root);splits=partitions(data);train=splits['train'];dev=splits['dev']
    audit=json.loads(a.audit.read_text());review=conflict_mask(data,audit,np.r_[train,dev])
    write_json(a.output/'supervision_audit.json',dict(train_known=len(train),dev_known=len(dev),train_review=int(review[train].sum()),dev_review=int(review[dev].sum()),
        training_review_roles={str(role):int((review[train]&(data['roles'][train]==role)).sum()) for role in range(5)},
        labels_unchanged=True,evaluation_frames_removed=0,review_train_ids=[data['rows'][i]['id'] for i in train if review[i]]))
    previous=dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100'))
    baseline=baseline_for(predict_component(previous,data,a.root),dev,data);registry={}
    def record(name,prob,component):
        assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
        cal=calibrate_v4(prob,dev,data,baseline);pred=apply_calibration(prob,cal)
        registry[name]=dict(name=name,component=component,calibration=cal,
            raw={key:metrics(prob.argmax(-1),index,data) for key,index in splits.items() if len(index)},
            reject={key:metrics(pred,index,data) for key,index in splits.items() if len(index)},
            dev_review=metrics(pred,dev[review[dev]],data),dev_other=metrics(pred,dev[~review[dev]],data))
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob);write_json(a.output/'registry.json',registry)
        write_json(a.output/'status.json',dict(stage='fitting',completed=len(registry),latest=name))
        print(name,cal['joint_goal_met'],cal['metrics']['correct'],cal['metrics']['wrong_acquire'],flush=True)
    parent=json.loads((r/'frozen_for_confirm4/selected.json').read_text())['result']['component']
    record('frozen_parent_control',predict_component(parent,data,a.root),parent)
    raw,context=feature_view(data,geometry=True);valid=data['valid']
    cm=raw[train][valid[train]].mean(0);cs=raw[train][valid[train]].std(0).clip(.05)
    xm=context[train].mean(0);xs=context[train].std(0).clip(.05)
    view=dict(raw=raw,context=context,candidates=torch.tensor((raw-cm)/cs),contexts=torch.tensor((context-xm)/xs),cm=cm,cs=cs,xm=xm,xs=xs)
    for factor in [1.,.5,.25]:
        weighted=downweight_training(data,train,review,factor)
        assert np.array_equal(weighted['weights'][dev],data['weights'][dev])
        for seed in [100,200,300]:
            label=f'geometry_confidence{factor}_seed{seed}'
            spec=dict(name=label,view='geometry',kind='mlp',width=64,weighted=True,training_review_factor=factor)
            prob,ckpt,info=train_torch(spec,weighted,view,splits,seed,150)
            torch.save(ckpt,a.output/(label+'.pt'));write_json(a.output/(label+'_training.json'),info)
            component=dict(kind='parent',name=label,screen=str(a.output))
            for offset in [0.,4.]:
                wrapped=dict(kind='availability',parent=component,model=None,mix=0.,offset=offset)
                record(label+'_bias'+str(offset),replace_availability(prob,prob[:,:8].sum(-1),0.,offset),wrapped)
    def merit(name):
        cal=registry[name]['calibration'];return cal['joint_goal_met'],cal['metrics']['recall'] if cal['joint_goal_met'] else cal['metrics']['f1'],cal['metrics']['precision']
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=9,selection_partition='expanded/dev only',
        acceptance_revision=4,confirmation5_evaluated=False,parent_control_retained=winner=='frozen_parent_control'))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner))
    print(json.dumps(dict(winner=winner,calibration=registry[winner]['calibration']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','audit','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
