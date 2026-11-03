"""Main-view MLP training-loss ablation; future GT timing never enters inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from availability_features import replace_availability
from benchmark import write_json
from benchmark_growth import train_model
from benchmark_growth_blend import summary
from calibration import apply_calibration
from confirm import evaluation_gate,incumbent_evaluation_gate,predict_component
from data import feature_view,metrics
from growth_data import load_growth_data,growth_partitions
from transition_weights import training_weights
from supervision_confidence import downweight_training
from supervision_timing import far_future_mask
from checkpoint_selection import current_retention_gate,checkpoint_merit


def main(a):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=a.root/'momart_target_research_20261101';holdout=r/'confirm9_corpus'
    if any((holdout/name).exists() for name in ['annotations_all.json','candidate_manifest.json']):raise ValueError('Confirmation9 already examined')
    data=load_growth_data(a.root);splits=growth_partitions(data);keys=['dev','old_dev','new_dev'];train=splits['train'];dev=splits['dev']
    if {row['episode_id'] for row in data['rows']}&{ep['episode_id'] for ep in json.loads((holdout/'manifest.json').read_text())['episodes']}:raise ValueError('Confirmation overlap')
    audit=json.loads(a.audit.read_text())
    review=far_future_mask(data,audit,np.r_[train,dev])
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'frozen_search_protocol.json',dict(method='train108 far-future motion proxy uncertainty loss ablation',confidence_factors=[1.,.5,.25],views=['geometry_main_only'],
        audit={str(a.audit):hashlib.sha256(a.audit.read_bytes()).hexdigest()},
        losses=['uniform','transition4'],seeds=[100,300],fit_count=12,probability_offsets=[0.,4.],tf32=False,device=a.device,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        selection='Original dual V4 AND current retention on combined24/old8/new16; no target or evaluation denominator change',
        review='Acquire label onset in future AND current privileged bowl distance >5cm; training loss factors only, not automatic truth correction',
        records=38,blends='best .5/.25 x uniform/transition4 family by registered gate merit; current-parent new mass25/50/75%',
        holdout_manifest_sha256=hashlib.sha256((holdout/'manifest.json').read_bytes()).hexdigest(),confirmation9_features_loaded=False,
        caveat='Audit flags change only known training weights; no deletion, target rewrite or privileged inference. Far anticipation does not prove a label wrong; all labels/denominators stay fixed'))
    write_json(a.output/'data_audit.json',json.loads((r/'growth_incumbent_blends_v4/data_audit.json').read_text()))
    write_json(a.output/'supervision_audit.json',dict(train_known=len(train),dev_known=len(dev),train_review=int(review[train].sum()),dev_review=int(review[dev].sum()),
        training_review_roles={str(role):int((review[train]&(data['roles'][train]==role)).sum()) for role in range(5)},labels_unchanged=True,evaluation_frames_removed=0,
        review_train_ids=[data['rows'][i]['id'] for i in train if review[i]],dev_weights_unchanged=True))
    old=json.loads((r/'frozen_for_confirm4/selected.json').read_text());old_prob=predict_component(old['result']['component'],data,a.root)
    old_pred=apply_calibration(old_prob,old['result']['calibration'])
    previous=predict_component(dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100')),data,a.root).argmax(-1)
    baseline={k:summary(previous,splits[k],data) for k in keys};incumbent={k:summary(old_pred,splits[k],data) for k in keys};registry={}
    blend=json.loads((r/'frozen_for_confirm6_incumbent_blend/selected.json').read_text());current_prob=predict_component(blend['result']['component'],data,a.root)
    current={k:summary(apply_calibration(current_prob,blend['result']['calibration']),splits[k],data) for k in keys}
    def record(name,prob,component,fixed=None):
        assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
        points=[];grid=[fixed] if fixed is not None else [dict(threshold=t,margin=m) for t in [0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98] for m in [0.,.1,.2]]
        for cal in grid:
            pred=apply_calibration(prob,cal);groups={k:summary(pred,splits[k],data) for k in keys}
            gates={k:evaluation_gate(groups[k],baseline[k],4) for k in keys};incumbent_gates={k:incumbent_evaluation_gate(groups[k],incumbent[k],4) for k in keys}
            retention={k:current_retention_gate(groups[k],current[k]) for k in keys}
            feasible=all(v['joint_goal_met'] for v in list(gates.values())+list(incumbent_gates.values())+list(retention.values()))
            points.append(dict(cal,metrics=groups['dev']['accepted'],roles=groups['dev']['roles'],sequence=groups['dev']['sequence'],
                acceptance_revision=4,joint_goal_met=bool(feasible),groups=groups,gates=gates,incumbent_gates=incumbent_gates,current_retention=retention))
        def merit(v):
            return checkpoint_merit(dict(joint_growth_goal_met=v['joint_goal_met'],calibration=v))
        chosen=max(points,key=merit)
        registry[name]=dict(name=name,component=component,calibration={k:v for k,v in chosen.items() if k not in ['groups','gates','incumbent_gates','current_retention']},
            joint_growth_goal_met=chosen['joint_goal_met'],subgroup_gates={k:chosen['gates'][k] for k in ['old_dev','new_dev']},subgroups={k:chosen['groups'][k] for k in ['old_dev','new_dev']},
            incumbent_gates=chosen['incumbent_gates'],current_retention=chosen['current_retention'],raw={k:metrics(prob.argmax(-1),splits[k],data) for k in keys})
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob);write_json(a.output/(name+'_calibration.json'),points);write_json(a.output/'registry.json',registry)
        write_json(a.output/'status.json',dict(stage='fitting',completed=len(registry),latest=name));print(name,chosen['joint_goal_met'],chosen['metrics']['correct'],chosen['metrics']['wrong_acquire'],flush=True)
    record('incumbent_frozen_configuration',old_prob,old['result']['component'],old['result']['calibration'])
    blend=json.loads((r/'frozen_for_confirm6_incumbent_blend/selected.json').read_text())
    record('blend_frozen_configuration',current_prob,blend['result']['component'],blend['result']['calibration'])
    for confidence_factor in [1.,.5,.25]:
        view=feature_view(data,geometry=True)
        for loss,factor in [('uniform',1.),('transition4',4.)]:
            weights,emphasis=training_weights(data,train,factor);fit_data=downweight_training(dict(data,weights=weights),train,review,confidence_factor)
            assert np.array_equal(fit_data['weights'][dev],data['weights'][dev])
            for seed in [100,300]:
                name=f'geometry_far_future{confidence_factor}_{loss}_seed{seed}'
                prob,ckpt,info=train_model(fit_data,view,train,dev,seed,a.device)
                ckpt['spec']=dict(name=name,view='geometry',kind='mlp',width=64,weighted=True,training_population='enlarged108',loss_emphasis=loss)
                ckpt['spec'].update(training_far_future_factor=confidence_factor)
                info['emphasis']=emphasis;torch.save(ckpt,a.output/(name+'.pt'));write_json(a.output/(name+'_training.json'),info)
                if confidence_factor==1.:
                    reference=r/'growth_confidence_v4_seed100_300'/f'geometry_confidence1.0_{loss}_seed{seed}.pt'
                    saved=torch.load(reference,map_location='cpu',weights_only=False)
                    parameter_delta=max(float((ckpt['state'][k]-saved['state'][k]).abs().max()) for k in saved['state'])
                    normalizer_delta=max(float(np.max(np.abs(ckpt['normalizer'][k]-saved['normalizer'][k]))) for k in saved['normalizer'])
                    write_json(a.output/(name+'_control_parity.json'),dict(parameter_max_abs=parameter_delta,normalizer_max_abs=normalizer_delta,reference=str(reference)))
                    assert parameter_delta==0 and normalizer_delta==0
                parent=dict(kind='parent',name=name,screen=str(a.output))
                for offset in [0.,4.]:
                    record(name+'_bias'+str(offset),replace_availability(prob,prob[:,:8].sum(-1),0.,offset),dict(kind='availability',parent=parent,model=None,mix=0.,offset=offset))
    sources={}
    for factor in [.5,.25]:
        for loss in ['uniform','transition4']:
            family=f'geometry_far_future{factor}_{loss}'
            name=max([n for n in registry if n.startswith(family+'_seed')],key=lambda n:checkpoint_merit(registry[n]));sources[family]=name
    write_json(a.output/'frozen_blend_sources.json',dict(sources=sources,fractions=[.25,.5,.75],parent='current frozen candidate',fit_count=0))
    for family,name in sources.items():
        new=np.load(a.output/(name+'_probabilities.npz'))['probability']
        for fraction in [.25,.5,.75]:record(family+'_new'+str(fraction),(1-fraction)*current_prob+fraction*new,
            dict(kind='blend',weights=[1-fraction,fraction],members=[blend['result']['component'],registry[name]['component']]))
    def merit(name):return checkpoint_merit(registry[name])
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=12,incumbent_gate_required=True,current_retention_required=True,retention_selection=str(r/'frozen_for_confirm6_incumbent_blend'),holdout_evaluated=False))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner));print('WINNER',winner,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--device',default='cpu');main(p.parse_args())
