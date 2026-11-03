"""Main-only aligned crop training vs unchanged and repeated-observation controls."""
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
from checkpoint_selection import current_retention_gate,checkpoint_merit


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=a.root/'momart_target_research_20261101';holdout=r/'confirm10_corpus'
    if any((holdout/n).exists() for n in ['annotations_all.json','candidate_manifest.json']):raise ValueError('Confirmation10 already examined')
    data=load_growth_data(a.root);splits=growth_partitions(data);keys=['dev','old_dev','new_dev'];train=splits['train'];dev=splits['dev']
    metadata=json.loads(a.augmentation.with_suffix('.json').read_text());plan=json.loads(a.augmentation.with_suffix('.plan.json').read_text())
    if metadata['protocol']!='aligned_main_crop_features_v1' or metadata['smoke'] or not plan['train_only']:raise ValueError('Complete train-only crop archive required')
    assert metadata['plan_sha256']==sha(a.augmentation.with_suffix('.plan.json'))
    assert plan['confirmation10_sha256']==sha(holdout/'manifest.json')
    cropped={key:value for key,value in np.load(a.augmentation,allow_pickle=False).items()}
    assert np.array_equal(cropped['base_index'],train)
    assert [row['id'] for row in metadata['records']]==[data['rows'][i]['id'] for i in train]
    reserved={e['episode_id'] for e in json.loads((holdout/'manifest.json').read_text())['episodes']}
    assert not {data['rows'][i]['episode_id'] for i in train}&reserved
    raw,context=feature_view(data,geometry=True)
    identity=dict(base_index=train,candidates=raw[train],context=context[train],valid=data['valid'][train],y=data['y'][train])
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'frozen_search_protocol.json',dict(method='aligned main-camera crop extra loss',fit_count=12,records=62,
        cases=[['none',0.],['identity',.5],['crop',.5]],losses=['uniform','transition4'],seeds=[100,300],
        checkpoint_policies=['raw development F1','complete fixed operating point merit'],fixed_point=dict(offset=4.,threshold=.6,margin=0.),
        optimizer='unchanged AdamW/normalization on original train only;150epochs/F1patience30',
        extra_loss='Original CE plus alpha times paired augmented CE; every original train query retained; inherited original loss weight; identity control repeats original feature with independent dropout',
        augmentation_sha256=sha(a.augmentation),metadata_sha256=sha(a.augmentation.with_suffix('.json')),plan_sha256=sha(a.augmentation.with_suffix('.plan.json')),
        blend='best member per nonzero mode/strength/loss family, current-parent new mass25/50/75%',probability_offsets=[0.,4.],
        acceptance='Original dual V4 AND current retention on combined24/old8/new16; unchanged original development frames',
        holdout='confirm10',holdout_manifest_sha256=sha(holdout/'manifest.json'),confirmation10_features_loaded=False,tf32=False,device=a.device,
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),caveat='Synthetic viewport training, not physical camera motion or intention truth; unchanged inference. No rollout/promotion claim.'))
    write_json(a.output/'data_audit.json',json.loads((r/'growth_incumbent_blends_v4/data_audit.json').read_text()))
    old=json.loads((r/'frozen_for_confirm4/selected.json').read_text());old_prob=predict_component(old['result']['component'],data,a.root)
    old_pred=apply_calibration(old_prob,old['result']['calibration'])
    previous=predict_component(dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100')),data,a.root).argmax(-1)
    baseline={k:summary(previous,splits[k],data) for k in keys};incumbent={k:summary(old_pred,splits[k],data) for k in keys};registry={}
    blend=json.loads((r/'frozen_for_confirm6_incumbent_blend/selected.json').read_text());current_probability=predict_component(blend['result']['component'],data,a.root)
    current_prediction=apply_calibration(current_probability,blend['result']['calibration']);current={k:summary(current_prediction,splits[k],data) for k in keys}
    def assess(prob,cal):
        pred=apply_calibration(prob,cal);groups={k:summary(pred,splits[k],data) for k in keys}
        gates={k:evaluation_gate(groups[k],baseline[k],4) for k in keys};incumbent_gates={k:incumbent_evaluation_gate(groups[k],incumbent[k],4) for k in keys}
        original_feasible=all(v['joint_goal_met'] for v in list(gates.values())+list(incumbent_gates.values()))
        retention={k:current_retention_gate(groups[k],current[k]) for k in keys};retained=all(v['joint_goal_met'] for v in retention.values())
        return dict(cal,metrics=groups['dev']['accepted'],roles=groups['dev']['roles'],sequence=groups['dev']['sequence'],
            acceptance_revision=4,joint_goal_met=bool(original_feasible and retained),original_dual_goal_met=bool(original_feasible),current_retention_met=bool(retained),
            groups=groups,gates=gates,incumbent_gates=incumbent_gates,current_retention=retention)
    def record(name,prob,component,fixed=None):
        assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
        grid=[fixed] if fixed is not None else [dict(threshold=t,margin=m) for t in [0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98] for m in [0.,.1,.2]]
        points=[assess(prob,cal) for cal in grid]
        chosen=max(points,key=lambda v:checkpoint_merit(dict(joint_growth_goal_met=v['joint_goal_met'],calibration=v)))
        registry[name]=dict(name=name,component=component,calibration={k:v for k,v in chosen.items() if k not in ['groups','gates','incumbent_gates','current_retention']},
            joint_growth_goal_met=chosen['joint_goal_met'],subgroup_gates={k:chosen['gates'][k] for k in ['old_dev','new_dev']},subgroups={k:chosen['groups'][k] for k in ['old_dev','new_dev']},
            incumbent_gates=chosen['incumbent_gates'],current_retention=chosen['current_retention'],raw={k:metrics(prob.argmax(-1),splits[k],data) for k in keys})
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob);write_json(a.output/(name+'_calibration.json'),points);write_json(a.output/'registry.json',registry)
        write_json(a.output/'status.json',dict(stage='fitting',completed=len(registry),latest=name));print(name,chosen['joint_goal_met'],chosen['metrics']['correct'],chosen['metrics']['wrong_acquire'],flush=True)
    record('incumbent_frozen_configuration',old_prob,old['result']['component'],old['result']['calibration'])
    blend=json.loads((r/'frozen_for_confirm6_incumbent_blend/selected.json').read_text())
    record('blend_frozen_configuration',predict_component(blend['result']['component'],data,a.root),blend['result']['component'],blend['result']['calibration'])
    families={}
    def assess_fixed(p):return assess(replace_availability(p,p[:,:8].sum(-1),0.,4.),dict(threshold=.6,margin=0.))
    for loss,factor in [('uniform',1.),('transition4',4.)]:
        weights,emphasis=training_weights(data,train,factor);fit_data=dict(data,weights=weights)
        for seed in [100,300]:
            for mode,strength in [('none',0.),('identity',.5),('crop',.5)]:
                family=f'{mode}{strength}_{loss}';base=f'geometry_crop_{family}_seed{seed}';best_fixed=None;snapshot={}
                def callback(epoch,net,p,norm):
                    nonlocal best_fixed,snapshot
                    point=assess_fixed(p);merit=checkpoint_merit(dict(joint_growth_goal_met=point['joint_goal_met'],calibration=point))
                    if best_fixed is None or merit>best_fixed:
                        best_fixed=merit;snapshot=dict(state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},normalizer=norm,seed=seed,best_epoch=epoch)
                augmentation=None if mode=='none' else identity if mode=='identity' else cropped
                probability,raw_checkpoint,info=train_model(fit_data,(raw,context),train,dev,seed,a.device,epoch_callback=callback,
                    augmentation=augmentation,augmentation_strength=strength)
                assert snapshot
                if mode=='none':
                    reference=r/'growth_confidence_v4_seed100_300'/f'geometry_confidence1.0_{loss}_seed{seed}.pt'
                    old_checkpoint=torch.load(reference,map_location='cpu',weights_only=False)
                    delta=max(float((raw_checkpoint['state'][k]-old_checkpoint['state'][k]).abs().max()) for k in old_checkpoint['state'])
                    norm_delta=max(float(np.max(np.abs(raw_checkpoint['normalizer'][k]-old_checkpoint['normalizer'][k]))) for k in old_checkpoint['normalizer'])
                    write_json(a.output/(base+'_control_parity.json'),dict(parameter_max_abs=delta,normalizer_max_abs=norm_delta,reference=str(reference)))
                    assert delta==0 and norm_delta==0
                write_json(a.output/(base+'_training.json'),dict(info,emphasis=emphasis,augmentation_mode=mode,augmentation_strength=strength))
                for policy,checkpoint in [('raw_f1',raw_checkpoint),('fixed_gate',snapshot)]:
                    name=base+'_'+policy;checkpoint['spec']=dict(name=name,view='geometry',kind='mlp',width=64,weighted=True,training_population='enlarged108',
                        loss_emphasis=loss,augmentation_mode=mode,augmentation_strength=strength,augmentation_sha256=sha(a.augmentation),checkpoint_policy=policy)
                    torch.save(checkpoint,a.output/(name+'.pt'));parent=dict(kind='parent',name=name,screen=str(a.output))
                    p=probability if policy=='raw_f1' else predict_component(parent,data,a.root)
                    for offset in [0.,4.]:
                        label=name+'_bias'+str(offset);record(label,replace_availability(p,p[:,:8].sum(-1),0.,offset),dict(kind='availability',parent=parent,model=None,mix=0.,offset=offset))
                        if mode!='none':families.setdefault(family,[]).append(label)
    sources={family:max(names,key=lambda name:checkpoint_merit(registry[name])) for family,names in families.items()}
    assert len(sources)==4
    write_json(a.output/'frozen_blend_sources.json',dict(sources=sources,fractions=[.25,.5,.75],parent='current frozen candidate',fit_count=0))
    for family,name in sources.items():
        p=np.load(a.output/(name+'_probabilities.npz'))['probability']
        for fraction in [.25,.5,.75]:record('current_'+family+'_new'+str(fraction),(1-fraction)*current_probability+fraction*p,
            dict(kind='blend',weights=[1-fraction,fraction],members=[blend['result']['component'],registry[name]['component']]))
    winner=max(registry,key=lambda name:checkpoint_merit(registry[name]));assert len(registry)==62
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=12,incumbent_gate_required=True,
        current_retention_required=True,retention_selection=str(r/'frozen_for_confirm6_incumbent_blend'),holdout_evaluated=False))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner));print('WINNER',winner,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','output','augmentation']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cpu');main(p.parse_args())
