"""Align training with evaluation's any-matching-box target semantics."""
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
    reserved={e['episode_id'] for e in json.loads((holdout/'manifest.json').read_text())['episodes']}
    assert not {row['episode_id'] for row in data['rows']}&reserved
    raw,context=feature_view(data,geometry=True)
    assert not (data['positive']&~data['valid']).any()
    assert np.array_equal(data['positive'].any(-1), (data['y']>=0)&(data['y']<8))
    a.output.mkdir(parents=True,exist_ok=False)
    audit={key:dict(samples=len(indices),multiple=int((data['positive'][indices].sum(-1)>1).sum()),
        by_role={str(role):int(((data['positive'][indices].sum(-1)>1)&(data['roles'][indices]==role)).sum()) for role in range(5)}) for key,indices in splits.items()}
    write_json(a.output/'positive_set_audit.json',audit)
    write_json(a.output/'frozen_search_protocol.json',dict(method='Any-matching-box training objective versus canonical-slot CE',fit_count=12,records=62,
        modes=['canonical','uniform','marginal'],losses=['uniform','transition4'],seeds=[100,300],positive_audit=audit,
        canonical='Unchanged original CE, exact saved reference-state audit for four paired controls',
        uniform='Equal target mass over all unchanged positive boxes; unavailable target supervises null',
        marginal='Negative log total probability over all unchanged positive boxes; unavailable target supervises null',
        unknown='Unknown target rows remain outside train/dev loss; no relabel or removal of any existing known query',
        checkpoint_policies=['raw development F1','complete fixed operating point merit'],fixed_point=dict(offset=4.,threshold=.6,margin=0.),
        optimizer='Unchanged original train normalization, AdamW, 150epochs, raw-F1 patience30',
        blend='best member per noncanonical objective/loss family, current-parent new mass25/50/75%',probability_offsets=[0.,4.],
        acceptance='Original dual V4 AND current retention on combined24/old8/new16; unchanged original development queries',
        train_ids=[data['rows'][i]['id'] for i in train],development_ids=[data['rows'][i]['id'] for i in dev],
        data_hashes={prefix:{name:sha(r/(prefix+'_corpus')/name) for name in ['manifest.json','annotations_all.json']} for prefix in ['dense','confirmation','confirm2','growth']},
        holdout='confirm10',holdout_manifest_sha256=sha(holdout/'manifest.json'),confirmation10_features_loaded=False,tf32=False,device=a.device,
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),caveat='Observed equivalent boxes only, not proof of same visual instance or human intention; unchanged public main-view inference'))
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
            for mode in ['canonical','uniform','marginal']:
                family=f'{mode}_{loss}';base=f'geometry_positive_{family}_seed{seed}';best_fixed=None;snapshot={}
                def callback(epoch,net,p,norm):
                    nonlocal best_fixed,snapshot
                    value=assess_fixed(p);merit=checkpoint_merit(dict(joint_growth_goal_met=value['joint_goal_met'],calibration=value))
                    if best_fixed is None or merit>best_fixed:
                        best_fixed=merit;snapshot=dict(state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},normalizer=norm,seed=seed,best_epoch=epoch)
                probability,raw_checkpoint,info=train_model(fit_data,(raw,context),train,dev,seed,a.device,epoch_callback=callback,positive_mode=mode)
                assert snapshot
                if mode=='canonical':
                    reference=r/'growth_confidence_v4_seed100_300'/f'geometry_confidence1.0_{loss}_seed{seed}.pt'
                    old_checkpoint=torch.load(reference,map_location='cpu',weights_only=False)
                    delta=max(float((raw_checkpoint['state'][k]-old_checkpoint['state'][k]).abs().max()) for k in old_checkpoint['state'])
                    norm_delta=max(float(np.max(np.abs(raw_checkpoint['normalizer'][k]-old_checkpoint['normalizer'][k]))) for k in old_checkpoint['normalizer'])
                    write_json(a.output/(base+'_control_parity.json'),dict(parameter_max_abs=delta,normalizer_max_abs=norm_delta,reference=str(reference)))
                    assert delta==0 and norm_delta==0
                write_json(a.output/(base+'_training.json'),dict(info,emphasis=emphasis,positive_mode=mode))
                for policy,checkpoint in [('raw_f1',raw_checkpoint),('fixed_gate',snapshot)]:
                    name=base+'_'+policy;checkpoint['spec']=dict(name=name,view='geometry',kind='mlp',width=64,weighted=True,training_population='enlarged108',
                        loss_emphasis=loss,positive_mode=mode,checkpoint_policy=policy)
                    torch.save(checkpoint,a.output/(name+'.pt'));parent=dict(kind='parent',name=name,screen=str(a.output))
                    p=probability if policy=='raw_f1' else predict_component(parent,data,a.root)
                    for offset in [0.,4.]:
                        label=name+'_bias'+str(offset);record(label,replace_availability(p,p[:,:8].sum(-1),0.,offset),dict(kind='availability',parent=parent,model=None,mix=0.,offset=offset))
                        if mode!='canonical':families.setdefault(family,[]).append(label)
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
    for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cpu');main(p.parse_args())
