"""Main-view train108 causal recurrence versus same-parameter reset controls.

Unknown labels remain public observation context. Persistent state advances on
a fixed public clock and resets only at episode starts. Confirmation8 stays
unprocessed during search.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from availability_features import replace_availability
from benchmark import write_json
from recurrent_selection import fit_recurrent
from main_history import load_history,query_context
from benchmark_growth_blend import summary
from calibration import apply_calibration
from confirm import evaluation_gate,incumbent_evaluation_gate,predict_component
from data import feature_view,metrics
from growth_data import load_growth_data,growth_partitions
from transition_weights import training_weights
from checkpoint_selection import current_retention_gate,checkpoint_merit


def main(a):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=a.root/'momart_target_research_20261101';holdout=r/(a.holdout_prefix+'_corpus')
    if any((holdout/name).exists() for name in ['annotations_all.json','candidate_manifest.json']):raise ValueError('Requested confirmation already examined')
    data=load_growth_data(a.root);splits=growth_partitions(data);keys=['dev','old_dev','new_dev'];train=splits['train'];dev=splits['dev']
    if {row['episode_id'] for row in data['rows']}&{ep['episode_id'] for ep in json.loads((holdout/'manifest.json').read_text())['episodes']}:raise ValueError('Confirmation overlap')
    history=load_history(a.main_history)
    assert not {r['episode_id'] for r in history['rows']}&{e['episode_id'] for e in json.loads((holdout/'manifest.json').read_text())['episodes']}
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'frozen_search_protocol.json',dict(method='paired reset/stateful GRU on train108 main-view features',fit_count=8,modes=['reset','gru'],skip_current=a.current_skip,
        architecture='Concatenate uncompressed current public context to recurrent hidden before pointer' if a.current_skip else 'recurrent hidden only',
        seeds=[100,300],losses=['uniform','transition4'],width=64,episode_minibatch=8,max_epochs=150,raw_f1_patience=30,
        checkpoint_policies=['maximum raw dev F1','complete fixed operating point gate merit'],
        fixed_operating_point='offset4, threshold .6, margin0 (current frozen comparator operating point)',
        probability_offsets=[0.,4.],family_blends='best reset/gru x loss family by registered merit, current-parent new mass25/50/75%',
        segmentation='Fixed canonical8-step public grid from episode start, including alignment-rejected context; off-grid queries are temporary and never change persistent state',
        main_history_sha256=hashlib.sha256(a.main_history.read_bytes()).hexdigest(),main_history_metadata_sha256=hashlib.sha256(a.main_history.with_suffix('.json').read_bytes()).hexdigest(),
        gap_transform='log1p(delta_steps/8), zero at episode start',views=['geometry_main_only'],
        acceptance='Original pointer and original frozen control V4 AND current frozen blend retention, combined24/old8/new16',
        holdout=a.holdout_prefix,holdout_manifest_sha256=hashlib.sha256((holdout/'manifest.json').read_bytes()).hexdigest(),holdout_features_loaded=False,
        tf32=False,device=a.device,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        caveat='Main-view temporal stage hypothesis after examined confirmation7 failure; no confirmation7/8 features used for gradients or checkpoint selection. Causal recurrent selector, not a policy rollout or human intention GT.'))
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
    initializations={};families={mode+'_'+loss:[] for mode in ['reset','gru'] for loss in ['uniform','transition4']}
    def assess_fixed(p):return assess(replace_availability(p,p[:,:8].sum(-1),0.,4.),dict(threshold=.6,margin=0.))
    for loss,factor in [('uniform',1.),('transition4',4.)]:
        weights,emphasis=training_weights(data,train,factor);fit_data=dict(data,weights=weights)
        for seed in [100,300]:
            for mode in ['reset','gru']:
                base=f'geometry_recurrent_{mode}_{loss}_seed{seed}'
                predictions,states,info=fit_recurrent(fit_data,(feature_view(data,geometry=True)[0],query_context(data,history)),splits,seed,a.device,mode,assess_fixed,history,skip_current=a.current_skip)
                initializations[base]=info['initial_parameter_sha256'];write_json(a.output/'paired_initializations.json',initializations)
                write_json(a.output/(base+'_training.json'),dict(info,emphasis=emphasis))
                for policy,checkpoint in states.items():
                    name=base+'_'+policy;checkpoint['spec']=dict(name=name,view='geometry',kind='grid_gru',width=64,reset_each_frame=mode=='reset',skip_current=a.current_skip,main_history_path=str(a.main_history),
                        gap_transform='log1p(delta_steps/8)',checkpoint_policy=policy,training_population='enlarged108',weighted=True,loss_emphasis=loss)
                    torch.save(checkpoint,a.output/(name+'.pt'));parent=dict(kind='parent',name=name,screen=str(a.output))
                    for offset in [0.,4.]:
                        label=name+'_bias'+str(offset);p=predictions[policy]
                        record(label,replace_availability(p,p[:,:8].sum(-1),0.,offset),dict(kind='availability',parent=parent,model=None,mix=0.,offset=offset));families[mode+'_'+loss].append(label)
            assert initializations[f'geometry_recurrent_reset_{loss}_seed{seed}']==initializations[f'geometry_recurrent_gru_{loss}_seed{seed}']
    sources={family:max(names,key=lambda name:checkpoint_merit(registry[name])) for family,names in families.items()}
    write_json(a.output/'frozen_blend_sources.json',dict(sources=sources,fractions=[.25,.5,.75],parent='current frozen blend',fit_count=0))
    for family,name in sources.items():
        p=np.load(a.output/(name+'_probabilities.npz'))['probability']
        for fraction in [.25,.5,.75]:
            record('current_'+family+'_new'+str(fraction),(1-fraction)*current_probability+fraction*p,
                dict(kind='blend',weights=[1-fraction,fraction],members=[blend['result']['component'],registry[name]['component']]))
    def merit(name):return checkpoint_merit(registry[name])
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=8,incumbent_gate_required=True,current_retention_required=True,retention_selection=str(r/'frozen_for_confirm6_incumbent_blend'),holdout_evaluated=False))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner));print('WINNER',winner,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--current-skip',action='store_true');p.add_argument('--main-history',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--holdout-prefix',default='confirm8');main(p.parse_args())
