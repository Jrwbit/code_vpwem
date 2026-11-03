"""Save and evaluate actual epoch13/37 snapshots of the logged priority tradeoff.

Exploratory unchanged-development study, not a fresh confirmation. Report exact
reconstruction separately; the snapshots are real newly fitted model weights.
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
from benchmark_growth import train_model
from benchmark_growth_blend import summary
from calibration import apply_calibration
from confirm import evaluation_gate,incumbent_evaluation_gate,predict_component
from data import feature_view,metrics
from growth_data import load_growth_data,growth_partitions
from transition_weights import training_weights
from training_consistency import stable_training_predecessors


def main(a):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=a.root/'momart_target_research_20261101';holdout=r/'confirm7_corpus'
    if any((holdout/name).exists() for name in ['annotations_all.json','candidate_manifest.json']):raise ValueError('Confirmation7 already examined')
    data=load_growth_data(a.root);splits=growth_partitions(data);keys=['dev','old_dev','new_dev'];train=splits['train'];dev=splits['dev']
    if {row['episode_id'] for row in data['rows']}&{ep['episode_id'] for ep in json.loads((holdout/'manifest.json').read_text())['episodes']}:raise ValueError('Confirmation overlap')
    predecessors=stable_training_predecessors(data,train)
    source=r/'growth_consistency_v4_seed100_300';reference_name='geometry_consistency_temporal2.0_transition4_seed300'
    reference_path=source/(reference_name+'.pt');reference_history=json.loads((source/(reference_name+'_training.json')).read_text())['history']
    assert np.array_equal(predecessors,np.load(source/'training_pairs.npz')['predecessor'])
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'frozen_search_protocol.json',dict(method='actual checkpoint13 vs37 replay with fixed blends',fit_count=1,epochs=[13,37],seed=300,
        consistency_mode='temporal',consistency_strength=2.,transition_factor=4.,probability_offsets=[0.,4.],blend_fractions=[.25,.5,.75],
        blend_parents=['original_frozen','current_frozen_blend'],blend_checkpoint='epoch13 offset4',views=['geometry_main_only'],
        acceptance='Unchanged original pointer and original frozen control V4, combined24/old8/new16; current frozen blend retained as full configuration candidate',
        tf32=False,device=a.device,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        reference_checkpoint_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),reference_history_sha256=hashlib.sha256((source/(reference_name+'_training.json')).read_bytes()).hexdigest(),
        holdout_manifest_sha256=hashlib.sha256((holdout/'manifest.json').read_bytes()).hexdigest(),confirmation7_features_loaded=False,
        caveat='Epoch13 motivated by examined dev logs. Real rerun snapshots, not a prospective checkpoint-selection rule. No labels, denominators or input features changed; report reconstruction parity explicitly.'))
    write_json(a.output/'data_audit.json',json.loads((r/'growth_incumbent_blends_v4/data_audit.json').read_text()))
    old=json.loads((r/'frozen_for_confirm4/selected.json').read_text());old_prob=predict_component(old['result']['component'],data,a.root)
    old_pred=apply_calibration(old_prob,old['result']['calibration'])
    previous=predict_component(dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100')),data,a.root).argmax(-1)
    baseline={k:summary(previous,splits[k],data) for k in keys};incumbent={k:summary(old_pred,splits[k],data) for k in keys};registry={}
    def record(name,prob,component,fixed=None):
        assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
        points=[];grid=[fixed] if fixed is not None else [dict(threshold=t,margin=m) for t in [0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98] for m in [0.,.1,.2]]
        for cal in grid:
            pred=apply_calibration(prob,cal);groups={k:summary(pred,splits[k],data) for k in keys}
            gates={k:evaluation_gate(groups[k],baseline[k],4) for k in keys};incumbent_gates={k:incumbent_evaluation_gate(groups[k],incumbent[k],4) for k in keys}
            feasible=all(v['joint_goal_met'] for v in list(gates.values())+list(incumbent_gates.values()))
            points.append(dict(cal,metrics=groups['dev']['accepted'],roles=groups['dev']['roles'],sequence=groups['dev']['sequence'],
                acceptance_revision=4,joint_goal_met=bool(feasible),groups=groups,gates=gates,incumbent_gates=incumbent_gates))
        def merit(v):
            m=v['metrics'];return v['joint_goal_met'],m['recall'] if v['joint_goal_met'] else m['f1'],-m['wrong_acquire'],m['precision']
        chosen=max(points,key=merit)
        registry[name]=dict(name=name,component=component,calibration={k:v for k,v in chosen.items() if k not in ['groups','gates','incumbent_gates']},
            joint_growth_goal_met=chosen['joint_goal_met'],subgroup_gates={k:chosen['gates'][k] for k in ['old_dev','new_dev']},subgroups={k:chosen['groups'][k] for k in ['old_dev','new_dev']},
            incumbent_gates=chosen['incumbent_gates'],raw={k:metrics(prob.argmax(-1),splits[k],data) for k in keys})
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob);write_json(a.output/(name+'_calibration.json'),points);write_json(a.output/'registry.json',registry)
        write_json(a.output/'status.json',dict(stage='fitting',completed=len(registry),latest=name));print(name,chosen['joint_goal_met'],chosen['metrics']['correct'],chosen['metrics']['wrong_acquire'],flush=True)
    record('incumbent_frozen_configuration',old_prob,old['result']['component'],old['result']['calibration'])
    blend=json.loads((r/'frozen_for_confirm6_incumbent_blend/selected.json').read_text())
    record('blend_frozen_configuration',predict_component(blend['result']['component'],data,a.root),blend['result']['component'],blend['result']['calibration'])
    snapshots={}
    def save_epoch(epoch,net,probability,normalizer):
        if epoch not in [13,37]:return
        name=reference_name+'_epoch'+str(epoch)
        ckpt=dict(state={key:value.detach().cpu().clone() for key,value in net.state_dict().items()},seed=300,best_epoch=epoch,normalizer=normalizer,
            spec=dict(name=name,view='geometry',kind='mlp',width=64,weighted=True,training_population='enlarged108',loss_emphasis='transition4',training_consistency_mode='temporal',training_consistency_strength=2.))
        torch.save(ckpt,a.output/(name+'.pt'));snapshots[epoch]=dict(name=name,probability=probability.copy(),checkpoint=ckpt)
        print('SAVED_EPOCH',epoch,flush=True)
    weights,emphasis=training_weights(data,train,4.);fit_data=dict(data,weights=weights)
    probability,checkpoint,info=train_model(fit_data,feature_view(data,geometry=True),train,dev,300,a.device,
        consistency_strength=2.,consistency_mode='temporal',consistency_predecessor=predecessors,epoch_callback=save_epoch)
    write_json(a.output/'replay_training.json',dict(info,emphasis=emphasis));assert set(snapshots)=={13,37}
    reference=torch.load(reference_path,map_location='cpu',weights_only=False);replayed=snapshots[37]['checkpoint']
    delta=max(float((replayed['state'][key]-reference['state'][key]).abs().max()) for key in reference['state'])
    normalization_delta=max(float(np.max(np.abs(replayed['normalizer'][key]-reference['normalizer'][key]))) for key in reference['normalizer'])
    changed=sum(left['dev']!=right['dev'] for left,right in zip(info['history'],reference_history))+abs(len(info['history'])-len(reference_history))
    best_epoch_matches=checkpoint['best_epoch']==reference['best_epoch']
    write_json(a.output/'reconstruction_audit.json',dict(max_parameter_difference=delta,max_normalizer_difference=normalization_delta,
        dev_history_rows_changed=changed,best_epoch=checkpoint['best_epoch'],reference_best_epoch=reference['best_epoch'],
        exact_reconstruction=bool(delta==0 and normalization_delta==0 and changed==0 and best_epoch_matches),
        inference='Both snapshots are saved actual weights; when parity fails do not call epoch13 a reconstruction of the old log.'))
    components={};adjusted={}
    for epoch,snapshot in snapshots.items():
        parent=dict(kind='parent',name=snapshot['name'],screen=str(a.output))
        for offset in [0.,4.]:
            name='epoch'+str(epoch)+'_bias'+str(offset)
            prob=replace_availability(snapshot['probability'],snapshot['probability'][:,:8].sum(-1),0.,offset)
            component=dict(kind='availability',parent=parent,model=None,mix=0.,offset=offset)
            record(name,prob,component);components[name]=component;adjusted[name]=prob
    current_prob=predict_component(blend['result']['component'],data,a.root)
    for base_name,base_prob,base_component in [('original_frozen',old_prob,old['result']['component']),('current_frozen_blend',current_prob,blend['result']['component'])]:
        for fraction in [.25,.5,.75]:
            record(base_name+'_epoch13_new'+str(fraction),(1-fraction)*base_prob+fraction*adjusted['epoch13_bias4.0'],
                dict(kind='blend',weights=[1-fraction,fraction],members=[base_component,components['epoch13_bias4.0']]))
    def merit(name):
        v=registry[name];m=v['calibration']['metrics'];return v['joint_growth_goal_met'],m['recall'] if v['joint_growth_goal_met'] else m['f1'],-m['wrong_acquire'],m['precision']
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=1,incumbent_gate_required=True,holdout_evaluated=False))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner));print('WINNER',winner,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cpu');main(p.parse_args())
