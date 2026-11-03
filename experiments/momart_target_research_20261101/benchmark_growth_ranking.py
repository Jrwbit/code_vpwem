"""Main-view hard-negative ranking, paired against identical CE controls.

Only train108/development24; fresh confirmation7 remains untouched.
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


def main(a):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=a.root/'momart_target_research_20261101';holdout=r/'confirm7_corpus'
    if any((holdout/name).exists() for name in ['annotations_all.json','candidate_manifest.json']):raise ValueError('Confirmation7 already examined')
    data=load_growth_data(a.root);splits=growth_partitions(data);keys=['dev','old_dev','new_dev'];train=splits['train'];dev=splits['dev']
    if {row['episode_id'] for row in data['rows']}&{ep['episode_id'] for ep in json.loads((holdout/'manifest.json').read_text())['episodes']}:raise ValueError('Confirmation overlap')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'frozen_search_protocol.json',dict(method='hardest incorrect observed box ranking on train108/dev24',ranking_strengths=[0.,.25,1.],ranking_margin=1.,views=['geometry_main_only'],
        losses=['uniform','transition4'],seeds=[100,300],fit_count=12,probability_offsets=[0.,4.],tf32=False,device=a.device,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        selection='Dual-comparator V4 on combined24, old8 and new16; no label/denominator change',
        holdout_manifest_sha256=hashlib.sha256((holdout/'manifest.json').read_bytes()).hexdigest(),confirmation7_features_loaded=False,
        caveat='Ranking uses train-only positive box masks, excludes alternative correct boxes, absent targets, unknowns and null rivals; inference unchanged; weak proxy labels are not human intention GT'))
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
    for ranking_strength in [0.,.25,1.]:
        view=feature_view(data,geometry=True)
        for loss,factor in [('uniform',1.),('transition4',4.)]:
            weights,emphasis=training_weights(data,train,factor);fit_data=dict(data,weights=weights)
            for seed in [100,300]:
                name=f'geometry_ranking{ranking_strength}_{loss}_seed{seed}'
                prob,ckpt,info=train_model(fit_data,view,train,dev,seed,a.device,ranking_strength=ranking_strength)
                ckpt['spec']=dict(name=name,view='geometry',kind='mlp',width=64,weighted=True,training_population='enlarged108',loss_emphasis=loss)
                ckpt['spec'].update(ranking_strength=ranking_strength,ranking_margin=1.)
                info['emphasis']=emphasis;torch.save(ckpt,a.output/(name+'.pt'));write_json(a.output/(name+'_training.json'),info)
                parent=dict(kind='parent',name=name,screen=str(a.output))
                for offset in [0.,4.]:
                    record(name+'_bias'+str(offset),replace_availability(prob,prob[:,:8].sum(-1),0.,offset),dict(kind='availability',parent=parent,model=None,mix=0.,offset=offset))
    def merit(name):
        v=registry[name];m=v['calibration']['metrics'];return v['joint_growth_goal_met'],m['recall'] if v['joint_growth_goal_met'] else m['f1'],-m['wrong_acquire'],m['precision']
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=12,incumbent_gate_required=True,holdout_evaluated=False))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner));print('WINNER',winner,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cpu');main(p.parse_args())
