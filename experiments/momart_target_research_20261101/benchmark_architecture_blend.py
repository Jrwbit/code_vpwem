"""Registered complementary fixed-clock architecture blends, development only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from benchmark import write_json
from benchmark_growth_blend import summary
from calibration import apply_calibration
from checkpoint_selection import current_retention_gate,checkpoint_merit
from confirm import predict_component,evaluation_gate,incumbent_evaluation_gate
from data import metrics
from growth_data import load_growth_data,growth_partitions


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    torch.set_num_threads(2);r=a.root/'momart_target_research_20261101';reserve=r/'confirm8_corpus'
    if any((reserve/n).exists() for n in ['annotations_all.json','candidate_manifest.json']):raise ValueError('Confirmation8 already examined')
    data=load_growth_data(a.root);splits=growth_partitions(data);keys=['dev','old_dev','new_dev']
    assert not {x['episode_id'] for x in data['rows']}&{x['episode_id'] for x in json.loads((reserve/'manifest.json').read_text())['episodes']}
    screens=[r/'growth_recurrent_fixed_clock_v4',r/'growth_recurrent_current_skip_v4'];registries=[];source_meta=[]
    for screen in screens:
        assert json.loads((screen/'status.json').read_text())['stage']=='complete'
        registries.append(json.loads((screen/'registry.json').read_text()))
        source_meta.append(dict(screen=str(screen),registry_sha256=sha(screen/'registry.json'),data_sha256=sha(screen/'data_audit.json'),protocol=json.loads((screen/'frozen_search_protocol.json').read_text())))
    assert source_meta[0]['data_sha256']==source_meta[1]['data_sha256']
    assert source_meta[0]['protocol']['main_history_sha256']==source_meta[1]['protocol']['main_history_sha256']
    sources=['current_gru_'+loss+'_new0.25' for loss in ['uniform','transition4']]
    for meta,screen in zip(source_meta,screens):meta['probabilities_sha256']={name:sha(screen/(name+'_probabilities.npz')) for name in sources}
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'frozen_search_protocol.json',dict(method='blend hidden-only and current-skip GRU public probabilities',fit_count=0,
        sources=source_meta,members=sources,skip_model_fractions=[.25,.5,.75],records=8,calibration_points=200,
        motivation='Examined development tradeoff: hidden-only fewer class switches but missed one change; skip restores change but increases chatter. Development search, not new confirmation.',
        acceptance='Original pointer V4 AND original frozen V4 AND current retention, combined24/old8/new16',
        holdout='confirm8',holdout_sha256=sha(reserve/'manifest.json'),holdout_loaded=False))
    original=json.loads((r/'frozen_for_confirm4/selected.json').read_text())['result'];current=json.loads((r/'frozen_for_confirm6_incumbent_blend/selected.json').read_text())['result']
    original_p=predict_component(original['component'],data,a.root);current_p=predict_component(current['component'],data,a.root)
    pointer=predict_component(dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100')),data,a.root).argmax(-1)
    references={}
    for label,pred in [('pointer',pointer),('original',apply_calibration(original_p,original['calibration'])),('current',apply_calibration(current_p,current['calibration']))]:
        references[label]={k:summary(pred,splits[k],data) for k in keys}
    registry={}
    def record(name,p,component,fixed=None):
        assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(-1),1,atol=1e-5)
        points=[]
        for cal in ([fixed] if fixed is not None else [dict(threshold=t,margin=m) for t in [0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98] for m in [0.,.1,.2]]):
            pred=apply_calibration(p,cal);groups={k:summary(pred,splits[k],data) for k in keys}
            gates={k:evaluation_gate(groups[k],references['pointer'][k],4) for k in keys}
            incumbent={k:incumbent_evaluation_gate(groups[k],references['original'][k],4) for k in keys}
            retention={k:current_retention_gate(groups[k],references['current'][k]) for k in keys}
            feasible=all(g['joint_goal_met'] for group in [gates,incumbent,retention] for g in group.values())
            points.append(dict(cal,metrics=groups['dev']['accepted'],roles=groups['dev']['roles'],sequence=groups['dev']['sequence'],joint_goal_met=feasible,
                groups=groups,gates=gates,incumbent_gates=incumbent,current_retention=retention,acceptance_revision=4))
        chosen=max(points,key=lambda x:checkpoint_merit(dict(joint_growth_goal_met=x['joint_goal_met'],calibration=x)))
        registry[name]=dict(name=name,component=component,joint_growth_goal_met=chosen['joint_goal_met'],
            calibration={k:v for k,v in chosen.items() if k not in ['groups','gates','incumbent_gates','current_retention']},
            subgroup_gates={k:chosen['gates'][k] for k in ['old_dev','new_dev']},subgroups={k:chosen['groups'][k] for k in ['old_dev','new_dev']},
            incumbent_gates=chosen['incumbent_gates'],current_retention=chosen['current_retention'],raw={k:metrics(p.argmax(-1),splits[k],data) for k in keys})
        write_json(a.output/(name+'_calibration.json'),points);np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=p)
        write_json(a.output/'registry.json',registry);print(name,chosen['joint_goal_met'],chosen['metrics']['correct'],chosen['sequence']['stable_class_switches'],flush=True)
    record('incumbent_frozen_configuration',original_p,original['component'],original['calibration'])
    record('blend_frozen_configuration',current_p,current['component'],current['calibration'])
    for name in sources:
        left,right=[np.load(screen/(name+'_probabilities.npz'))['probability'] for screen in screens]
        assert left.shape==right.shape==(len(data['rows']),9)
        for fraction in [.25,.5,.75]:
            record(name+'_skip'+str(fraction),(1-fraction)*left+fraction*right,
                dict(kind='blend',weights=[1-fraction,fraction],members=[registry_source[name]['component'] for registry_source in registries]))
    winner=max(registry,key=lambda k:checkpoint_merit(registry[k]))
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=0,incumbent_gate_required=True,
        current_retention_required=True,retention_selection=str(r/'frozen_for_confirm6_incumbent_blend'),holdout_evaluated=False))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner));print('WINNER',winner,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
