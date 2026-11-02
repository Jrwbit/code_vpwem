"""Public-only probability blends; calibrate against pointer and frozen incumbent.

No fitting or confirmation labels. Five predeclared weighting-family winners
from train108/dev24 are blended with the incumbent at .25/.5/.75 new-model mass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark import write_json
from confirm import apply_calibration,evaluation_gate,incumbent_evaluation_gate,predict_component
from growth_data import load_growth_data,growth_partitions
from data import metrics
from refine import role_breakdown
from sequence_metrics import sequence_metrics


def summary(pred,index,data):
    m=metrics(pred,index,data)
    return dict(raw=m,accepted=m,roles=role_breakdown(pred,index,data),sequence=sequence_metrics(pred,data,index))


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    research=a.root/'momart_target_research_20261101';source=research/'transition_emphasis_v4_seed100_300'
    if json.loads((source/'status.json').read_text())['stage']!='complete':raise ValueError('Source search incomplete')
    holdout=research/'confirm6_corpus'
    if any((holdout/name).exists() for name in ['annotations_all.json','candidate_manifest.json']):raise ValueError('Fresh confirmation already examined')
    data=load_growth_data(a.root);splits=growth_partitions(data);keys=['dev','old_dev','new_dev']
    if {r['episode_id'] for r in data['rows']}&{e['episode_id'] for e in json.loads((holdout/'manifest.json').read_text())['episodes']}:raise ValueError('Confirmation overlap')
    old=json.loads((research/'frozen_for_confirm4/selected.json').read_text());old_component=old['result']['component']
    old_probability=predict_component(old_component,data,a.root);old_pred=apply_calibration(old_probability,old['result']['calibration'])
    previous=predict_component(dict(kind='parent',name='previous_pointer',screen=str(research/'screen_seed100')),data,a.root).argmax(-1)
    baseline={k:summary(previous,splits[k],data) for k in keys};incumbent={k:summary(old_pred,splits[k],data) for k in keys}
    original=json.loads((source/'registry.json').read_text());registry={};families={}
    def merit(value):
        m=value['calibration']['metrics'];return value['joint_growth_goal_met'],m['recall'] if value['joint_growth_goal_met'] else m['f1'],-m['wrong_acquire']
    for family in ['uniform','transition2','transition4','acquire2','combined2']:
        names=[name for name in original if '_'+family+'_bias' in name]
        families[family]=max(names,key=lambda name:merit(original[name]))
    write_json(a.output/'frozen_search_protocol.json',dict(method='incumbent + weighting-family probability blends',families=families,
        fractions=[.25,.5,.75],thresholds=[0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98],margins=[0.,.1,.2],fit_count=0,
        selection='Both original pointer and fixed calibrated incumbent V4 on combined24 and old8/new16 separately',
        holdout='confirm6, untouched selector holdout only',holdout_manifest_sha256=hashlib.sha256((holdout/'manifest.json').read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((research/'growth_corpus/manifest.json').read_bytes()).hexdigest(),
        control_calibration=old['result']['calibration'],source_registry_sha256=hashlib.sha256((source/'registry.json').read_bytes()).hexdigest()))
    write_json(a.output/'data_audit.json',json.loads((source/'data_audit.json').read_text()))
    def record(name,prob,component,fixed_calibration=None):
        points=[];confidence=prob.max(-1);order=np.sort(prob,axis=-1);margins=order[:,-1]-order[:,-2]
        grid=[(fixed_calibration['threshold'],fixed_calibration['margin'])] if fixed_calibration else [(t,m) for t in [0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98] for m in [0.,.1,.2]]
        for threshold,margin in grid:
            pred=np.where((confidence>=threshold)&(margins>=margin),prob.argmax(-1),8)
            groups={k:summary(pred,splits[k],data) for k in keys}
            gates={k:evaluation_gate(groups[k],baseline[k],4) for k in keys}
            incumbent_gates={k:incumbent_evaluation_gate(groups[k],incumbent[k],4) for k in keys}
            feasible=all(v['joint_goal_met'] for v in list(gates.values())+list(incumbent_gates.values()))
            points.append(dict(threshold=threshold,margin=margin,metrics=groups['dev']['accepted'],roles=groups['dev']['roles'],sequence=groups['dev']['sequence'],
                acceptance_revision=4,joint_goal_met=bool(feasible),groups=groups,gates=gates,incumbent_gates=incumbent_gates))
        chosen=max(points,key=lambda v:(v['joint_goal_met'],v['metrics']['recall'] if v['joint_goal_met'] else v['metrics']['f1'],v['metrics']['precision']))
        value=dict(name=name,component=component,calibration={k:v for k,v in chosen.items() if k not in ['groups','gates','incumbent_gates']},
            joint_growth_goal_met=chosen['joint_goal_met'],subgroup_gates={k:chosen['gates'][k] for k in ['old_dev','new_dev']},subgroups={k:chosen['groups'][k] for k in ['old_dev','new_dev']},incumbent_gates=chosen['incumbent_gates'],
            raw={k:metrics(prob.argmax(-1),splits[k],data) for k in keys})
        registry[name]=value;write_json(a.output/(name+'_calibration.json'),points)
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob);write_json(a.output/'registry.json',registry)
        write_json(a.output/'status.json',dict(stage='calibrating',completed=len(registry),latest=name));print(name,chosen['joint_goal_met'],chosen['metrics']['correct'],chosen['metrics']['wrong_acquire'],flush=True)
    record('incumbent_frozen_configuration',old_probability,old_component,old['result']['calibration'])
    growth=json.loads((research/'frozen_for_confirm5_growth108/selected.json').read_text())
    record('growth_frozen_configuration',predict_component(growth['result']['component'],data,a.root),growth['result']['component'],growth['result']['calibration'])
    for family,name in families.items():
        new=np.load(source/(name+'_probabilities.npz'))['probability']
        for fraction in [.25,.5,.75]:
            record(f'{family}_new{fraction}',(1-fraction)*old_probability+fraction*new,
                dict(kind='blend',weights=[1-fraction,fraction],members=[old_component,original[name]['component']]))
    winner=max(registry,key=lambda name:merit(registry[name]))
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=0,incumbent_gate_required=True,holdout_evaluated=False))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
