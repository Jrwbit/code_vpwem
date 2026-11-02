"""Public predicted-class rejection calibration on unchanged growth-dev24 only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark import write_json
from benchmark_growth_blend import summary
from calibration import apply_calibration
from confirm import evaluation_gate,incumbent_evaluation_gate,predict_component
from growth_data import load_growth_data,growth_partitions
from data import metrics


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);r=a.root/'momart_target_research_20261101';holdout=r/'confirm7_corpus'
    if any((holdout/name).exists() for name in ['annotations_all.json','candidate_manifest.json']):raise ValueError('Confirmation7 already examined')
    data=load_growth_data(a.root);splits=growth_partitions(data);keys=['dev','old_dev','new_dev'];classes=data['candidates'][:,:,133:].argmax(-1)
    if {row['episode_id'] for row in data['rows']}&{ep['episode_id'] for ep in json.loads((holdout/'manifest.json').read_text())['episodes']}:raise ValueError('Confirmation overlap')
    parent=json.loads((r/'frozen_for_confirm6_incumbent_blend/selected.json').read_text());component=parent['result']['component'];prob=predict_component(component,data,a.root)
    old=json.loads((r/'frozen_for_confirm4/selected.json').read_text());old_prob=predict_component(old['result']['component'],data,a.root)
    old_pred=apply_calibration(old_prob,old['result']['calibration'],classes)
    previous=predict_component(dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100')),data,a.root).argmax(-1)
    incumbent={k:summary(old_pred,splits[k],data) for k in keys};baseline={k:summary(previous,splits[k],data) for k in keys}
    write_json(a.output/'frozen_search_protocol.json',dict(method='predicted bowl/non-bowl rejection thresholds',
        bowl_thresholds=[0.,.25,.4,.5,.6,.7,.8],other_thresholds=[.4,.5,.6,.7,.8],margins=[0.,.1,.2],class_margins=a.class_margins,fit_count=0,
        selection='Unchanged dual-comparator combined24 and old8/new16 V4 gates, then recall, acquisition errors, precision',
        parent_selection_sha256=hashlib.sha256((r/'frozen_for_confirm6_incumbent_blend/selected.json').read_bytes()).hexdigest(),
        holdout='confirm7 selector-unseen only; confirm6 examined and never used for calibration',
        holdout_manifest_sha256=hashlib.sha256((holdout/'manifest.json').read_bytes()).hexdigest(),
        public_class_source='Predicted proposal class at argmax slot; no reference class or phase label',
        manifest_sha256=hashlib.sha256((r/'growth_corpus/manifest.json').read_bytes()).hexdigest()))
    write_json(a.output/'data_audit.json',json.loads((r/'growth_incumbent_blends_v4/data_audit.json').read_text()))
    registry={}
    def record(name,prob,component,cals):
        points=[]
        for cal in cals:
            pred=apply_calibration(prob,cal,classes);groups={k:summary(pred,splits[k],data) for k in keys}
            gates={k:evaluation_gate(groups[k],baseline[k],4) for k in keys}
            incumbent_gates={k:incumbent_evaluation_gate(groups[k],incumbent[k],4) for k in keys}
            feasible=all(v['joint_goal_met'] for v in list(gates.values())+list(incumbent_gates.values()))
            points.append(dict(cal,metrics=groups['dev']['accepted'],roles=groups['dev']['roles'],sequence=groups['dev']['sequence'],
                acceptance_revision=4,joint_goal_met=bool(feasible),groups=groups,gates=gates,incumbent_gates=incumbent_gates))
        def merit(v):
            m=v['metrics'];return v['joint_goal_met'],m['recall'] if v['joint_goal_met'] else m['f1'],-m['wrong_acquire'],m['precision']
        chosen=max(points,key=merit)
        registry[name]=dict(name=name,component=component,calibration={k:v for k,v in chosen.items() if k not in ['groups','gates','incumbent_gates']},
            joint_growth_goal_met=chosen['joint_goal_met'],subgroup_gates={k:chosen['gates'][k] for k in ['old_dev','new_dev']},
            subgroups={k:chosen['groups'][k] for k in ['old_dev','new_dev']},incumbent_gates=chosen['incumbent_gates'],
            raw={k:metrics(prob.argmax(-1),splits[k],data) for k in keys})
        write_json(a.output/(name+'_calibration.json'),points);np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob)
        write_json(a.output/'registry.json',registry);write_json(a.output/'status.json',dict(stage='calibrating',completed=len(registry),latest=name))
        print(name,chosen['joint_goal_met'],chosen['metrics']['correct'],chosen['metrics']['wrong_acquire'],flush=True)
    record('incumbent_frozen_configuration',old_prob,old['result']['component'],[old['result']['calibration']])
    record('blend_frozen_configuration',prob,component,[parent['result']['calibration']])
    for bowl in [0.,.25,.4,.5,.6,.7,.8]:
        for other in [.4,.5,.6,.7,.8]:
            cals=[dict(threshold=other,margin=margin,class_thresholds={'0':bowl}) for margin in [0.,.1,.2]]
            if a.class_margins:cals=[dict(cal,class_margins={'0':gap}) for cal in cals for gap in [0.,.1,.2]]
            record(f'bowl{bowl}_other{other}',prob,component,cals)
    def merit(name):
        v=registry[name];m=v['calibration']['metrics'];return v['joint_growth_goal_met'],m['recall'] if v['joint_growth_goal_met'] else m['f1'],-m['wrong_acquire'],m['precision']
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=0,incumbent_gate_required=True,holdout_evaluated=False,
        selection_partition='combined24-dev with dual-comparator old8/new16 gates',frozen_blend_retained=winner=='blend_frozen_configuration'))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner));print('WINNER',winner,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--class-margins',action='store_true');main(p.parse_args())
