"""Select comparable growth-dev screens, requiring incumbent-relative V4 too.

All thresholds/checkpoints remain development-selected. The additional gate is
registered before fresh confirmation labels/features are read, not retrofitted
to excuse failures on examined cohorts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark import write_json
from confirm import apply_calibration,incumbent_evaluation_gate,predict_component
from data import metrics
from growth_data import load_growth_data,growth_partitions
from refine import role_breakdown
from sequence_metrics import sequence_metrics


def require_same_development(audit,reference):
    if audit['episodes']!=reference['episodes'] or audit['counts']!=reference['counts']:
        raise ValueError('Development episode membership/counts differ')


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    reserved=json.loads((a.holdout/'manifest.json').read_text())
    if any((a.holdout/p).exists() for p in ['annotations_all.json','candidate_manifest.json']):
        raise ValueError('Fresh confirmation already examined')
    data=load_growth_data(a.root);splits=growth_partitions(data)
    if {r['episode_id'] for r in data['rows']}&{ep['episode_id'] for ep in reserved['episodes']}:
        raise ValueError('Fresh confirmation overlaps examined data')
    control=json.loads((a.control/'selected.json').read_text())
    prob=predict_component(control['result']['component'],data,a.root)
    pred=apply_calibration(prob,control['result']['calibration'])
    def summarize(pred,indices):
        return dict(accepted=metrics(pred,indices,data),roles=role_breakdown(pred,indices,data),sequence=sequence_metrics(pred,data,indices))
    keys=['dev','old_dev','new_dev'];incumbent={key:summarize(pred,splits[key]) for key in keys}
    reference=json.loads((a.screens[0]/'data_audit.json').read_text());candidates=[];comparisons=[];sources=[]
    for screen in a.screens:
        if json.loads((screen/'status.json').read_text())['stage']!='complete':
            raise ValueError('Search has not completed: '+str(screen))
        audit=json.loads((screen/'data_audit.json').read_text());require_same_development(audit,reference)
        registry=json.loads((screen/'registry.json').read_text())
        sources.append(dict(path=str(screen),records=len(registry),sha256=hashlib.sha256((screen/'registry.json').read_bytes()).hexdigest()))
        for name,value in registry.items():
            prediction=apply_calibration(np.load(screen/(name+'_probabilities.npz'))['probability'],value['calibration'])
            checked={key:incumbent_evaluation_gate(summarize(prediction,splits[key]),incumbent[key],4) for key in keys}
            eligible=bool(value['joint_growth_goal_met'] and all(x['joint_goal_met'] for x in checked.values()))
            comparisons.append(dict(screen=str(screen),name=name,eligible=eligible,incumbent_gates=checked))
            if eligible:candidates.append((screen,name,dict(value,incumbent_gates=checked)))
    a.output.mkdir(parents=True)
    write_json(a.output/'incumbent_comparisons.json',dict(incumbent=incumbent,records=comparisons,sources=sources,
        policy='Original V4 against original pointer AND original frozen incumbent, combined24 and old8/new16 separately',
        control_selection_sha256=hashlib.sha256((a.control/'selected.json').read_bytes()).hexdigest(),
        holdout_manifest_sha256=hashlib.sha256((a.holdout/'manifest.json').read_bytes()).hexdigest()))
    if not candidates:
        write_json(a.output/'status.json',dict(stage='no_eligible_candidate'));print('No candidate passes incumbent-relative gates');return
    def merit(item):
        m=item[2]['calibration']['metrics'];return m['recall'],-m['wrong_acquire'],m['precision']
    screen,name,winner=max(candidates,key=merit)
    write_json(a.output/'selected.json',dict(name=name,result=winner,source_screen=str(screen),screens=sources,
        method_count=sum(x['records'] for x in sources),acceptance_revision=4,incumbent_gate_required=True,
        selection_partition='combined24-dev with old8/new16 and incumbent-relative V4 gates',holdout_evaluated=False))
    write_json(a.output/'status.json',dict(stage='selected',winner=name,eligible=len(candidates)))
    print(json.dumps(dict(winner=name,eligible=len(candidates),metrics=winner['calibration']['metrics'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','control','holdout','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--screens',type=Path,nargs='+',required=True);main(p.parse_args())
