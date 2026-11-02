"""Explicitly promote examined confirmation cohorts to training/development.

Their historical prospective results stay archived; they are never called holdout
again for models trained with this loader. Confirmation 3 remains excluded.
"""
import numpy as np
from benchmark_dense import load_dense_data
from data import load_group,attach_geometry,combine,metrics
from mask_benchmark import attach_masks
from refine import role_breakdown
from sequence_metrics import sequence_metrics


def load_expanded_data(root):
    r=root/'momart_target_research_20261101'
    base=attach_masks(load_dense_data(root),[r/'mask_dense.npz',r/'mask_old.npz',r/'mask_new.npz']);groups=[base]
    for prefix in ['confirmation','confirm2']:
        cohort=attach_geometry(load_group(r/(prefix+'_corpus'),r/(prefix+'_features.npz'),r/(prefix+'_phases.json'),'old',r/('extra_'+prefix+'.npz')),r/('geometry_'+prefix+'.npz'))
        cohort=attach_masks(cohort,[r/('mask_'+prefix+'.npz')]);assignments={}
        tasks=sorted({row['task'] for row in cohort['rows']})
        for task in tasks:
            episodes=sorted({row['episode_id'] for row in cohort['rows'] if row['task']==task})
            assert len(episodes)==2
            for rank,ep in enumerate(episodes):assignments[ep]='dev' if prefix=='confirmation' and rank==1 else 'train'
        cohort['rows']=[dict(row,id=prefix+'::'+row['id'],source_id=row['id'],source_split=row['split'],split=assignments[row['episode_id']],tag='old',cohort=prefix+'_promoted') for row in cohort['rows']]
        groups.append(cohort)
    return combine(groups)


def baseline_for(prob,index,data):
    pred=prob.argmax(-1)
    return dict(metrics=metrics(pred,index,data),roles=role_breakdown(pred,index,data),sequence=sequence_metrics(pred,data,index))


def priority_gate(value,seq,baseline):
    b=baseline['metrics'];s=baseline['sequence']
    acquire_limit=max(0,b['wrong_acquire']-1) if b['acquire_samples']>=10 else b['wrong_acquire']
    acquisition=value['wrong_acquire']<=acquire_limit;carry=value['wrong_carry']<=b['wrong_carry']
    chatter=(seq['stable_class_switch_rate'] or 0)<=(s['stable_class_switch_rate'] or 0)
    transition=(seq['stale_previous_class_at_change']<=s['stale_previous_class_at_change'] and seq['correct_box_at_change']>=s['correct_box_at_change']) if s['gold_subject_changes']>=3 else True
    return dict(priority_errors_improved=bool(acquisition and carry),switching_retained=bool(chatter and transition),
        acquisition_error_limit=acquire_limit,transition_gate_enabled=s['gold_subject_changes']>=3)


def calibrate_v4(prob,index,data,baseline):
    raw=prob.argmax(-1);confidence=prob.max(-1);order=np.sort(prob,axis=-1);margin=order[:,-1]-order[:,-2];points=[]
    floors={role:max(0.,(v['recall'] or 0)-.05) for role,v in baseline['roles'].items() if v['visible']>=10}
    for threshold in [0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98]:
        for gap in [0.,.1,.2]:
            pred=np.where((confidence>=threshold)&(margin>=gap),raw,8);m=metrics(pred,index,data);roles=role_breakdown(pred,index,data);seq=sequence_metrics(pred,data,index)
            if m['selected']<10:continue
            covered=all(v['correct']>=1 for v in roles.values() if v['visible']>=3);retained=all((roles[role]['recall'] or 0)>=floor for role,floor in floors.items())
            priority=priority_gate(m,seq,baseline)
            points.append(dict(threshold=threshold,margin=gap,metrics=m,roles=roles,sequence=seq,critical_roles_covered=covered,role_recall_retained=retained,**priority))
    feasible=[v for v in points if (v['metrics']['precision'] or 0)>=.9 and (v['metrics']['recall'] or 0)>=baseline['metrics']['recall'] and v['critical_roles_covered'] and v['role_recall_retained'] and v['priority_errors_improved'] and v['switching_retained']]
    if feasible:chosen=max(feasible,key=lambda v:(v['metrics']['recall'],v['metrics']['precision'],-v['threshold'],-v['margin']))
    else:
        covered=[v for v in points if v['critical_roles_covered'] and v['role_recall_retained'] and v['priority_errors_improved'] and v['switching_retained']]
        chosen=max(covered or points,key=lambda v:(v['metrics']['f1'],v['metrics']['recall'] or 0,-v['threshold'],-v['margin']))
    return dict(chosen,joint_goal_met=bool(feasible),baseline_recall=baseline['metrics']['recall'],role_recall_floors=floors,acceptance_revision=4)
