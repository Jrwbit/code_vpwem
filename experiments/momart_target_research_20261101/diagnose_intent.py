"""Post-hoc intent/grounding decomposition on unchanged archived predictions.

This does not select, train, relabel, or certify a deployed model. Oracle class
queries read privileged labels and must never be used as autonomous results.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from data import load_group,metrics
from query_grounding import ground_query


def frame_key(row):
    return str(Path(row['source']).resolve()),row['demo'],int(row['step'])


def failure_bucket(i,pred,data):
    slot=int(pred[i])
    if slot<8 and data['positive'][i,slot]:return 'correct'
    if not data['positive'][i].any():return 'no_matching_instance_proposal'
    if slot==8:return 'rejected_matching_proposal'
    target=data['rows'][i]['label']['target']
    if data['rows'][i]['objects'][slot]['class_id']!=target['class_id']:return 'wrong_class_with_matching_proposal'
    return 'wrong_instance_or_box_with_matching_proposal'


def subject_changes(data,pred,flags,max_gap_steps=16):
    """Return every eligible change, including successes, without bridging unknowns."""
    changes=[]
    for episode in sorted({r['episode_id'] for r in data['rows']}):
        ids=sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==episode],key=lambda i:data['rows'][i]['step'])
        for left,right in zip(ids,ids[1:]):
            a,b=data['rows'][left],data['rows'][right]
            if a['label'] is None or b['label'] is None or not 0<b['step']-a['step']<=max_gap_steps:continue
            old,new=a['label']['target'],b['label']['target']
            if old['instance_id']==new['instance_id']:continue
            slot=int(pred[right]);cls=b['objects'][slot]['class_id'] if slot<8 else None
            changes.append(dict(episode_id=episode,id=b['id'],demo=b['demo'],step=b['step'],previous_step=a['step'],
                old_class=old['class_id'],new_class=new['class_id'],predicted_class=cls,
                outcome=failure_bucket(right,pred,data),carrier_conflict_review=bool(flags[right]),
                stale_previous_class=bool(cls is not None and old['class_id']!=new['class_id'] and cls==old['class_id'])))
    return changes


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    data=load_group(a.research/(a.prefix+'_corpus'),a.research/(a.prefix+'_features.npz'),a.research/(a.prefix+'_phases.json'),'confirm')
    archive=np.load(a.predictions,allow_pickle=False)
    assert archive['ids'].tolist()==[r['id'] for r in data['rows']]
    prob=archive['probability'];pred=archive['prediction'];n=len(data['rows'])
    classes=np.full((n,8),-1,int);query=np.full(n,-1,int)
    for i,row in enumerate(data['rows']):
        for j,obj in enumerate(row['objects']):classes[i,j]=obj['class_id']
        if row['label'] is not None:query[i]=row['label']['target']['class_id']
    index=np.flatnonzero(data['y']>=0)
    oracle=ground_query(prob,classes,data['valid'],query,keep_null=False).argmax(-1)
    audited=json.loads(a.audit.read_text())
    lookup={frame_key(row):row for row in audited['records']}
    assert len(lookup)==len(audited['records'])
    flags=[]
    for row in data['rows']:
        audit=lookup[frame_key(row)]
        assert audit['old_holding_reproduced']
        flags.append(bool(audit['old_held_with_carrier_conflict'] or audit['all_old_future_holds_carrier_conflicted']))
    flags=np.asarray(flags)
    subsets={'all_known':index,'carrier_conflict_review':index[flags[index]],'other_known':index[~flags[index]]}
    report={}
    for name,indices in subsets.items():
        report[name]=dict(original=metrics(pred,indices,data),oracle_class_ranker=metrics(oracle,indices,data),
            original_outcomes=dict(Counter(failure_bucket(i,pred,data) for i in indices)),
            oracle_outcomes=dict(Counter(failure_bucket(i,oracle,data) for i in indices)))
    examples=[]
    for i in index:
        bucket=failure_bucket(i,pred,data)
        if bucket=='correct' and not flags[i]:continue
        examples.append(dict(id=data['rows'][i]['id'],source=data['rows'][i]['source'],demo=data['rows'][i]['demo'],step=data['rows'][i]['step'],
            role=int(data['roles'][i]),query_class=int(query[i]),original_slot=int(pred[i]),oracle_slot=int(oracle[i]),
            outcome=bucket,oracle_outcome=failure_bucket(i,oracle,data),carrier_conflict_review=bool(flags[i]),
            candidate_classes=classes[i].tolist(),probability=prob[i].tolist()))
    payload=dict(protocol='posthoc_intent_grounding_diagnosis_v1',prefix=a.prefix,total_frames=n,known_frames=len(index),unknown_frames=n-len(index),
        subsets=report,examples=examples,subject_changes=subject_changes(data,pred,flags),input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.predictions,a.audit]},
        caveat='Examined regression only. Oracle class reads weak labels, not autonomous intent. Same-class instance/box ranking unchanged. Null removal is an optimistic grounding diagnostic, not calibrated deployment. Carrier conflicts remain in every full-denominator metric; no label changed. Review flags do not prove intent errors.')
    a.output.write_text(json.dumps(payload,indent=2))
    print(json.dumps(dict(prefix=a.prefix,total_frames=n,known_frames=len(index),subsets=report),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['research','predictions','audit','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prefix',required=True);main(p.parse_args())
