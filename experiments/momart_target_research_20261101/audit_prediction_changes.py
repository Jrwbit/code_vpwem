"""Posthoc paired prediction/switch audit; physical intent remains unreviewed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from data import load_group
from compare_confirmation import paired_error_sets
from diagnose_intent import failure_bucket


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    score=json.loads((a.evaluation/'results.json').read_text());frozen=json.loads((a.evaluation/'frozen_evaluation_manifest.json').read_text())
    assert frozen['corpus_prefix']==a.prefix
    data=load_group(a.research/(a.prefix+'_corpus'),a.research/(a.prefix+'_features.npz'),a.research/(a.prefix+'_phases.json'),'examined')
    archives={k:np.load(a.evaluation/(name+'_predictions.npz'),allow_pickle=False) for k,name in [('primary','primary'),('reference',a.reference)]}
    for z in archives.values():assert z['ids'].tolist()==[r['id'] for r in data['rows']]
    pred={k:z['prediction'] for k,z in archives.items()};index=np.flatnonzero(data['y']>=0);known=set(map(int,index));sets=paired_error_sets(data,pred['primary'],pred['reference'],index)
    def outcome(i,key):
        slot=int(pred[key][i]);row=data['rows'][i]
        return dict(slot=slot,class_id=row['objects'][slot]['class_id'] if slot<8 else None,outcome=failure_bucket(i,pred[key],data),probability=archives[key]['probability'][i].tolist())
    def frame(i):
        row=data['rows'][i]
        return dict(id=row['id'],source=row['source'],demo=row['demo'],episode_id=row['episode_id'],step=row['step'],role=int(data['roles'][i]),target=row['label']['target'],
            positive_slots=np.flatnonzero(data['positive'][i]).tolist(),objects=row['objects'],primary=outcome(i,'primary'),reference=outcome(i,'reference'))
    examples=[dict(frame(i),changes=[key for key,ids in sets.items() if i in ids]) for i in sorted({i for ids in sets.values() for i in ids})]
    transitions=[]
    for ep in sorted({r['episode_id'] for r in data['rows']}):
        ids=sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==ep],key=lambda i:data['rows'][i]['step'])
        for left,right in zip(ids,ids[1:]):
            if left not in known or right not in known or not 0<data['rows'][right]['step']-data['rows'][left]['step']<=16:continue
            old=data['rows'][left]['label']['target'];new=data['rows'][right]['label']['target']
            if old['instance_id']==new['instance_id']:continue
            item=frame(right);item['previous_id']=data['rows'][left]['id'];item['previous_target']=old
            for key in pred:
                slot=int(pred[key][right]);item[key]['correct_at_change']=bool(slot<8 and data['positive'][right,slot])
                item[key]['stale_previous_class']=bool(slot<8 and old['class_id']!=new['class_id'] and item[key]['class_id']==old['class_id'])
            transitions.append(item)
    for key,name in [('primary','primary'),('reference',a.reference)]:
        expected=score['results'][name]['sequence'];assert len(transitions)==expected['gold_subject_changes']
        assert sum(x[key]['correct_at_change'] for x in transitions)==expected['correct_box_at_change']
        assert sum(x[key]['stale_previous_class'] for x in transitions)==expected['stale_previous_class_at_change']
    payload=dict(prefix=a.prefix,reference=a.reference,total_frames=len(data['rows']),known_frames=len(index),counts={k:len(v) for k,v in sets.items()},examples=examples,transitions=transitions,
        physical_intent_review='not performed by this audit; no physical-ambiguity exemptions',
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.evaluation/'results.json',a.evaluation/'primary_predictions.npz',a.evaluation/(a.reference+'_predictions.npz')]},
        caveat='Examined proxy-label diagnosis only; all gains/losses/transitions retained; no targets changed, no threshold retuning, no independent intent or policy-success claim')
    a.output.write_text(json.dumps(payload,indent=2));print(json.dumps(dict(counts=payload['counts'],transition_changes=[x for x in transitions if x['primary']['correct_at_change']!=x['reference']['correct_at_change'] or x['primary']['stale_previous_class']!=x['reference']['stale_previous_class']]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['research','evaluation','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prefix',required=True);p.add_argument('--reference',choices=['baseline','control','retention'],default='retention');main(p.parse_args())
