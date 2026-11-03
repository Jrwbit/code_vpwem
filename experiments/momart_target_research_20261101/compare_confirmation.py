"""Paired posthoc error changes, preserving every known evaluation frame.

Reports both gains and losses. Privileged audit flags remain review evidence,
never labels or online input; an examined confirmation is not fresh again.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark import write_json
from data import load_group
from diagnose_intent import failure_bucket,frame_key,subject_changes


def paired_error_sets(data,primary,reference,index):
    def hit(i,p):return p[i]<8 and data['positive'][i,p[i]]
    def wrong(i,p,role):return data['roles'][i]==role and p[i]<8 and data['rows'][i]['objects'][p[i]]['class_id']!=0
    return dict(correct_gained=[int(i) for i in index if hit(i,primary) and not hit(i,reference)],
        correct_lost=[int(i) for i in index if hit(i,reference) and not hit(i,primary)],
        acquire_added=[int(i) for i in index if wrong(i,primary,0) and not wrong(i,reference,0)],
        acquire_fixed=[int(i) for i in index if wrong(i,reference,0) and not wrong(i,primary,0)],
        carry_added=[int(i) for i in index if wrong(i,primary,4) and not wrong(i,reference,4)],
        carry_fixed=[int(i) for i in index if wrong(i,reference,4) and not wrong(i,primary,4)])


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    data=load_group(a.research/(a.prefix+'_corpus'),a.research/(a.prefix+'_features.npz'),a.research/(a.prefix+'_phases.json'),'examined')
    archives={name:np.load(path,allow_pickle=False) for name,path in [('primary',a.primary),('reference',a.reference)]}
    for archive in archives.values():assert archive['ids'].tolist()==[r['id'] for r in data['rows']]
    audit=json.loads(a.audit.read_text());lookup={frame_key(row):row for row in audit['records']};assert len(lookup)==len(audit['records'])
    flags=[]
    for row in data['rows']:
        z=lookup[frame_key(row)];assert z['old_holding_reproduced']
        flags.append(bool(z['old_held_with_carrier_conflict'] or z['all_old_future_holds_carrier_conflicted']))
    flags=np.asarray(flags);p=archives['primary']['prediction'];r=archives['reference']['prediction'];index=np.flatnonzero(data['y']>=0)
    sets=paired_error_sets(data,p,r,index);examples=[]
    for i in sorted(set(i for indices in sets.values() for i in indices)):
        row=data['rows'][i]
        def outcome(name,slot):
            return dict(slot=int(slot),class_id=row['objects'][slot]['class_id'] if slot<8 else None,
                outcome=failure_bucket(i,archives[name]['prediction'],data),probability=archives[name]['probability'][i].tolist())
        examples.append(dict(id=row['id'],source=row['source'],demo=row['demo'],step=row['step'],episode_id=row['episode_id'],role=int(data['roles'][i]),
            target=row['label']['target'],label_reason=row['label']['reason'],carrier_conflict_review=bool(flags[i]),
            changes=[name for name,ids in sets.items() if i in ids],primary=outcome('primary',p[i]),reference=outcome('reference',r[i])))
    payload=dict(prefix=a.prefix,aligned_frames=len(data['rows']),known_frames=len(index),counts={k:len(v) for k,v in sets.items()},examples=examples,
        primary_outcomes=dict(Counter(failure_bucket(i,p,data) for i in index)),reference_outcomes=dict(Counter(failure_bucket(i,r,data) for i in index)),
        all_primary_subject_changes=subject_changes(data,p,flags),all_reference_subject_changes=subject_changes(data,r,flags),
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.primary,a.reference,a.audit]},
        caveat='Examined weak-proxy regression diagnosis only. Includes gains and losses with unchanged denominators. Physical conflicts are not independent intention GT; no labels changed or score retuned.')
    write_json(a.output,payload);print(json.dumps(dict(prefix=a.prefix,counts=payload['counts'],examples=len(examples)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['research','primary','reference','audit','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prefix',required=True);main(p.parse_args())
