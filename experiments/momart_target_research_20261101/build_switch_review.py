"""Unlabelled review sheets for all examined subject changes plus stable controls.

Sampling uses unchanged proxy labels; reviewer pages/JSON omit both those labels
and every model prediction. Future frames are offline review evidence only. No
annotation is created, imported into training, or used to change an old score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import h5py
import numpy as np
from data import load_group
from growth_data import load_growth_data,growth_partitions


OFFSETS=[-16,-8,0,8,16]


def physical_key(row):return str(Path(row['source']).resolve()),row['demo'],int(row['step'])


def select_cases(groups,seed=20261105):
    rng=np.random.default_rng(seed);selected=[]
    for cohort,(rows,allowed) in groups.items():
        allowed=set(map(int,allowed));episodes={rows[i]['episode_id'] for i in allowed};changes=[];stable=[]
        for episode in sorted(episodes):
            ids=sorted([i for i,r in enumerate(rows) if r['episode_id']==episode],key=lambda i:rows[i]['step'])
            by_step={rows[i]['step']:i for i in ids}
            for left,right in zip(ids,ids[1:]):
                a,b=rows[left],rows[right]
                if left not in allowed or right not in allowed or a['label'] is None or b['label'] is None:continue
                if not 0<b['step']-a['step']<=16:continue
                if a['label']['target']['instance_id']!=b['label']['target']['instance_id']:
                    changes.append(dict(cohort=cohort,row=b,kind='proxy_change',old_proxy_class=a['label']['target']['class_id'],proxy_class=b['label']['target']['class_id']))
            for i in ids:
                row=rows[i];neighbors=[by_step.get(row['step']+offset) for offset in OFFSETS]
                if any(j is None or j not in allowed or rows[j]['label'] is None for j in neighbors):continue
                if len({rows[j]['label']['target']['instance_id'] for j in neighbors})==1:stable.append(row)
        used=set()
        for change in changes:
            selected.append(change)
            candidates=[r for r in stable if physical_key(r) not in used and all(r['episode_id']!=c['row']['episode_id'] or abs(r['step']-c['row']['step'])>32 for c in changes)]
            levels=[('task_and_class',lambda r:r['task']==change['row']['task'] and r['label']['target']['class_id']==change['proxy_class']),
                ('class',lambda r:r['label']['target']['class_id']==change['proxy_class']),('task',lambda r:r['task']==change['row']['task']),('any',lambda r:True)]
            for level,condition in levels:
                pool=[r for r in candidates if condition(r)]
                if pool:
                    row=pool[int(rng.integers(len(pool)))];used.add(physical_key(row))
                    selected.append(dict(cohort=cohort,row=row,kind='stable_control',proxy_class=row['label']['target']['class_id'],matching=level));break
            else:raise ValueError('Insufficient stable controls: '+cohort)
    keys=[physical_key(x['row']) for x in selected]
    if len(set(keys))!=len(keys):raise ValueError('Repeated physical review centers across cohorts')
    rng.shuffle(selected);return selected


def reviewer_entry(case_id,row,steps):
    return dict(case_id=case_id,task=row['task'],sheet='sheets/'+case_id+'.png',
        annotations=[dict(requested_offset=offset,relative_steps=int(step-row['step']),sensor_step=int(step),
            frame_uid=hashlib.sha256(json.dumps([*physical_key(row)[:2],int(step)]).encode()).hexdigest()[:20],
            status='pending',target_class=None,confidence=None,comment='') for offset,step in zip(OFFSETS,steps)])


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    research=a.root/'momart_target_research_20261101';data=load_growth_data(a.root);splits=growth_partitions(data)
    groups={name:(data['rows'],splits[name]) for name in ['train','dev']}
    for prefix in ['confirm5','confirm6']:
        group=load_group(research/(prefix+'_corpus'),research/(prefix+'_features.npz'),research/(prefix+'_phases.json'),'examined')
        groups[prefix]=(group['rows'],np.flatnonzero(group['y']>=0))
    cases=select_cases(groups,a.seed)
    reserved=json.loads((research/'confirm7_corpus/manifest.json').read_text())
    reserved_keys={(str(Path(e['source']).resolve()),e['demo']) for e in reserved['episodes']}
    if reserved_keys&{physical_key(c['row'])[:2] for c in cases}:raise ValueError('Review includes reserved confirmation7')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.output.mkdir(parents=True);(a.output/'sheets').mkdir();(a.output/'audit_only').mkdir()
    files={};entries=[];audit=[]
    try:
        for number,case in enumerate(cases,1):
            row=case['row'];case_id=f'R{number:03d}';source=str(Path(row['source']).resolve())
            if source not in files:files[source]=h5py.File(source,'r')
            episode=files[source]['data'][row['demo']];n=len(episode['actions']);steps=[max(0,min(n-1,row['step']+o)) for o in OFFSETS]
            views=['main','wrist'] if a.wrist else ['main'];images={}
            fig,axes=plt.subplots(len(views),5,figsize=(12,2.7*len(views)),squeeze=False)
            for vi,view in enumerate(views):
                key='rgb' if view=='main' else 'rgb_wrist'
                if key not in episode['obs']:raise ValueError('Requested review camera missing: '+view)
                images[view]=[np.asarray(episode['obs'][key][step]) for step in steps]
                for ax,offset,step,rgb in zip(axes[vi],OFFSETS,steps,images[view]):
                    actual=step-row['step'];suffix=' (clipped)' if actual!=offset else ''
                    ax.imshow(rgb,interpolation='nearest');ax.axis('off');ax.set_title(view+f' {actual:+d} | step {step}'+suffix,fontsize=8)
            fig.suptitle(case_id+' | '+row['task']+'\nCenter=0; +8/+16 are offline review continuation, NOT inference inputs',fontsize=9)
            fig.tight_layout(rect=(0,0,1,.88 if a.wrist else .78));path=a.output/'sheets'/(case_id+'.png');fig.savefig(path,dpi=100);plt.close(fig)
            entry=reviewer_entry(case_id,row,steps);entry['views']=views;entry['sheet_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();entries.append(entry)
            audit.append(dict(case_id=case_id,**{k:v for k,v in case.items() if k!='row'},source=source,demo=row['demo'],center_step=row['step'],episode_id=row['episode_id'],
                sensor_steps=steps,rgb_sha256={view:[hashlib.sha256(image.tobytes()).hexdigest() for image in frames] for view,frames in images.items()}))
    finally:
        for handle in files.values():handle.close()
    (a.output/'review_cases.json').write_text(json.dumps(dict(protocol='blinded_switch_review_v1',status='unreviewed',cases=entries,
        target_class_choices={'0':'bowl','1':'door','2':'door handle','3':'rack','4':'drawer','5':'drawer handle','other':'target outside six-class vocabulary; specify in comment','none':'no intended contact target','uncertain':'insufficient evidence'},
        rule='Review current intended contact subject using task and continuation; timing semantics must be agreed before adjudication. Never copy proxy/model labels.'),indent=2))
    (a.output/'audit_only/selection.json').write_text(json.dumps(dict(seed=a.seed,selection='All eligible proxy changes in train/dev and examined confirm5/6, one stable control per change; shuffled independently of models',cases=audit),indent=2))
    (a.output/'README.md').write_text('''# Unreviewed subject-switch evidence

Open sheets/ and record answers in a COPY of review_cases.json. No old target
labels or model predictions are drawn or prefilled. Keep audit_only/ closed until
independent review is complete. That directory contains the sampling provenance,
not human ground truth. The task name is public context. Positive offsets show
recorded continuation for offline adjudication; never feed them to live inference.
When present, main/wrist rows are simultaneous public camera observations.

Every annotation starts pending. Choose class 0..5, other, none, or uncertain; do not
force a class when intention cannot be established. First agree whether switching
means beginning the approach, first contact, or confirmed grasp. These are review
materials only; no training labels or old evaluation denominators are changed.
Use other for an intended target outside the six-class vocabulary (e.g. a placement
recipient) and specify it in comment; do not turn it into none or a forced bowl.

Sampling enriches proxy changes and matched stable controls; it is not a random
frame accuracy benchmark. It excludes confirmation7. Do not report pending or
agent-inferred labels as independent human intention ground truth.
''')
    report=dict(status='unreviewed',cases=len(cases),pending_annotations=len(cases)*5,
        unique_physical_frames=len({(c['source'],c['demo'],step) for c in audit for step in c['sensor_steps']}),seed=a.seed,views=['main','wrist'] if a.wrist else ['main'],
        cohorts={name:{kind:sum(c['cohort']==name and c['kind']==kind for c in cases) for kind in ['proxy_change','stable_control']} for name in groups},
        confirmation7_episode_overlap=0,review_sha256=hashlib.sha256((a.output/'review_cases.json').read_bytes()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2))
    with zipfile.ZipFile(a.output/'review_only.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        paths=[a.output/name for name in ['README.md','review_cases.json','manifest.json']]+sorted((a.output/'sheets').glob('*.png'))
        for path in paths:archive.write(path,str(path.relative_to(a.output)))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261105);p.add_argument('--wrist',action='store_true');main(p.parse_args())
