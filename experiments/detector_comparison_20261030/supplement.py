"""Add drawer training coverage and untouched confirmation episodes.

Selection uses episode identities and uniformly spaced times, never test errors.
The first pilot's dev/test remain in their original splits. New validation-source
confirmation episodes are excluded from training and threshold calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from PIL import Image

DRAWER_TASKS = ['table_setup_from_dresser_expert',
                'unload_dishwasher_to_dresser_suboptimal']


def export(base, output, source):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((base/'manifest.json').read_text())
    rng = np.random.default_rng(20261101)
    records, episodes = [], []
    used = {e['episode_id'] for e in manifest['episodes']}
    old_frames = {(r['episode_id'], r['step']) for r in manifest['records']}
    (output/'images').mkdir(parents=True)
    for task in DRAWER_TASKS:
        path = source/(task+'.hdf5')
        with h5py.File(path, 'r') as f:
            decode = lambda xs: [x.decode() if isinstance(x, bytes) else str(x) for x in xs]
            train = [d for d in decode(f['mask/train'][:]) if task+'/'+d not in used]
            valid = [d for d in decode(f['mask/valid'][:]) if task+'/'+d not in used]
            chosen = [(e['demo'], 'train', 32) for e in manifest['episodes']
                      if e['task']==task and e['split']=='train']
            chosen += [(d, 'train', 32) for d in rng.choice(sorted(train), 2, replace=False)]
            chosen += [(d, 'test', 16) for d in rng.choice(sorted(valid), 1, replace=False)]
            for demo, split, count in chosen:
                ep=f['data'][demo]; n=len(ep['actions']); identity=task+'/'+demo
                steps=sorted(set(np.linspace(0,n-1,count,dtype=int).tolist()))
                episodes.append(dict(source=str(path),task=task,demo=demo,split=split,
                                     episode_id=identity,length=n,anchors=steps,clip=[]))
                for step in steps:
                    if (identity,step) in old_frames: continue
                    im=np.asarray(ep['obs/rgb'][step]); assert im.shape==(90,120,3)
                    frame_id=f'supp_{DRAWER_TASKS.index(task)}_{demo}_{step}'
                    filename='images/'+frame_id+'.png'; Image.fromarray(im).save(output/filename)
                    records.append(dict(id=frame_id,episode_id=identity,source=str(path),task=task,
                        demo=demo,step=step,split=split,file=filename,anchor=True,clip=False,
                        sha256=hashlib.sha256(im.tobytes()).hexdigest(),round='drawer_supplement_v1'))
    result=dict(seed=20261101,image_hw=[90,120],classes=manifest['classes'],episodes=episodes,
                records=records,protocol='episode_disjoint_drawer_supplement_v1',
                base_manifest_sha256=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(frames=len(records), splits={s:sum(r['split']==s for r in records)
                                                      for s in ('train','test')})),flush=True)


def merge(base, supplement, output):
    if output.exists(): raise FileExistsError(output)
    first=json.loads((base/'manifest.json').read_text())
    second=json.loads((supplement/'manifest.json').read_text())
    assert first['classes']==second['classes']
    assert second['base_manifest_sha256']==hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()
    manifest=dict(first);manifest['records']=first['records']+second['records']
    episodes={e['episode_id']:e for e in first['episodes']}
    for e in second['episodes']:
        if e['episode_id'] in episodes:
            old=episodes[e['episode_id']];assert old['split']==e['split']
            episodes[e['episode_id']]=dict(old,anchors=sorted(set(old['anchors']+e['anchors'])))
        else: episodes[e['episode_id']]=e
    manifest['episodes']=list(episodes.values());manifest['protocol']='episode_disjoint_drawer_augmented_v1'
    assert len({r['id'] for r in manifest['records']})==len(manifest['records'])
    splits={s:{e['episode_id'] for e in episodes.values() if e['split']==s} for s in ('train','dev','test')}
    assert not (splits['train']&splits['dev'] or splits['train']&splits['test'] or splits['dev']&splits['test'])
    output.mkdir(parents=True)
    annotations=[];rejected=[]
    for root in (base,supplement):
        labels=json.loads((root/'annotations_all.json').read_text())
        annotations+=labels['annotations'];rejected+=labels.get('rejected',[])
        for directory in ('images','masks'):
            shutil.copytree(root/directory,output/directory,dirs_exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Preserve the annotation producer's schema and review status.
    labels=dict(labels);labels['annotations']=annotations;labels['rejected']=rejected
    (output/'annotations_all.json').write_text(json.dumps(labels,indent=2)+'\n')
    print(json.dumps(dict(accepted=len(annotations),rejected=len(rejected))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--merge',type=Path);a=p.parse_args()
    if a.merge: merge(a.base,a.merge,a.output)
    else: export(a.base,a.output,a.source)
