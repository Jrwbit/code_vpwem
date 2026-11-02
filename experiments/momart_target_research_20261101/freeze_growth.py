"""Register new train/dev episodes before labels or model search.

Exclude all examined episodes and the untouched confirmation set. New episodes
come only from source train masks; split whole episodes, never individual frames.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def select_growth(pool,excluded,task,train_count,dev_count,rng):
    if train_count<=0 or dev_count<=0:raise ValueError('Both episode groups must be nonempty')
    choices=sorted({str(demo) for demo in pool if task+'/'+str(demo) not in excluded})
    if len(choices)<train_count+dev_count:raise ValueError('Insufficient unused source-train episodes: '+task)
    drawn=rng.choice(choices,train_count+dev_count,replace=False).tolist()
    return [(demo,'train' if i<train_count else 'dev') for i,demo in enumerate(drawn)]


def decode(values):return [x.decode() if isinstance(x,bytes) else str(x) for x in values]


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    reference=json.loads((a.reference/'manifest.json').read_text())
    docs=[reference]+[json.loads((p/'manifest.json').read_text()) for p in a.exclude]
    excluded={ep['episode_id'] for doc in docs for ep in doc['episodes']}
    rng=np.random.default_rng(a.seed);episodes=[]
    # Validate every pool before writing images or output directories.
    for task in sorted({ep['task'] for ep in reference['episodes']}):
        source=next(ep['source'] for ep in reference['episodes'] if ep['task']==task)
        with h5py.File(source,'r') as f:
            train=decode(f['mask/train'][:]);valid=set(decode(f['mask/valid'][:]))
            if set(train)&valid:raise ValueError('Source train/valid overlap')
            for demo,split in select_growth(train,excluded,task,a.train_per_task,a.dev_per_task,rng):
                episodes.append(dict(source=source,task=task,demo=demo,split=split,episode_id=task+'/'+demo,
                    length=len(f['data'][demo]['actions']),source_partition='train'))
    a.output.mkdir();(a.output/'images').mkdir();records=[]
    for ei,ep in enumerate(episodes):
        with h5py.File(ep['source'],'r') as f:
            for step in range(0,ep['length'],a.stride):
                image=np.asarray(f['data'][ep['demo']]['obs/rgb'][step]);key=f'growth_{ei}_{step}';file='images/'+key+'.png'
                Image.fromarray(image).save(a.output/file)
                records.append(dict(ep,id=key,step=step,file=file,anchor=True,clip=False,sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    manifest=dict(reference,episodes=episodes,records=records,protocol='frozen_growth_train_dev_v1',seed=a.seed,
        train_per_task=a.train_per_task,dev_per_task=a.dev_per_task,stride=a.stride,source_pool='source_train_only',
        excluded_episodes=sorted(excluded),exclusions_sha256=hashlib.sha256('\n'.join(sorted(excluded)).encode()).hexdigest(),
        purpose='New selector train/dev; not prospective confirmation. Old examined regressions and confirm5 remain excluded.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(train_episodes=sum(ep['split']=='train' for ep in episodes),dev_episodes=sum(ep['split']=='dev' for ep in episodes),frames=len(records),excluded_episodes=len(excluded))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--exclude',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--train-per-task',type=int,default=20);p.add_argument('--dev-per-task',type=int,default=4)
    p.add_argument('--stride',type=int,default=8);p.add_argument('--seed',type=int,default=20261104);main(p.parse_args())
