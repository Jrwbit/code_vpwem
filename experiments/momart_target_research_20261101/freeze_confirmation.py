"""Freeze independent confirmation IDs before model search; no rendering or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def main(a):
    previous=[json.loads((path/'manifest.json').read_text()) for path in a.exclude]
    excluded={ep['episode_id'] for doc in previous for ep in doc['episodes']}
    rng=np.random.default_rng(20261103);episodes=[];rows=[]
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'images').mkdir()
    for task in sorted({ep['task'] for ep in previous[0]['episodes']}):
        source=next(ep['source'] for ep in previous[0]['episodes'] if ep['task']==task)
        with h5py.File(source,'r') as f:
            names=[x.decode() if isinstance(x,bytes) else str(x) for x in f['mask/valid'][:]]
            pool=sorted(x for x in names if task+'/'+x not in excluded)
            if len(pool)<a.episodes_per_task:
                if a.source_pool!='valid-first':raise ValueError('Insufficient unused valid episodes for '+task)
                remaining=sorted(x for x in f['data'] if task+'/'+x not in excluded and x not in pool)
                if len(remaining)<a.episodes_per_task-len(pool):raise ValueError('Insufficient unused source episodes for '+task)
                chosen=list(pool)+list(rng.choice(remaining,a.episodes_per_task-len(pool),replace=False))
            else:chosen=rng.choice(pool,a.episodes_per_task,replace=False)
            for demo in chosen:
                demo=str(demo);ep=f['data'][demo];n=len(ep['actions'])
                episodes.append(dict(source=source,task=task,demo=demo,split='confirm',episode_id=task+'/'+demo,length=n,source_partition='valid' if demo in names else 'nonvalid_selector_unseen'))
                for step in range(0,n,8):
                    image=np.asarray(ep['obs/rgb'][step]);key=f'{a.id_prefix}_{len(episodes)-1}_{step}';file='images/'+key+'.png'
                    Image.fromarray(image).save(a.output/file)
                    rows.append(dict(id=key,episode_id=task+'/'+demo,source=source,task=task,demo=demo,split='confirm',step=step,
                        file=file,anchor=True,clip=False,sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    doc=previous[0].copy();doc.update(episodes=episodes,records=rows,seed=20261103,
        protocol='frozen_subject_confirmation_v1',id_prefix=a.id_prefix,source_pool=a.source_pool,episodes_per_task=a.episodes_per_task,excluded_episodes=sorted(excluded),
        exclusions_sha256=hashlib.sha256('\n'.join(sorted(excluded)).encode()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(doc,indent=2))
    print(json.dumps(dict(episodes=episodes,frames=len(rows))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exclude',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--id-prefix',default='confirm');p.add_argument('--source-pool',choices=['valid','valid-first'],default='valid');p.add_argument('--episodes-per-task',type=int,default=2);main(p.parse_args())
