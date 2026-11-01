"""Freeze a new evaluation subset before fitting stage models; no error-based selection."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def main(a):
    previous=json.loads((a.previous/'manifest.json').read_text())
    excluded={ep['episode_id'] for ep in previous['episodes']}
    rng=np.random.default_rng(20261102);episodes=[];records=[]
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'images').mkdir()
    for task in sorted({row['task'] for row in previous['records']}):
        source=next(ep['source'] for ep in previous['episodes'] if ep['task']==task)
        with h5py.File(source,'r') as f:
            names=[x.decode() if isinstance(x,bytes) else str(x) for x in f['mask/valid'][:]]
            pool=sorted(name for name in names if task+'/'+name not in excluded)
            if len(pool)<2:raise ValueError('Not enough new valid episodes: '+task)
            for demo in rng.choice(pool,2,replace=False):
                demo=str(demo);ep=f['data'][demo];n=len(ep['actions']);identity=task+'/'+demo
                episodes.append(dict(source=source,task=task,demo=demo,split='test',episode_id=identity,length=n))
                for step in range(0,n,16):
                    key=f'new_{len(episodes)-1}_{step}';image=np.asarray(ep['obs/rgb'][step]);file='images/'+key+'.png'
                    Image.fromarray(image).save(a.output/file)
                    records.append(dict(id=key,episode_id=identity,source=source,task=task,demo=demo,step=step,
                        split='test',file=file,anchor=True,clip=False,sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    assert not excluded & {e['episode_id'] for e in episodes}
    previous.update(episodes=episodes,records=records,protocol='new_valid_episode_stage_test_v1',seed=20261102,
                    exclusion='All 26 previous detector/selector corpus episodes; no claim of globally unused data')
    (a.output/'manifest.json').write_text(json.dumps(previous,indent=2))
    print(json.dumps(dict(episodes=episodes,frames=len(records))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
