"""Fixed episode split for sink-task extension; never opens existing reserves."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'images').mkdir()
    episodes=[];records=[];rng=np.random.default_rng(20261103)
    for style in ['expert','suboptimal']:
        task='table_cleanup_to_sink_'+style;source=a.data_root/(task+'.hdf5')
        with h5py.File(source,'r') as f:
            def names(key):return sorted(x.decode() if isinstance(x,bytes) else str(x) for x in f['mask'][key][:])
            train=names('train');valid=names('valid');rng.shuffle(train);rng.shuffle(valid)
            choices=[('train',train[:a.train]),('dev',train[a.train:a.train+a.dev]),('test',valid[:a.test])]
            for split,demos in choices:
                for demo in demos:
                    ep=dict(source=str(source.resolve()),demo=demo,task=task,episode_id=task+'/'+demo,split=split)
                    episodes.append(ep);obs=f['data'][demo]['obs'];length=len(obs['rgb'])
                    for step in range(0,length,a.stride):
                        key='sink_'+str(len(episodes)-1)+'_'+str(step);image=np.asarray(obs['rgb'][step]);file='images/'+key+'.png'
                        Image.fromarray(image).save(a.output/file)
                        records.append(dict(ep,id=key,step=step,file=file,anchor=True,clip=False,sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    classes=['a red bowl.','a dishwasher door.','a dishwasher door handle.','a dishwasher rack.','a drawer.','a drawer handle.']
    value=dict(protocol='sink_extension_fixed_split_v1',seed=20261103,episodes=episodes,records=records,
        image_hw=[90,120],classes=classes,stride=a.stride,
        split_rule='8 train + 2 dev episodes per sink source from policy-train, 2 test from policy-valid; no existing reserve opened')
    (a.output/'manifest.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(dict(episodes=len(episodes),frames=len(records)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train',type=int,default=8);p.add_argument('--dev',type=int,default=2);p.add_argument('--test',type=int,default=2);p.add_argument('--stride',type=int,default=16);main(p.parse_args())
