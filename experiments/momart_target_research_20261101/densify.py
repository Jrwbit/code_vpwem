"""Resample ONLY existing train/dev episodes; never move an episode between splits."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def main(a):
    source=json.loads((a.corpus/'manifest.json').read_text());a.output.mkdir(parents=True,exist_ok=False);(a.output/'images').mkdir()
    episodes=[ep for ep in source['episodes'] if ep['split'] in ('train','dev')];rows=[];files={}
    try:
        for ei,ep in enumerate(episodes):
            if ep['source'] not in files:files[ep['source']]=h5py.File(ep['source'],'r')
            data=files[ep['source']]['data'][ep['demo']]
            for step in range(0,len(data['actions']),a.stride):
                image=np.asarray(data['obs/rgb'][step]);key=f'dense_{ei}_{step}';file='images/'+key+'.png';Image.fromarray(image).save(a.output/file)
                rows.append(dict(ep,id=key,step=step,file=file,anchor=True,clip=False,sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    finally:
        for f in files.values():f.close()
    result=dict(source,episodes=episodes,records=rows,protocol='dense_original_train_dev_v1',stride=a.stride,
        source_manifest_sha256=hashlib.sha256((a.corpus/'manifest.json').read_bytes()).hexdigest(),
        selection_restriction='No test/regression/confirmation episodes included; preserve original train/dev membership')
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(episodes=len(episodes),frames=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=8);main(p.parse_args())
