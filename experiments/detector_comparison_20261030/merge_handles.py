"""Correct pilot v2 instance taxonomy: one drawer pull, two support ends.

Merge existing visible masks without rerendering or inspecting predictions.
The current annotate.py already emits this corrected taxonomy directly.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def merge(source,destination):
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'images',destination/'images')
    (destination/'masks').mkdir()
    manifest=json.loads((source/'manifest.json').read_text())
    labels=json.loads((source/'annotations_all.json').read_text())
    for a in labels['annotations']:
        masks=np.load(source/'masks'/(a['id']+'.npz'))['masks']
        groups={}
        for o,mask in zip(a['objects'],masks):
            identity=o['instance_id']
            if o['class_id']==5:
                assert identity.endswith((':left_handle',':right_handle'))
                identity=identity.rsplit(':',1)[0]+':handle'
            if identity not in groups:groups[identity]=[dict(o,instance_id=identity),mask.copy()]
            else:groups[identity][1] |= mask
        objects=[];saved=[]
        for obj,mask in groups.values():
            y,x=np.where(mask);assert len(x)
            obj.update(box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],
                       pixels=int(len(x)),tiny=len(x)<4)
            objects.append(obj);saved.append(mask)
        a['objects']=objects
        np.savez_compressed(destination/'masks'/(a['id']+'.npz'),
                            masks=np.stack(saved) if saved else np.empty((0,90,120),bool))
    manifest['annotation_version']='visible_CAD_v3_single_drawer_pull'
    labels['taxonomy_correction']='One continuous pull per drawer; support ends merged before any detector training.'
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (destination/'annotations_all.json').write_text(json.dumps(labels,indent=2))
    shutil.copy2(source/'candidate_manifest.json',destination/'candidate_manifest.json')
    print('Corrected labels:',len(labels['annotations']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args();merge(a.source,a.destination)
