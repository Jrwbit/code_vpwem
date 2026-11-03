"""Detector adaptation labels: visible instances only, including unknown-intent frames."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from sensor_crop import apply_crop,mask_box
from growth_data import growth_partitions


def key(row):return str(Path(row['source']).resolve()),row['demo'],int(row['step'])


def collect_detector_queries(data,research):
    sources={};hashes={}
    for prefix in ['dense','confirmation','confirm2','growth']:
        corpus=research/(prefix+'_corpus');manifest=json.loads((corpus/'manifest.json').read_text());path=corpus/'annotations_all.json'
        annotations={x['id']:x['objects'] for x in json.loads(path.read_text())['annotations']}
        hashes[prefix]=dict(manifest_sha256=hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest(),annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for row in manifest['records']:
            k=key(row)
            if k in sources:raise ValueError('Duplicate detector source observation')
            sources[k]=dict(corpus=str(corpus),source_row=row,objects=annotations[row['id']])
    splits=growth_partitions(data);episodes={s:{data['rows'][i]['episode_id'] for i in splits[s]} for s in ['train','dev']}
    assert not episodes['train']&episodes['dev'];queries={}
    for split,ep in episodes.items():
        queries[split]=[dict(source=sources[key(row)],row=row,base_index=i,partition=split) for i,row in enumerate(data['rows']) if row['episode_id'] in ep]
        assert len({key(x['row']) for x in queries[split]})==len(queries[split])
        assert all(x['row']['split']==split and x['row']['tag']=='old' for x in queries[split])
    return queries,hashes


def detector_target(objects,masks,crop=None):
    masks=np.asarray(masks);assert masks.shape==(len(objects),90,120)
    kept=[];boxes=[];labels=[]
    for obj,mask in zip(objects,masks):
        value=apply_crop(mask.astype(bool),crop,True) if crop is not None else mask.astype(bool)
        box=mask_box(value)
        if box is None:continue
        cls=int(obj['class_id']);assert 0<=cls<6
        kept.append(value);boxes.append(box);labels.append(cls+1)
    stack=np.stack(kept).astype(np.uint8) if kept else np.empty((0,90,120),np.uint8)
    return dict(boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),labels=torch.tensor(labels,dtype=torch.int64),masks=torch.tensor(stack,dtype=torch.uint8),
        area=torch.tensor(stack.sum((1,2)),dtype=torch.float32),iscrowd=torch.zeros(len(kept),dtype=torch.int64))


class DetectorFrames(Dataset):
    def __init__(self,queries):self.queries=queries
    def __len__(self):return len(self.queries)
    def __getitem__(self,index):return self.observation(index,None)
    def observation(self,index,crop):
        item=self.queries[index];source=item['source'];corpus=Path(source['corpus']);row=source['source_row']
        rgb=np.asarray(Image.open(corpus/row['file']).convert('RGB'));assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['sha256']
        masks=np.load(corpus/'masks'/(row['id']+'.npz'),allow_pickle=False)['masks'];target=detector_target(source['objects'],masks,crop)
        if crop is not None:rgb=apply_crop(rgb,crop)
        target['image_id']=torch.tensor(item['base_index']);return torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255,target
