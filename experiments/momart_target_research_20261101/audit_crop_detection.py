"""Audit frozen perception losses on train-only partial targets; no gate tuning."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from sensor_crop import apply_crop
from proxy import target_candidate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from momart_object_memory.mask_detector import FrozenMaskDetector,select_boxes
from momart_object_memory.perception import bounded_detections
from momart_bbox_memory.detector import red_support_filter


def public(items):return [dict(class_id=int(c),score=float(s),box=(np.asarray(b)*[120,90,120,90]).tolist()) for c,b,s in items]


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads(a.archive.with_suffix('.plan.json').read_text());meta=json.loads(a.archive.with_suffix('.json').read_text())
    assert plan['train_only'] and not meta['smoke'];rows=[(x,y) for x,y in zip(plan['rows'],meta['records']) if x['targeted_edge']]
    labels={};files={}
    for row,_ in rows:
        path=Path(row['source_corpus'])/'annotations_all.json'
        if str(path) not in labels:labels[str(path)]={x['id']:x['objects'] for x in json.loads(path.read_text())['annotations']}
    cfg=OmegaConf.load(a.config).object_memory;detector=FrozenMaskDetector(cfg.detector_checkpoint,list(cfg.thresholds),a.device);records=[];counts=defaultdict(Counter)
    try:
        for start in range(0,len(rows),a.batch_size):
            batch=rows[start:start+a.batch_size];images=[]
            for row,expected in batch:
                if row['source'] not in files:files[row['source']]=h5py.File(row['source'],'r')
                rgb=np.asarray(files[row['source']]['data'][row['demo']]['obs/rgb'][row['step']]);cropped=apply_crop(rgb,row['crop']);assert hashlib.sha256(cropped.tobytes()).hexdigest()==expected['cropped_rgb_sha256'];images.append(cropped)
            with torch.inference_mode():predictions=detector.model([torch.from_numpy(x.copy()).to(a.device).permute(2,0,1).float()/255 for x in images])
            for (row,expected),image,prediction in zip(batch,images,predictions):
                refs=labels[str(Path(row['source_corpus'])/'annotations_all.json')][row['source_id']];original=next(x for x in refs if x['instance_id']==row['reference_target'])
                target=dict(instance_id=original['instance_id'],class_id=original['class_id']);truth=[dict(target,box=row['cropped_target_box'])]
                boxes=prediction['boxes'].detach().cpu().numpy();classes=prediction['labels'].detach().cpu().numpy()-1;scores=prediction['scores'].detach().cpu().numpy()
                raw=[dict(class_id=int(c),score=float(s),box=b.tolist()) for b,c,s in zip(boxes,classes,scores)]
                threshold=[x for x in raw if x['score']>=max(.01,cfg.thresholds[x['class_id']])]
                color=[x for x in threshold if x['class_id']!=0 or red_support_filter(image,np.asarray([x['box']]))[0]]
                nms=select_boxes(image,prediction,list(cfg.thresholds));bounded=bounded_detections(nms,8)
                stages=dict(network_output=raw,class_threshold=threshold,red_support=color,class_nms=public(nms),bounded8=public(bounded))
                available={k:target_candidate(v,truth,target)>=0 for k,v in stages.items()};values=list(available.values());assert all(not b or a for a,b in zip(values,values[1:]))
                match=bool(expected['positive_slots']);agrees=available['bounded8']==match
                first=next((key for key,value in available.items() if not value),'retained');group=counts[str(target['class_id'])];group['queries']+=1;group[first]+=1;group['archive_matching_disagreements']+=not agrees
                records.append(dict(id=row['id'],base_index=row['base_index'],target_class=target['class_id'],available=available,first_missing_stage=first,archive_agrees=agrees))
            print(json.dumps(dict(processed=len(records),total=len(rows))),flush=True)
    finally:
        for f in files.values():f.close()
    payload=dict(protocol='frozen_crop_perception_stage_audit_v1',records=records,counts={k:dict(v) for k,v in counts.items()},
        archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),device=a.device,tf32=False,
        caveat='Network output already includes built-in score threshold/NMS; cannot distinguish RPN vs classifier loss. Training crops only; no thresholds changed, no confirmation10 data, not human intent truth.')
    a.output.write_text(json.dumps(payload,indent=2));print('COMPLETE',json.dumps(payload['counts']),flush=True)
    assert not sum(x['archive_matching_disagreements'] for x in counts.values())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['archive','config','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=16);main(p.parse_args())
