"""Observed-depth geometry restricted to predicted masks; preserves original proposals."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from robot_geometry import box_geometry
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from momart_object_memory.mask_detector import FrozenMaskDetector
from momart_object_memory.perception import file_sha256


def overlap(a,boxes):
    inter=np.maximum(0,np.minimum(a[2:],boxes[:,2:])-np.maximum(a[:2],boxes[:,:2])).prod(-1)
    union=np.prod(a[2:]-a[:2])+np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(-1)-inter
    return inter/np.maximum(union,1e-8)


def surface_features(box,depth,probability,ee,uv,fov,high):
    image=np.asarray(depth)
    if image.ndim==3:image=image[...,0]
    masked=np.where(probability>=.5,image,0)
    geo=box_geometry(box,masked,ee,fov,high)
    h,w=image.shape;x1,y1,x2,y2=(np.asarray(box)*[w,h,w,h]).astype(int)
    x1=max(0,min(w-1,x1));y1=max(0,min(h-1,y1));x2=max(x1+1,min(w,x2));y2=max(y1+1,min(h,y2))
    crop=probability[y1:y2,x1:x2];ys,xs=np.where(probability>=.5)
    distance=float(np.sqrt(((xs+.5)/w-uv[0])**2+((ys+.5)/h-uv[1])**2).min()) if len(xs) else 2.
    u,v=int(uv[0]*w),int(uv[1]*h);at=float(probability[v,u]) if 0<=u<w and 0<=v<h else 0.
    return np.r_[geo,crop.mean(),(crop>=.3).mean(),(crop>=.5).mean(),(crop>=.7).mean(),distance,at,1.].astype(np.float32)


def main(a):
    torch.set_num_threads(2)
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.corpus/'manifest.json').read_text());meta=json.loads(a.features.with_suffix('.json').read_text())
    geometry_meta=json.loads(a.geometry.with_suffix('.json').read_text());geometry=np.load(a.geometry,allow_pickle=False)
    assert meta['ids']==geometry_meta['ids']==[r['id'] for r in manifest['records']]
    assert meta['manifest_sha256']==file_sha256(a.corpus/'manifest.json')
    base=np.load(a.features,allow_pickle=False);cfg=OmegaConf.load(a.config).object_memory
    assert file_sha256(cfg.detector_checkpoint)==meta['perception']['detector_checkpoint_sha256']
    net=FrozenMaskDetector(cfg.detector_checkpoint,cfg.detector_thresholds if 'detector_thresholds' in cfg else meta['perception']['thresholds'],'cpu').model
    features=np.zeros((len(meta['ids']),8,23),np.float32);unmatched=[];files={}
    try:
        for start in range(0,len(meta['ids']),4):
            rows=manifest['records'][start:start+4];images=[]
            for row in rows:
                rgb=np.asarray(Image.open(a.corpus/row['file']).convert('RGB'));assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['sha256']
                images.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255)
            with torch.inference_mode():outputs=net(images)
            for j,(row,result) in enumerate(zip(rows,outputs)):
                i=start+j
                if row['source'] not in files:files[row['source']]=h5py.File(row['source'],'r')
                depth=np.asarray(files[row['source']]['data'][row['demo']]['obs/depth'][row['step']])
                ctx=geometry['context'][i];ee,uv=ctx[:3],ctx[3:5];fov,high=ctx[7]*90,ctx[8]
                boxes=result['boxes'].numpy()/[120,90,120,90];classes=result['labels'].numpy()-1;masks=result['masks'][:,0].numpy()
                for slot in np.flatnonzero(base['valid'][i]):
                    candidate=base['candidates'][i,slot];box=candidate[128:132];cls=candidate[133:].argmax()
                    indices=np.flatnonzero(classes==cls)
                    if not len(indices):unmatched.append([row['id'],int(slot)]);continue
                    values=overlap(box,boxes[indices]);best=int(values.argmax())
                    if values[best]<.9:unmatched.append([row['id'],int(slot)]);continue
                    features[i,slot]=surface_features(box,depth,masks[indices[best]],ee,uv,fov,high)
            print(json.dumps(dict(processed=min(start+4,len(meta['ids'])),total=len(meta['ids']),unmatched=len(unmatched))),flush=True)
    finally:
        for f in files.values():f.close()
    np.savez_compressed(a.output,candidates=features)
    a.output.with_suffix('.json').write_text(json.dumps(dict(ids=meta['ids'],manifest_sha256=meta['manifest_sha256'],unmatched=unmatched,
        proposal_policy='Original proposals retained; same-class CPU mask matched at IoU>=.9; unmatched explicit zero validity flag',
        device='cpu',batch_size=4,checkpoint_sha256=file_sha256(cfg.detector_checkpoint),
        inputs='Current RGB predicted mask, observed depth and measured-robot FK; no scene GT, actions, targets or future'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['corpus','features','geometry','config','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
