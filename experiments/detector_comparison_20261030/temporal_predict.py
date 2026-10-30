"""Read-only RGB + frozen ROI inference. No GT/event input, no policy actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmark import YOLOAdapter,MaskAdapter
from momart_object_memory.perception import bounded_detections,raw_roi_features,file_sha256
from momart_object_memory.state import ObjectObservation
from momart_object_memory.prepare_train import encode
from cleandiffuser.nn_condition.multi_image_condition import get_resnet,replace_submodules


def predict(a):
    torch.set_num_threads(4)
    baseline=json.loads(a.baseline.read_text());thresholds=baseline['selection']['thresholds']
    weights=Path(baseline['weights']);assert file_sha256(weights)==baseline['weights_sha256']
    detector=MaskAdapter(weights) if baseline['mode']=='mask_ft' else YOLOAdapter(weights,False,baseline['imgsz'])
    detector.thresholds=thresholds
    resnet=get_resnet('resnet18');resnet=replace_submodules(resnet,lambda x:isinstance(x,nn.BatchNorm2d),lambda x:nn.GroupNorm(x.num_features//16,x.num_features))
    ckpt=torch.load(a.feature_checkpoint,map_location='cpu',weights_only=False)
    prefix='condition.encoder.key_model_map.rgb.'
    resnet.load_state_dict({k[len(prefix):]:v for k,v in ckpt['model_ema'].items() if k.startswith(prefix)},strict=True)
    del ckpt
    visual=nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu,resnet.maxpool,resnet.layer1,resnet.layer2).cuda().eval().requires_grad_(False)
    manifest=json.loads((a.corpus/'manifest.json').read_text())
    if a.output.exists() or a.output.with_suffix('.sqlite').exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(str(a.output.with_suffix('.sqlite')))
    db.execute('CREATE TABLE observations (frame INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
    predictions=[]
    for i,r in enumerate(manifest['records']):
        rgb=np.asarray(Image.open(a.corpus/r['file']).convert('RGB'))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['sha256']
        torch.cuda.synchronize();start=time.perf_counter()
        objects=[o for o in detector(rgb) if o['score']>=thresholds[o['class_id']]]
        torch.cuda.synchronize();detect_seconds=time.perf_counter()-start
        selected=bounded_detections([(o['class_id'],np.asarray(o['box'],np.float32)/[120,90,120,90],o['score']) for o in objects],16)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            if selected:
                tensor=torch.from_numpy(rgb.copy()).cuda().permute(2,0,1)[None].float()/127.5-1
                grid=visual(tensor);boxes=torch.as_tensor(np.stack([o[1] for o in selected]),dtype=torch.float32,device='cuda')
                features=raw_roi_features(grid,boxes).cpu().numpy()
            else:features=[]
        torch.cuda.synchronize();roi_seconds=time.perf_counter()-start
        observations=[ObjectObservation(c,np.asarray(b,np.float32),s,f) for (c,b,s),f in zip(selected,features)]
        db.execute('INSERT INTO observations VALUES (?,?)',(i,encode(observations,128)))
        predictions.append(dict(id=r['id'],episode_id=r['episode_id'],step=r['step'],objects=objects,detect_seconds=detect_seconds,roi_seconds=roi_seconds))
        if (i+1)%64==0:db.commit();print(json.dumps(dict(processed=i+1,total=len(manifest['records']))),flush=True)
    db.commit();db.close()
    a.output.write_text(json.dumps(dict(predictions=predictions,thresholds=thresholds,
        baseline_sha256=file_sha256(a.baseline),weights_sha256=file_sha256(weights),
        feature_checkpoint_sha256=file_sha256(a.feature_checkpoint),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        feature_protocol='raw90x120_minus1_plus1_layer2_stride8_ROIAlign3_aligned_sampling2_mean128',
        implementation_sha256=file_sha256(__file__),ground_truth_input=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--feature-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);predict(p.parse_args())
