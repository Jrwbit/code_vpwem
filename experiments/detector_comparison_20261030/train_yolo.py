"""Small in-domain detector, trained only on the pilot train split."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from ultralytics import YOLO


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--imgsz',type=int,default=320)
    a=p.parse_args();torch.set_num_threads(4)
    review=json.loads((a.corpus/'review.json').read_text());assert review['accepted_for']=='pilot_comparison'
    assert review['annotation_sha256']==hashlib.sha256((a.corpus/'annotations_all.json').read_bytes()).hexdigest()
    model=YOLO(str(a.weights));start=time.perf_counter()
    model.train(data=str(a.corpus/'data.yaml'),epochs=a.epochs,imgsz=a.imgsz,batch=16,device=0,
                amp=False,workers=2,optimizer='AdamW',lr0=.001,cos_lr=True,warmup_epochs=3,
                patience=25,seed=100,deterministic=True,mosaic=0.,mixup=0.,degrees=5.,
                translate=.05,scale=.15,fliplr=.5,plots=False,cache=True,
                project=str(a.output),name='yolo_n',exist_ok=False,verbose=False)
    (a.output/'yolo_n'/'training_elapsed.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        seed=100,epochs_requested=a.epochs,pretrained=str(a.weights),test_used_for_training=False),indent=2))
