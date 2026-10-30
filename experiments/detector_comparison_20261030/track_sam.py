"""Causal SAM2 pilot, detector-seeded or explicitly separate oracle-seeded bound.

Only the first frame supplies prompts. Later frames use forward mask memory;
no corrections, later-frame boxes, or backwards propagation are permitted.
The short eight-frame clips cannot validate long occlusion/re-identification.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics.models.sam.predict import SAM2VideoPredictor
from benchmark import YOLOAdapter,MaskAdapter,GDINO
from metrics import measure


def track(args):
    torch.set_num_threads(4)
    manifest=json.loads((args.corpus/'manifest.json').read_text())
    labels={a['id']:a for a in json.loads((args.corpus/'annotations_all.json').read_text())['annotations']}
    baseline=json.loads(args.baseline.read_text());thresholds=baseline['selection']['thresholds']
    rows=[r for r in manifest['records'] if r['clip']]
    detector=None
    if not args.oracle:
        if baseline['mode'].startswith('gdino'):detector=GDINO(baseline['mode'])
        elif baseline['mode']=='mask_ft':detector=MaskAdapter(Path(baseline['weights']))
        else:detector=YOLOAdapter(Path(baseline['weights']),baseline['mode']=='world',baseline['imgsz'])
        detector.thresholds=thresholds
    predictor=SAM2VideoPredictor(overrides=dict(model=str(args.weights),task='segment',mode='predict',
        imgsz=1024,device=0,half=False,verbose=False,conf=.01,save=False,save_txt=False,show=False,
        project=str(args.output.parent/'sam_runtime'),name='oracle' if args.oracle else 'detector',exist_ok=True))
    predictor.non_overlap_masks=False
    predictions=[];initializations=[]
    for episode in sorted(set(r['episode_id'] for r in rows)):
        clip=sorted([r for r in rows if r['episode_id']==episode],key=lambda r:r['step'])
        assert all(b['step']==a['step']+1 for a,b in zip(clip[:-1],clip[1:]))
        images=[np.asarray(Image.open(args.corpus/r['file']).convert('RGB')) for r in clip]
        video=args.output.parent/('clip_'+clip[0]['id']+'.avi');video.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'FFV1'),20.,(120,90))
        assert writer.isOpened()
        for image in images:writer.write(image[:,:,::-1].copy())
        writer.release();reader=cv2.VideoCapture(str(video));ok,probe=reader.read();reader.release()
        assert ok and np.array_equal(probe[:,:,::-1],images[0]),'Lossy video round trip'
        torch.cuda.synchronize();start=time.perf_counter()
        if args.oracle:
            prompts=[dict(o,score=1.) for o in labels[clip[0]['id']]['objects']]
        else:
            prompts=[o for o in detector(images[0]) if o['score']>=thresholds[o['class_id']]]
        torch.cuda.synchronize();seed_seconds=time.perf_counter()-start
        initializations.append(dict(episode=episode,prompts=len(prompts),detector_seconds=seed_seconds))
        if not prompts:
            # Record the failed initialization rather than silently removing a clip.
            predictions.extend(dict(id=r['id'],split=r['split'],objects=[],seconds=seed_seconds if i==0 else 0.)
                               for i,r in enumerate(clip))
            continue
        predictor.inference_state={}
        boxes=[o['box'] for o in prompts]
        stream=iter(predictor(source=str(video),bboxes=boxes,stream=True))
        for index,r in enumerate(clip):
            torch.cuda.synchronize();start=time.perf_counter();result=next(stream)
            objects=[]
            if result.masks is not None:
                for mask,object_id in zip(result.masks.data.cpu().numpy(),result.boxes.cls.cpu().tolist()):
                    obj=prompts[int(object_id)];y,x=np.where(mask)
                    if not len(x):continue
                    # SAM mask-to-box helpers use inclusive maxima. Derive our
                    # shared exclusive bounds from the actual visible mask.
                    b=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
                    objects.append(dict(class_id=obj['class_id'],box=b,score=obj['score']))
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            predictions.append(dict(id=r['id'],split=r['split'],objects=objects,
                                    seconds=elapsed+(seed_seconds if index==0 else 0.)))
        try:
            next(stream)
            raise AssertionError('Unexpected extra video frame')
        except StopIteration:pass
    test=[r for r in predictions if r['split']=='test']
    ids={r['id'] for r in test};base_test=[r for r in baseline['predictions'] if r['id'] in ids]
    # Oracle mask scores are not detector confidences; use all oracle-seeded masks.
    active=[0.]*6 if args.oracle else thresholds
    result=dict(mode='sam2_oracle_first_box' if args.oracle else 'sam2_detector_first_box',
        oracle_initialization=args.oracle,mask_weights=str(args.weights),baseline=str(args.baseline),
        thresholds=active,test=measure(test,labels,active),test_iou75=measure(test,labels,active,.75),
        paired_detector_test=measure(base_test,labels,thresholds),
        latency_ms=dict(median=float(np.median([r['seconds']*1000 for r in test])),
                        p95=float(np.percentile([r['seconds']*1000 for r in test],95)),
                        mean=float(np.mean([r['seconds']*1000 for r in test]))),
        initializations=initializations,predictions=predictions,
        limitation='8-frame forward clips, no re-detection or new-object discovery; oracle is annotation upper bound only')
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='predictions'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--oracle',action='store_true');track(p.parse_args())
