"""RGB-only detector pilot. Calibrate on dev, evaluate test once per model.

All model inference is FP32, batch one, physical GPU selected by CUDA visibility.
No simulator states/masks/labels are passed to detector adapters. Label access
is confined to metric computation after predictions have been recorded.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from metrics import measure,calibrate
from momart_bbox_memory.detector import MODEL_ID, REVISION, red_support_filter

PHRASES=['a red bowl','a dishwasher door','a dishwasher door handle',
         'a dishwasher rack','a drawer','a drawer handle']


def sanitize(rgb, objects):
    """Same clipping, class NMS, and class-0 red support for all adapters."""
    from torchvision.ops import nms
    result=[]
    for cls in range(6):
        candidates=[o for o in objects if o['class_id']==cls and o['score']>=.01]
        if not candidates:continue
        boxes=np.asarray([o['box'] for o in candidates],np.float32)
        boxes=np.clip(boxes,[0,0,0,0],[120,90,120,90]).astype(np.float32)
        good=np.isfinite(boxes).all(1)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
        if cls==0:good &= red_support_filter(rgb,boxes)
        indices=np.where(good)[0]
        if not len(indices):continue
        keep=nms(torch.tensor(boxes[indices]),torch.tensor([candidates[i]['score'] for i in indices],dtype=torch.float32),.5)
        for j in keep.tolist():
            idx=int(indices[j]);o=dict(candidates[idx]);o['box']=boxes[idx].tolist();result.append(o)
    return sorted(result,key=lambda x:-x['score'])[:100]


class GDINO:
    def __init__(self,mode):
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        self.mode=mode
        self.thresholds=[.01]*6
        self.processor=AutoProcessor.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True)
        self.processor.image_processor.size=dict(shortest_edge=600,longest_edge=1333)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID,revision=REVISION,
            local_files_only=True).cuda().eval().requires_grad_(False)
        self.caption='. '.join(PHRASES)+'.'
        offsets=self.processor.tokenizer(self.caption,return_offsets_mapping=True)['offset_mapping']
        self.spans=[];start=0
        for phrase in PHRASES:
            start=self.caption.index(phrase,start)
            self.spans.append([i for i,(lo,hi) in enumerate(offsets)
                               if hi>lo and lo<start+len(phrase) and hi>start])
            start+=len(phrase)

    @torch.inference_mode()
    def __call__(self,rgb):
        image=Image.fromarray(rgb);objects=[]
        if self.mode=='gdino_joint':
            inputs=self.processor(images=image,text=self.caption,return_tensors='pt').to('cuda')
            outputs=self.model(**inputs);p=outputs.logits[0].sigmoid();b=outputs.pred_boxes[0]
            boxes=torch.cat([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],-1)*b.new_tensor([120,90,120,90])
            for cls,span in enumerate(self.spans):
                scores=p[:,span].mean(-1)
                candidate_boxes=boxes.detach().cpu().numpy();candidate_scores=scores.detach().cpu().numpy()
                for i in np.where(candidate_scores>=self.thresholds[cls])[0]:
                    objects.append(dict(class_id=cls,box=candidate_boxes[i].tolist(),score=float(candidate_scores[i])))
        else:
            inputs=self.processor(images=[image]*6,text=[p+'.' for p in PHRASES],padding=True,
                                  return_tensors='pt').to('cuda')
            outputs=self.model(**inputs);post=self.processor.post_process_grounded_object_detection
            key='box_threshold' if 'box_threshold' in inspect.signature(post).parameters else 'threshold'
            results=post(outputs,inputs.input_ids,**{key:min(self.thresholds)},text_threshold=.25,target_sizes=[(90,120)]*6)
            for cls,result in enumerate(results):
                labels=result.get('text_labels',result.get('labels'))
                for box,score,label in zip(result['boxes'].cpu().numpy(),result['scores'].cpu().numpy(),labels):
                    if score<self.thresholds[cls]:continue
                    if self.mode=='gdino_strict':
                        import re
                        tokens=lambda x:[w for w in re.findall('[a-z]+',x.lower()) if w not in ('a','an','the')]
                        if tokens(label)!=tokens(PHRASES[cls]):continue
                    objects.append(dict(class_id=cls,box=box.tolist(),score=float(score)))
        return sanitize(rgb,objects)


class YOLOAdapter:
    def __init__(self,weights,world=False,imgsz=320):
        from ultralytics import YOLO,YOLOWorld
        self.model=(YOLOWorld if world else YOLO)(str(weights));self.imgsz=imgsz
        self.thresholds=[.01]*6
        if world:self.model.set_classes([p.removeprefix('a ') for p in PHRASES])

    @torch.inference_mode()
    def __call__(self,rgb):
        # Ultralytics ndarray API expects BGR; all other adapters receive RGB.
        result=self.model.predict(source=rgb[:,:,::-1].copy(),imgsz=self.imgsz,conf=min(self.thresholds),iou=.5,
                                  max_det=100,device=0,half=False,verbose=False)[0]
        objects=[dict(class_id=int(cls),box=box.tolist(),score=float(score))
                 for box,score,cls in zip(result.boxes.xyxy.cpu(),result.boxes.conf.cpu(),result.boxes.cls.cpu())
                 if float(score)>=self.thresholds[int(cls)]]
        return sanitize(rgb,objects)


class MaskAdapter:
    def __init__(self,weights):
        from train_mask import make_model
        self.model=make_model().cuda().eval()
        self.thresholds=[.01]*6
        self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=False)['model'])

    @torch.inference_mode()
    def __call__(self,rgb):
        image=torch.from_numpy(rgb.copy()).permute(2,0,1).cuda().float()/255
        self.model.roi_heads.score_thresh=min(self.thresholds)
        result=self.model([image])[0]
        objects=[dict(class_id=int(cls)-1,box=box.tolist(),score=float(score))
                 for box,score,cls in zip(result['boxes'].cpu(),result['scores'].cpu(),result['labels'].cpu())
                 if float(score)>=self.thresholds[int(cls)-1]]
        return sanitize(rgb,objects)


class BowlAdapter:
    def __init__(self,bundle):
        from momart_bbox_memory.precision_detector import PrecisionDetector
        self.model=PrecisionDetector(bundle,device='cuda:0')

    def reset(self):self.model.reset()

    def __call__(self,rgb):
        result=self.model.detect(rgb)
        return ([dict(class_id=0,box=(result.box*np.array([120,90,120,90])).tolist(),score=result.score)]
                if result.valid else [])


def benchmark(args):
    torch.set_num_threads(4)
    manifest=json.loads((args.corpus/'manifest.json').read_text())
    annotations=json.loads((args.corpus/'annotations_all.json').read_text())
    # An explicit review record is mandatory; the exporter initially writes pending.
    review=json.loads((args.corpus/'review.json').read_text())
    assert review['accepted_for']=='pilot_comparison' and review['independent_human_gt'] is False
    assert review['annotation_sha256']==hashlib.sha256((args.corpus/'annotations_all.json').read_bytes()).hexdigest()
    assert review['manifest_sha256']==hashlib.sha256((args.corpus/'manifest.json').read_bytes()).hexdigest()
    labels={r['id']:r for r in annotations['annotations']}
    rows=[r for r in manifest['records'] if r['split'] in ('dev','test')]
    images={r['id']:np.asarray(Image.open(args.corpus/r['file']).convert('RGB')) for r in rows}
    if args.mode.startswith('gdino'):detector=GDINO(args.mode)
    elif args.mode=='mask_ft':detector=MaskAdapter(args.weights)
    elif args.mode=='bowl_combined':detector=BowlAdapter(args.weights)
    else:detector=YOLOAdapter(args.weights,args.mode=='world',args.imgsz)
    for r in rows[:2]:detector(images[r['id']])
    predictions=[];previous=None
    for r in rows:
        if hasattr(detector,'reset') and (previous is None or previous['episode_id']!=r['episode_id'] or
                                         previous['step']+1!=r['step']):detector.reset()
        torch.cuda.synchronize();start=time.perf_counter()
        objects=detector(images[r['id']]);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        predictions.append(dict(id=r['id'],split=r['split'],objects=objects,seconds=elapsed))
        previous=r
        if len(predictions)%16==0:
            print(json.dumps(dict(processed=len(predictions),total=len(rows),mode=args.mode)),flush=True)
    dev=[r for r in predictions if r['split']=='dev'];test=[r for r in predictions if r['split']=='test']
    selection=calibrate(dev,labels)
    # Low-score proposal collection is for calibration. Re-measure deployment
    # latency with frozen dev thresholds, without changing test model selection.
    if hasattr(detector,'thresholds'):detector.thresholds=selection['thresholds']
    latencies=[];previous=None
    torch.cuda.reset_peak_memory_stats()
    for r in [r for r in rows if r['split']=='test']:
        if hasattr(detector,'reset') and (previous is None or previous['episode_id']!=r['episode_id'] or
                                         previous['step']+1!=r['step']):detector.reset()
        torch.cuda.synchronize();start=time.perf_counter();detector(images[r['id']]);torch.cuda.synchronize()
        latencies.append((time.perf_counter()-start)*1000);previous=r
    result=dict(mode=args.mode,imgsz=args.imgsz,weights=str(args.weights),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu_name=torch.cuda.get_device_name(0),
                weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest() if args.weights else None,
                protocol=manifest['protocol'],reference=annotations['reference'],review=review,selection=selection,
                test=measure(test,labels,selection['thresholds']),test_iou75=measure(test,labels,selection['thresholds'],.75),
                test_default25=measure(test,labels,[.25]*6),
                latency_ms=dict(median=float(np.median(latencies)),p95=float(np.percentile(latencies,95)),
                                mean=float(np.mean(latencies))),
                peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                predictions=predictions)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--mode',choices=['gdino_six','gdino_strict','gdino_joint','world','yolo_ft','mask_ft','bowl_combined'],required=True)
    p.add_argument('--weights',type=Path);p.add_argument('--imgsz',type=int,default=320)
    p.add_argument('--output',type=Path,required=True);benchmark(p.parse_args())
