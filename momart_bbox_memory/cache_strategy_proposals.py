"""Cache low-score RGB proposals and causal local re-detections, without GT."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .detector import BowlDetector, select_detection
from .strategies import pixels


def restore_crop(boxes, crop, shape):
    x1,y1,x2,y2=crop
    h,w=shape[:2]
    boxes=np.asarray(boxes,np.float32).reshape(-1,4).copy()
    boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,x2-x1)+x1
    boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,y2-y1)+y1
    return boxes/np.array([w,h,w,h],np.float32)


def cache_task(folder,detector):
    manifest=json.loads((folder/"manifest.json").read_text())
    if not manifest["complete"]:raise ValueError("Incomplete corpus")
    for item in manifest["streams"]:
        if item["split"] == "fit":
            continue
        path=folder/item["file"]
        output=path.with_suffix(".proposals.json")
        if output.exists():continue
        with np.load(path) as data:images=data["rgb"]
        started=time.perf_counter()
        proposals=[]
        for start in range(0,len(images),8):
            proposals.extend(detector.proposals(list(images[start:start+8]),threshold=.10))
        base_seconds=time.perf_counter()-started
        rows=[];previous=None;age=0
        for image,prop in zip(images,proposals):
            h,w=image.shape[:2]
            boxes=prop["boxes"]/np.array([w,h,w,h],np.float32)
            scores=prop["scores"]
            strong=select_detection(boxes,scores,1,1,.25)
            local_boxes=np.empty((0,4),np.float32);local_scores=np.empty(0,np.float32)
            roi_seconds=0.
            if not strong.valid and previous is not None and age<3:
                b=previous
                pad=max(8,int(max((b[2:]-b[:2])*[w,h])))
                crop=pixels(b,image.shape,pad=pad)
                x1,y1,x2,y2=crop
                if x2-x1>=4 and y2-y1>=4:
                    tick=time.perf_counter()
                    r=detector.proposals([image[y1:y2,x1:x2]],threshold=.10)[0]
                    roi_seconds=time.perf_counter()-tick
                    local_boxes=restore_crop(r["boxes"],crop,image.shape)
                    local_scores=r["scores"]
            if strong.valid:previous=strong.box.copy();age=0
            else:age+=1
            rows.append(dict(boxes=boxes.tolist(),scores=scores.tolist(),
                roi_boxes=local_boxes.tolist(),roi_scores=local_scores.tolist(),roi_seconds=roi_seconds))
        payload=dict(detector=detector.metadata,proposal_threshold=.10,frames=rows,
                     base_inference_seconds=base_seconds)
        output.write_text(json.dumps(payload)+"\n")
    (folder/"proposals_complete.json").write_text(json.dumps(dict(complete=True,streams=sum(s["split"]!="fit" for s in manifest["streams"]),splits=["dev","test"]))+"\n")
    print(json.dumps(dict(task=folder.name,proposals_complete=True)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--wait",action="store_true")
    p.add_argument("--tasks",nargs="*")
    args=p.parse_args()
    root=Path(args.root)
    detector=BowlDetector(device=args.device,local_files_only=True)
    done=set()
    expected=set(args.tasks) if args.tasks else None
    while True:
        for path in sorted(root.glob("*/manifest.json")):
            if path.parent.name not in done and (expected is None or path.parent.name in expected):
                cache_task(path.parent,detector);done.add(path.parent.name)
        if (len(done)==len(expected) if expected is not None else len(done)==10) or not args.wait:break
        time.sleep(10)


if __name__=="__main__":main()
