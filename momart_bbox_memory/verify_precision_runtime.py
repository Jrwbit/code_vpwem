"""Run the frozen deployment detector frame by frame; verify test metrics."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .precision_detector import PrecisionDetector
from .interaction import mask_to_box
from .strategies import iou
from .evaluate_online import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--comparison",required=True)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    output=Path(args.comparison)
    result=json.loads((output/"test_results.json").read_text())
    detector=PrecisionDetector(output/"detector_bundle.json",device=args.device)
    expected=[json.loads(line) for line in (output/f"test_{result['selected_family']}.jsonl").read_text().splitlines()]
    expected={(r["task"],r["file"],r["t"]):r for r in expected}
    rows=[];times=[];differences=0
    for path in sorted(Path(args.root).glob("*/manifest.json")):
        meta=json.loads(path.read_text())
        for item in meta["streams"]:
            if item["split"]!="test":continue
            detector.reset(0)
            with np.load(path.parent/item["file"]) as data:images=data["rgb"];masks=data["mask"]
            for t,(image,mask) in enumerate(zip(images,masks)):
                tick=time.perf_counter();d=detector.detect(image,stream_id=0);times.append(time.perf_counter()-tick)
                reference=expected[(path.parent.name,item["file"],t)]
                different=d.valid!=reference["pred_valid"]
                if d.valid and reference["pred_valid"]:
                    different=different or not np.allclose(d.box,reference["box"],atol=1e-4)
                differences+=int(different)
                gt=mask_to_box(mask,minimum_pixels=1)
                rows.append(dict(task=path.parent.name,demo=path.parent.name+"/"+item["demo"],
                    segment=item["segment"],t=t,file=item["file"],visibility_pixels=int(mask.sum()),
                    pred_valid=d.valid,pred_iou=iou(d.box,gt) if d.valid and gt is not None else 0.,
                    box=d.box.tolist(),gt_box=gt))
        print(json.dumps(dict(task=path.parent.name,runtime_frames=len(rows))),flush=True)
    metrics=summarize(rows,"pred")
    metrics.update(frames_differing_from_cached_evaluation=differences,
                   mean_latency_ms=1000*float(np.mean(times)),p95_latency_ms=1000*float(np.quantile(times,.95)),
                   selected_spec=detector.spec,policy_used=False,
                   protocol="framewise_causal_inference_on_new_online_environment_streams")
    (output/"runtime_results.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (output/"runtime_frames.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    print(json.dumps(metrics),flush=True)


if __name__=="__main__":main()
