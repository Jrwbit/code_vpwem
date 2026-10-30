"""Aggregate fresh live confirmation and preserve paired error accounting."""
import argparse
import json
from pathlib import Path
import numpy as np
from .evaluate_online import summarize
from .summarize_online import TASKS


def aggregate(root):
    root=Path(root);rows=[];tasks={};missing=[];spec=None
    for task in TASKS:
        for split in ("expert","suboptimal"):
            name=task+"_"+split
            path=root/name/"result.json"
            if not path.exists():missing.append(name);continue
            result=json.loads(path.read_text())
            if spec is not None and result["selected_spec"]!=spec:
                raise ValueError("Confirmation strategies differ across tasks")
            spec=result["selected_spec"]
            if set(result["episodes"])&set(result["excluded_episodes"]):
                raise ValueError("Confirmation leaked earlier test episodes")
            task_rows=[json.loads(line) for line in (path.parent/"frames.jsonl").read_text().splitlines()]
            if len(task_rows)!=result["metrics"]["frames"]:raise ValueError("Incomplete rows")
            for row in task_rows:row["demo"]=name+"/"+row["demo"];row["task"]=name
            rows.extend(task_rows);tasks[name]=result
    metrics=summarize(rows,"pred");raw=summarize(rows,"raw")
    visible=[r for r in rows if r["visibility_pixels"]>0]
    for prefix,target in (("pred",metrics),("raw",raw)):
        target["iou75_hit_rate"]=sum(r[prefix+"_iou"]>=.75 for r in visible)/len(visible) if visible else None
        target["visible_mean_iou_including_missing"]=float(np.mean([r[prefix+"_iou"] for r in visible])) if visible else None
        pairs=[r for r in visible if r[prefix+"_valid"]]
        errors=[float(np.linalg.norm(((np.asarray(r[prefix+"_box"][:2])+r[prefix+"_box"][2:])/2
              -(np.asarray(r["gt_box"][:2])+r["gt_box"][2:])/2)*[120,90])) for r in pairs]
        target["paired_mean_center_error_pixels"]=float(np.mean(errors)) if errors else None
    recovered=sum(r["pred_iou"]>=.5 and r["raw_iou"]<.5 for r in visible)
    lost=sum(r["pred_iou"]<.5 and r["raw_iou"]>=.5 for r in visible)
    sizes={}
    for label,low,high in (("1_to_8_pixels",1,8),("9_to_49_pixels",9,49),("50plus_pixels",50,10**9)):
        selected=[r for r in rows if low<=r["visibility_pixels"]<=high]
        sizes[label]=dict(pred=summarize(selected,"pred"),raw=summarize(selected,"raw"))
    return dict(complete=not missing,missing_tasks=missing,selected_spec=spec,tasks=tasks,
                metrics=metrics,baseline=raw,recovered_frames=int(recovered),lost_frames=int(lost),
                visibility_size_bins=sizes,policy_used=False,
                protocol="fresh_framewise_live_simulator_confirmation")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root");args=p.parse_args()
    result=aggregate(args.root)
    (Path(args.root)/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("tasks","visibility_size_bins")},indent=2))


if __name__=="__main__":main()
