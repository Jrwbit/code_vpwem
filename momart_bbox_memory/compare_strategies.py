"""Select configurations on dev only; evaluate frozen choices on new test.

Ground-truth masks are used only after causal RGB predictions have been made.
All candidate configurations are declared before final test evaluation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from .detector import Detection, select_detection
from .strategies import Config, CausalTracker, color_refine, iou, pixels
from .interaction import mask_to_box
from .evaluate_online import summarize


def candidates():
    result=[dict(family="baseline",method="baseline",threshold=.25),
            dict(family="strict",method="strict",threshold=.35),
            dict(family="color_refine",method="color_refine",threshold=.25)]
    for threshold in (.15,.20):
        result.append(dict(family="low_threshold",method="baseline",threshold=threshold))
    for low in (.10,.15,.20):
        for overlap in (.05,.15,.30):
            result.append(dict(family="association",method="track",track=asdict(Config(
                method="association",low_threshold=low,association_iou=overlap))))
    for method in ("template","flow"):
        for low in (.15,.20):
            for age in (2,3):
                for refine in (False,True):
                    result.append(dict(family=method,method="track",track=asdict(Config(
                        method=method,low_threshold=low,max_age=age,refine=refine))))
    for threshold in (.20,.25,.30):
        result.append(dict(family="roi",method="roi",threshold=threshold))
    for low in (.15,.20):
        for refine in (False,True):
            result.append(dict(family="combined",method="track",track=asdict(Config(
                method="combined",low_threshold=low,refine=refine)),use_roi=True))
    for threshold in (.3,.5,.7,.85):
        result.append(dict(family="tiny",method="tiny",mask_threshold=threshold))
        result.append(dict(family="tiny_refine",method="tiny_refine",threshold=.25,mask_threshold=threshold))
        for confidence in (.80,.90,.97):
            result.append(dict(family="hybrid",method="hybrid",threshold=.25,
                               mask_threshold=threshold,confidence=confidence))
    # Additional dev-only trials preserve reliable original boxes and calibrate
    # boundary refinement strength. Confirmation uses fresh, unseen episodes.
    for low in (.15,.20):
        for alpha in (.25,.5,.75):
            for method in ("template","combined"):
                result.append(dict(family=method+"_blend",method="track",track=asdict(Config(
                    method=method,low_threshold=low,refine=True,refine_alpha=alpha)),
                    use_roi=method=="combined"))
    for threshold in (.3,.5,.7):
        for padding in (-.5,.5,1.):
            result.append(dict(family="tiny_padded",method="tiny",mask_threshold=threshold,mask_padding=padding))
        for confidence in (.8,.9,.97):
            result.append(dict(family="tiny_fallback",method="hybrid",threshold=.25,
                               mask_threshold=threshold,confidence=confidence,refine_existing=False))
    return result


def streams(root,split,need_tiny=False):
    output=[]
    for path in sorted(Path(root).glob("*/manifest.json")):
        meta=json.loads(path.read_text())
        if not meta["complete"]:raise ValueError("Incomplete corpus")
        for item in meta["streams"]:
            if item["split"]!=split:continue
            source=path.parent/item["file"]
            with np.load(source) as data:images=data["rgb"];masks=data["mask"]
            props=json.loads(source.with_suffix(".proposals.json").read_text())["frames"]
            probs=None
            if need_tiny:
                with np.load(source.with_suffix(".tiny.npz")) as data:probs=data["prob"].astype(np.float32)
            output.append((path.parent.name,item,images,masks,props,probs))
    if len({s[0] for s in output})!=10:raise ValueError("Expected all ten tasks")
    return output


def predict_stream(spec,images,props,probs=None):
    from .tiny_mask import mask_box
    tracker=CausalTracker(Config(**spec["track"])) if spec["method"]=="track" else None
    output=[]
    for t,(image,prop) in enumerate(zip(images,props)):
        boxes=np.asarray(prop["boxes"],np.float32).reshape(-1,4)
        scores=np.asarray(prop["scores"],np.float32)
        if spec.get("use_roi"):
            boxes=np.concatenate([boxes,np.asarray(prop["roi_boxes"],np.float32).reshape(-1,4)])
            scores=np.concatenate([scores,np.asarray(prop["roi_scores"],np.float32)])
        method=spec["method"]
        source="detector"
        if tracker is not None:
            detected,source=tracker.step(image,boxes,scores)
        else:
            detected=select_detection(boxes,scores,1,1,spec.get("threshold",.25))
        if method=="strict" and detected.ambiguous:
            detected=Detection(np.zeros(4,np.float32),detected.score,False,True,detected.candidates)
        if method=="roi" and not detected.valid:
            # Crops were generated causally from a recent baseline observation.
            detected=select_detection(prop["roi_boxes"],prop["roi_scores"],1,1,spec["threshold"])
            source="roi"
        if method=="color_refine" and detected.valid:
            detected=Detection(color_refine(image,detected.box),detected.score,True,False,1)
        if method in ("tiny","tiny_refine","hybrid"):
            probability=probs[t]
            region=None
            if method in ("tiny_refine","hybrid") and detected.valid:
                region=pixels(detected.box,image.shape,pad=2)
            box,score=mask_box(probability,spec["mask_threshold"],region)
            padding=spec.get("mask_padding",0.)
            if box is not None and padding:
                adjusted=box+np.array([-padding/120,-padding/90,padding/120,padding/90],np.float32)
                adjusted=np.clip(adjusted,0,1)
                if adjusted[2]>adjusted[0] and adjusted[3]>adjusted[1]:box=adjusted
            if method=="tiny":
                detected=Detection(box if box is not None else np.zeros(4,np.float32),score,box is not None,False,int(box is not None))
                source="tiny"
            elif detected.valid:
                if spec.get("refine_existing",True) and box is not None and iou(box,detected.box)>=.25:
                    detected=Detection(box,score,True,False,1);source="tiny_refine"
            elif method=="hybrid" and box is not None and score>=spec["confidence"]:
                detected=Detection(box,score,True,False,1);source="tiny_fallback"
        output.append((detected,source))
    return output


def evaluate(spec,data,save=None):
    rows=[];elapsed=0.
    for task,item,images,masks,props,probs in data:
        start=time.perf_counter()
        predictions=predict_stream(spec,images,props,probs)
        elapsed+=time.perf_counter()-start
        for t,((d,source),mask) in enumerate(zip(predictions,masks)):
            gt=mask_to_box(mask,minimum_pixels=1)
            overlap=iou(d.box,gt) if d.valid and gt is not None else 0.
            row=dict(task=task,demo=task+"/"+item["demo"],segment=item["segment"],t=t,
                     file=item["file"],visibility_pixels=int(mask.sum()),gt_box=gt,
                     pred_valid=d.valid,pred_iou=overlap,box=d.box.tolist(),source=source)
            rows.append(row)
    result=summarize(rows,"pred")
    visible=[r for r in rows if r["visibility_pixels"]>0]
    result["iou75_hit_rate"]=sum(r["pred_iou"]>=.75 for r in visible)/len(visible)
    result["visible_mean_iou_including_missing"]=float(np.mean([r["pred_iou"] for r in visible]))
    paired=[r for r in visible if r["pred_valid"]]
    errors=[float(np.linalg.norm(((np.asarray(r["box"][:2])+r["box"][2:])/2
              -(np.asarray(r["gt_box"][:2])+r["gt_box"][2:])/2)*[120,90])) for r in paired]
    result["paired_mean_center_error_pixels"]=float(np.mean(errors)) if errors else None
    result["center_within_2px_rate"]=sum(error<=2 for error in errors)/len(visible)
    result["postprocess_ms_per_frame"]=1000*elapsed/len(rows)
    result["per_task"]={task:summarize([r for r in rows if r["task"]==task],"pred") for task in sorted({r["task"] for r in rows})}
    if save is not None:
        Path(save).write_text("".join(json.dumps(row)+"\n" for row in rows))
    return result


def rank(record):
    metrics=record["dev"]
    # Fixed precision requirement; optimize recall only among eligible methods.
    precision=metrics["localization_precision"] or 0
    return (precision>=.995, metrics["hit_rate"] if precision>=.995 else precision,
            metrics["iou75_hit_rate"], metrics["visible_mean_iou_including_missing"])


def select(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    data=streams(args.root,"dev",need_tiny=True)
    records=[]
    for spec in candidates():
        record=dict(spec=spec,dev=evaluate(spec,data))
        records.append(record)
        print(json.dumps(dict(family=spec["family"],hit=record["dev"]["hit_rate"],
                              precision=record["dev"]["localization_precision"])),flush=True)
    families=sorted({r["spec"]["family"] for r in records})
    choices={family:max([r for r in records if r["spec"]["family"]==family],key=rank) for family in families}
    winner=max(records,key=rank)
    payload=dict(precision_floor=.995,selection_split="dev",selected=winner,
                 family_choices=choices,all_trials=records,frozen=True)
    (output/"frozen_dev_selection.json").write_text(json.dumps(payload,indent=2)+"\n")
    (output/"detector_bundle.json").write_text(json.dumps(dict(
        spec=winner["spec"],weights="../tiny/tiny_mask.pt",selection_split="dev",
        policy_used=False,source="rgb_only"),indent=2)+"\n")
    print(json.dumps(dict(selected=winner["spec"],dev=winner["dev"])),flush=True)


def test(args):
    output=Path(args.output)
    frozen=json.loads((output/"frozen_dev_selection.json").read_text())
    if not frozen["frozen"]:raise ValueError("Configuration must be frozen before test")
    data=streams(args.root,"test",need_tiny=True)
    records={}
    for family,choice in frozen["family_choices"].items():
        metrics=evaluate(choice["spec"],data,save=output/f"test_{family}.jsonl")
        records[family]=dict(spec=choice["spec"],metrics=metrics)
        print(json.dumps(dict(family=family,hit=metrics["hit_rate"],precision=metrics["localization_precision"])),flush=True)
    selected=frozen["selected"]["spec"]["family"]
    result=dict(selection_split="dev",test_used_for_selection=False,selected_family=selected,
                selected=records[selected],families=records,policy_used=False)
    (output/"test_results.json").write_text(json.dumps(result,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["select","test"])
    p.add_argument("--root",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args();(select if args.mode=="select" else test)(args)


if __name__=="__main__":main()
