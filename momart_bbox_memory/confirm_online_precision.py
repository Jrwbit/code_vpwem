"""Fresh live confirmation after strategy exploration, without a policy.

The detector runs synchronously before the next physics action. RGB alone is
passed to it. Reference segmentation is used only after inference. Prior audit
and exploratory-test episodes are excluded from confirmation selection.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from .precision_detector import PrecisionDetector
from .detector import BowlDetector
from .interaction import mask_to_box
from .strategies import iou
from .evaluate_online import summarize


def confirm(args):
    import h5py
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    from .annotate_sim_gt import _segmentation_mask,_target_class_id
    corpus=Path(args.corpus)
    old=json.loads(Path(args.previous).read_text())
    manifest=json.loads((corpus/"manifest.json").read_text())
    excluded=set(old["episodes"])|set(manifest["episodes"]["test"])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    np.random.seed(args.seed)
    decode=lambda names:[n.decode() if isinstance(n,bytes) else str(n) for n in names]
    started=time.perf_counter()
    with h5py.File(manifest["source"],"r") as f:
        available=sorted(set(decode(f["mask/valid"][:]))-excluded)
        if len(available)<2:raise ValueError("Need two previously unseen validation episodes")
        names=list(np.random.default_rng(args.seed).choice(available,2,replace=False))
        meta=json.loads(f["data"].attrs["env_args"])
        env=EnvGibsonMOMART(env_name=meta["env_name"],ig_config=meta["env_kwargs"]["ig_config"],
            postprocess_visual_obs=False,render=False,render_offscreen=True,use_image_obs=True)
        env.env.hard_reset=False
        renderer=env.env.simulator.renderer
        detector=PrecisionDetector(args.bundle,device=args.device)
        baseline=detector.global_detector
        if baseline is None:baseline=BowlDetector(device=args.device,local_files_only=True)
        class_id=_target_class_id(meta)
        rows=[];times=[];maes=[]
        with (output/"frames.jsonl").open("w") as stream:
            for demo in names:
                ep=f["data"][demo]
                for segment,start in enumerate(np.linspace(0,max(0,len(ep["actions"])-16),5,dtype=int)):
                    env.env.reset_to(np.asarray(ep["states"][start]),exclude=[])
                    detector.reset(0)
                    rgbs=[];masks=[]
                    for t in range(min(16,len(ep["actions"])-int(start))):
                        render=renderer.render_robot_cameras(modes=["rgb","seg"])
                        image=cv2.resize((np.asarray(render[0])[...,:3]*255).astype(np.uint8),(120,90),interpolation=cv2.INTER_CUBIC)
                        if t==0:
                            sensor=cv2.resize(np.asarray(env.get_observation()["rgb"]),(120,90),interpolation=cv2.INTER_CUBIC)
                            mae=float(np.abs(sensor.astype(float)-image.astype(float)).mean());maes.append(mae)
                            if mae>1:raise ValueError(f"Live sensor mismatch {mae}")
                        # No mask or state reaches the online detector.
                        tick=time.perf_counter();d=detector.detect(image,stream_id=0);times.append(time.perf_counter()-tick)
                        if detector.global_detector is not None:
                            base=detector.states[0]["raw"]
                        else:
                            base=baseline([image])[0]
                        mask=_segmentation_mask(render[1],class_id)
                        gt=mask_to_box(mask,minimum_pixels=1)
                        row=dict(demo=demo,segment=segment,t=t,action_index=int(start)+t,
                            visibility_pixels=int(mask.sum()),gt_box=gt,
                            pred_valid=d.valid,pred_box=d.box.tolist(),pred_score=d.score,
                            pred_iou=iou(d.box,gt) if d.valid and gt is not None else 0.,
                            raw_valid=base.valid,raw_box=base.box.tolist(),raw_score=base.score,
                            raw_iou=iou(base.box,gt) if base.valid and gt is not None else 0.)
                        rows.append(row);stream.write(json.dumps(row)+"\n");stream.flush()
                        rgbs.append(image);masks.append(mask)
                        _,_,done,_=env.step(np.asarray(ep["actions"][int(start)+t]))
                        if done:break
                    np.savez_compressed(output/f"{demo}_{segment}.npz",rgb=np.stack(rgbs),mask=np.stack(masks))
                    print(json.dumps(dict(demo=demo,segment=segment,frames=len(rows))),flush=True)
        metrics=summarize(rows,"pred");raw=summarize(rows,"raw")
        visible=[r for r in rows if r["visibility_pixels"]>0]
        for prefix,target in (("pred",metrics),("raw",raw)):
            target["iou75_hit_rate"]=sum(r[prefix+"_iou"]>=.75 for r in visible)/len(visible)
            target["visible_mean_iou_including_missing"]=float(np.mean([r[prefix+"_iou"] for r in visible]))
        result=dict(source=manifest["source"],episodes=names,excluded_episodes=sorted(excluded),
            protocol="fresh_framewise_live_simulator_confirmation",selection="frozen_dev_v2",
            selected_spec=detector.spec,metrics=metrics,baseline=raw,policy_used=False,
            live_sensor_max_mae=max(maes),mean_latency_ms=1000*float(np.mean(times)),
            p95_latency_ms=1000*float(np.quantile(times,.95)),elapsed_seconds=time.perf_counter()-started)
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",required=True)
    p.add_argument("--previous",required=True)
    p.add_argument("--bundle",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=20261022)
    confirm(p.parse_args())


if __name__=="__main__":main()
