"""Export complete main-view episodes with cached RGB-only bbox annotations."""
import argparse
import json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
import zarr

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",default="/home/jrw/vpwem_data/momart_zarr")
    parser.add_argument("--bbox-root",default="/home/jrw/vpwem_data/momart_bbox_gdino_red_v1")
    parser.add_argument("--output",required=True)
    parser.add_argument("--episode",type=int,default=0)
    parser.add_argument("--fps",type=int,default=20,help="Playback fps, not inferred acquisition rate")
    args=parser.parse_args()
    if args.fps <= 0 or args.episode < 0:
        parser.error("FPS must be positive and episode nonnegative")
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    tasks=["table_cleanup_to_dishwasher","table_cleanup_to_sink",
           "table_setup_from_dishwasher","table_setup_from_dresser",
           "unload_dishwasher_to_dresser"]
    manifest=dict(playback_fps=args.fps,episode_index=args.episode,videos=[])
    combined=imageio.get_writer(str(output/"all_tasks_main_view_bbox.mp4"),
        fps=args.fps,codec="libx264",quality=8,macro_block_size=1)
    try:
        for task in tasks:
            for quality in ["expert","suboptimal"]:
                name=f"{task}_{quality}"
                source=zarr.open(str(Path(args.data_root)/(name+".zarr")),mode="r")
                bbox=zarr.open(str(Path(args.bbox_root)/(name+".bbox.zarr")),mode="r")
                if not bbox.attrs.get("complete",False):
                    raise ValueError(f"Incomplete bbox cache: {name}")
                rgb=source["data/rgb"];ends=source["meta/episode_ends"][:]
                if args.episode>=len(ends):
                    raise ValueError(f"Episode unavailable: {name}")
                start=0 if args.episode==0 else int(ends[args.episode-1])
                end=int(ends[args.episode])
                demo=json.loads(source.attrs["demo_names"])[args.episode]
                path=output/f"{name}_{demo}_bbox.mp4"
                writer=imageio.get_writer(str(path),fps=args.fps,codec="libx264",
                    quality=8,macro_block_size=1)
                valid_count=0
                try:
                    for idx in range(start,end):
                        image=cv2.resize(np.asarray(rgb[idx]),(480,360),interpolation=cv2.INTER_NEAREST)
                        canvas=np.zeros((420,480,3),np.uint8)
                        valid=bool(bbox["valid"][idx])
                        score=float(bbox["scores"][idx])
                        ambiguous=bool(bbox["ambiguous"][idx])
                        if valid:
                            box=np.asarray(bbox["boxes"][idx])*[480,360,480,360]
                            x1,y1,x2,y2=np.rint(box).astype(int)
                            cv2.rectangle(image,(x1,y1),(x2,y2),(255,50,50),2)
                            valid_count+=1
                        canvas[60:]=image
                        cv2.putText(canvas,task,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                        cv2.putText(canvas,f"{quality} | {demo} | frame {idx-start}/{end-start-1}",
                            (8,36),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                        label=f"bowl: {score:.2f}"+(" | MULTIPLE CANDIDATES" if ambiguous else "") if valid else "bowl: NOT DETECTED"
                        cv2.putText(canvas,label,(8,53),cv2.FONT_HERSHEY_SIMPLEX,.43,
                            (255,100,100) if valid else (255,220,100),1)
                        writer.append_data(canvas)
                        combined.append_data(canvas)
                finally:
                    writer.close()
                manifest["videos"].append(dict(file=path.name,dataset=name,demo=demo,
                    frames=end-start,valid_frames=valid_count,source_start=start,source_end=end))
                print(json.dumps(manifest["videos"][-1]),flush=True)
    finally:
        combined.close()
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__=="__main__":
    main()
