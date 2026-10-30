"""Inspect RGB-only bowl localization on fixed training episodes before launch."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import zarr
from PIL import Image, ImageDraw
from .detector import BowlDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/jrw/vpwem_data/momart_zarr")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for path in sorted(Path(args.data_root).glob("*.zarr")):
        if path.stem not in {
            f"{task}_{quality}" for task in ["unload_dishwasher_to_dresser",
            "table_cleanup_to_sink", "table_cleanup_to_dishwasher",
            "table_setup_from_dishwasher", "table_setup_from_dresser"]
            for quality in ["expert", "suboptimal"]}:
            continue
        group = zarr.open(str(path), mode="r")
        ends = np.asarray(group["meta/episode_ends"])
        # Episode 0 and the midpoint episode; fixed without examining outcomes.
        for ep in sorted({0, len(ends)//2}):
            start, end = (0 if ep == 0 else int(ends[ep-1])), int(ends[ep])
            for offset in np.linspace(0, end-start-1, 6, dtype=int):
                idx = start + int(offset)
                samples.append((path.stem, ep, int(offset),
                                np.asarray(group["data/rgb"][idx])))
    detector = BowlDetector(device=args.device)
    detections = []
    begun = time.perf_counter()
    for i in range(0,len(samples),args.batch_size):
        batch = samples[i:i+args.batch_size]
        detections.extend(detector([s[3] for s in batch]))
        print(f"frames={len(detections)}/{len(samples)} elapsed={time.perf_counter()-begun:.1f}",flush=True)
    rows = []
    for index, ((dataset,ep,step,rgb),d) in enumerate(zip(samples,detections)):
        rows.append(dict(dataset=dataset,episode=ep,step=step,box=d.box.tolist(),
                         score=d.score,valid=d.valid,ambiguous=d.ambiguous,
                         candidates=d.candidates))
    (output/"detections.json").write_text(json.dumps(dict(detector=detector.metadata,
        elapsed_seconds=time.perf_counter()-begun,frames=rows),indent=2)+"\n")
    # One page per dataset: two rows of six temporally spaced original frames.
    for name in sorted({s[0] for s in samples}):
        pairs=[(s,d) for s,d in zip(samples,detections) if s[0]==name]
        sheet=Image.new("RGB",(6*360,2*310+45),"white")
        draw=ImageDraw.Draw(sheet)
        draw.text((8,8),name,fill="black")
        for n,(s,d) in enumerate(pairs):
            image=Image.fromarray(s[3]).resize((360,270),Image.Resampling.NEAREST)
            pen=ImageDraw.Draw(image)
            if d.valid:
                box=(d.box*np.array([360,270,360,270])).tolist()
                pen.rectangle(box,outline="red",width=3)
            x,y=(n%6)*360,45+(n//6)*310
            sheet.paste(image,(x,y))
            draw.text((x+4,y+274),f"ep{s[1]} t{s[2]} score={d.score:.2f} valid={int(d.valid)} amb={int(d.ambiguous)}",fill="black")
        sheet.save(output/(name+".png"))
    print(json.dumps(dict(frames=len(rows),valid=sum(r["valid"] for r in rows),
                          ambiguous=sum(r["ambiguous"] for r in rows),output=str(output))),flush=True)

if __name__=="__main__":
    main()
