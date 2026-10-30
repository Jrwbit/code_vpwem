"""Resumable bbox sidecars for full MoMaRT Zarr trajectories.

The source dataset is read-only. Completion advances only after all arrays in a
batch are written. A sidecar is usable for training only when complete=true.
"""
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import zarr
from .detector import BowlDetector


def annotate(source, destination, detector, batch_size):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(str(destination)+".lock", "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        src = zarr.open(str(source), mode="r")
        rgb = src["data/rgb"]
        ends = np.asarray(src["meta/episode_ends"])
        signature = dict(source=str(source.resolve()), shape=list(rgb.shape),
                         episode_ends_sha256=hashlib.sha256(ends.tobytes()).hexdigest(),
                         detector=detector.metadata)
        dst = zarr.open(str(destination), mode="a")
        if "signature" in dst.attrs:
            if dst.attrs["signature"] != signature:
                raise ValueError("Existing bbox cache has different source/detector")
        else:
            if list(dst.keys()):
                raise ValueError("Refusing unrecognized nonempty sidecar")
            count = len(rgb)
            for name, shape, dtype in [("boxes",(count,4),"f4"),
                ("scores",(count,),"f4"),("valid",(count,),"bool"),
                ("ambiguous",(count,),"bool"),("candidates",(count,),"i4")]:
                dst.create_dataset(name,shape=shape,dtype=dtype,
                                   chunks=(min(1024,count),)+shape[1:])
            dst.attrs.update(signature=signature,completed_frames=0,complete=False)
        start = int(dst.attrs["completed_frames"])
        if start < 0 or start > len(rgb):
            raise ValueError("Invalid resume cursor")
        begun=time.perf_counter()
        for i in range(start,len(rgb),batch_size):
            images=np.asarray(rgb[i:i+batch_size])
            results=detector(list(images))
            if len(results)!=len(images):
                raise RuntimeError("Detector batch length mismatch")
            end=i+len(results)
            dst["boxes"][i:end]=np.stack([d.box for d in results])
            for key,attr in [("scores","score"),("valid","valid"),
                             ("ambiguous","ambiguous"),("candidates","candidates")]:
                dst[key][i:end]=np.asarray([getattr(d,attr) for d in results])
            dst.attrs["completed_frames"]=end
            if (i-start)//batch_size % 25 == 0:
                print(json.dumps(dict(dataset=source.stem,completed=end,total=len(rgb),
                    elapsed_seconds=time.perf_counter()-begun)),flush=True)
        dst.attrs["complete"]=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sources",nargs="+",required=True)
    parser.add_argument("--output-root",required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--model-dir",help="Load a frozen model from a local directory without downloads")
    args=parser.parse_args()
    if args.batch_size<1:
        parser.error("batch-size must be positive")
    from cleandiffuser.dataset.imagecodecs import register_codecs
    register_codecs()
    model_kwargs=dict(model_id=str(Path(args.model_dir).resolve()),revision=None,local_files_only=True) if args.model_dir else {}
    detector=BowlDetector(device=args.device,**model_kwargs)
    for source in args.sources:
        destination=Path(args.output_root)/(Path(source).stem+".bbox.zarr")
        if Path(source).resolve()==destination.resolve():
            raise ValueError("Output must not overwrite source data")
        annotate(source,destination,detector,args.batch_size)

if __name__=="__main__":
    main()
