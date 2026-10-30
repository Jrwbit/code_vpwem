"""Launch balanced, independent annotation workers; data stays outside Git."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import zarr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--data-root", default="/home/jrw/vpwem_data/momart_zarr")
    parser.add_argument("--output-root", default="/home/jrw/vpwem_data/momart_bbox_gdino_red_v1")
    parser.add_argument("--model-dir", help="Use transferred local weights in every worker")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU IDs must be distinct")
    tasks = ["table_cleanup_to_dishwasher", "table_cleanup_to_sink",
             "table_setup_from_dishwasher", "table_setup_from_dresser",
             "unload_dishwasher_to_dresser"]
    sources = [Path(args.data_root)/f"{task}_{quality}.zarr"
               for task in tasks for quality in ["expert", "suboptimal"]]
    lengths = {str(p): len(zarr.open(str(p), mode="r")["data/rgb"]) for p in sources}
    loads = [0]*len(args.gpus)
    shards = [[] for _ in args.gpus]
    for source, count in sorted(lengths.items(), key=lambda item: -item[1]):
        worker = min(range(len(loads)), key=loads.__getitem__)
        shards[worker].append(source)
        loads[worker] += count
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = dict(commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        total_frames=sum(loads), workers=[])
    for gpu, sources, count in zip(args.gpus, shards, loads):
        if not sources:
            continue
        log = output/f"worker_{stamp}_gpu{gpu}.log"
        command = [sys.executable, "-m", "momart_bbox_memory.annotate_bbox",
                   "--sources", *sources, "--output-root", str(output), "--batch-size", "4"]
        if args.model_dir:command.extend(["--model-dir",str(Path(args.model_dir).resolve())])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
                   OMP_NUM_THREADS="4")
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=repo, env=env,
                stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        manifest["workers"].append(dict(gpu=gpu, pid=process.pid, frames=count,
                                        sources=sources, log=str(log)))
    path = output/f"launch_{stamp}.json"
    path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(manifest=str(path), **manifest), indent=2))

if __name__ == "__main__":
    main()
