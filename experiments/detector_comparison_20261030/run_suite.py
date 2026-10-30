"""Sequential GPU5 experiment stages, with explicit logs and concrete outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    here=Path(__file__).resolve().parent
    args.output.mkdir(parents=True,exist_ok=True)
    results=args.output/'results';results.mkdir(exist_ok=True)
    corpus=args.output/'corpus_v3'
    def run(script,params,name):
        command=[sys.executable,str(here/script),*map(str,params)]
        start=time.perf_counter()
        with (args.output/(name+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(completed=name,seconds=time.perf_counter()-start)),flush=True)
    def bench(mode,weights,name,size=320):
        params=['--corpus',corpus,'--mode',mode,'--output',results/(name+'.json'),'--imgsz',size]
        if weights is not None:params+=['--weights',weights]
        run('benchmark.py',params,name)
    if args.stage=='detectors':
        for mode in ['gdino_six','gdino_strict','gdino_joint']:bench(mode,None,mode)
        for size in [320,640]:
            bench('world',args.output/'yolov8s-worldv2.pt','world_'+str(size),size)
            bench('yolo_ft',args.output/'training/yolo_n/weights/best.pt','yolo_n_'+str(size),size)
        bench('bowl_combined',Path('/home/jrw/vpwem_outputs/bbox_strategies_20261020/comparison_v2/detector_bundle.json'),'bowl_combined')
    elif args.stage=='mask':
        run('train_mask.py',['--corpus',corpus,'--weights',args.output/'maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
                            '--output',args.output/'training/mask_rcnn','--epochs',25],'train_mask')
        bench('mask_ft',args.output/'training/mask_rcnn/best.pt','mask_ft')
    elif args.stage=='tracking':
        candidates=[]
        for path in results.glob('*.json'):
            result=json.loads(path.read_text())
            if result.get('mode') in ['gdino_six','gdino_strict','gdino_joint','world','yolo_ft','mask_ft']:
                candidates.append((result['selection']['dev']['macro_f1'],path))
        best=max(candidates,key=lambda x:x[0])[1]
        (args.output/'tracking_baseline.json').write_text(json.dumps(dict(path=str(best),selected_on='dev_macro_f1')))
        for oracle,name in [(False,'sam_detector'),(True,'sam_oracle')]:
            params=['--corpus',corpus,'--weights',args.output/'sam2.1_t.pt','--baseline',best,
                    '--output',results/(name+'.json')]
            if oracle:params.append('--oracle')
            run('track_sam.py',params,name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['detectors','mask','tracking'],required=True);main(p.parse_args())
