import argparse,gc,json,time
import numpy as np
import torch,zarr
from omegaconf import OmegaConf
from momart_no_history_baseline.cache import training_replay
from momart_object_memory.perception import FrozenPerception
from momart_object_memory.batch_perception import perceive_batch

parser=argparse.ArgumentParser(description='Benchmark frozen Mask batches, including reads and postprocessing')
parser.add_argument('--config',required=True)
parser.add_argument('--device',default='cuda:0')
parser.add_argument('--sizes',nargs='+',type=int,default=[16,64,128,256])
parser.add_argument('--frames',type=int,default=512)
parser.add_argument('--start',type=int,default=1000)
args=parser.parse_args()
if min(args.sizes)<1 or args.frames<1 or args.start<0:parser.error('Invalid batch or frame range')
torch.set_num_threads(4)
cfg=OmegaConf.load(args.config)
p=FrozenPerception(cfg.object_memory,args.device)
r,_,_=training_replay(zarr.open_group(cfg.task_datasets[0].path,mode='r'))
end=min(args.start+args.frames,len(r['rgb']))
if args.start>=end:parser.error('Frame range outside dataset')
perceive_batch(p,np.asarray(r['rgb'][args.start:min(args.start+16,end)]))
for bs in args.sizes:
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    try:
        torch.cuda.synchronize();t=time.monotonic();frames=0
        for start in range(args.start,end,bs):
            images=np.asarray(r['rgb'][start:min(start+bs,end)])
            result=perceive_batch(p,images)
            frames+=len(result)
        torch.cuda.synchronize()
        print(json.dumps(dict(bs=bs,frames=frames,fps=frames/(time.monotonic()-t),peak_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
    except torch.OutOfMemoryError:
        print(json.dumps(dict(bs=bs,error='CUDA OOM',peak_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
