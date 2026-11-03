"""Measure equal-index main-view policy input loading, without model changes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import robomimic.utils.obs_utils as ObsUtils
from momart_no_history_baseline.dataset import ZarrNoHistorySequenceDataset


class SerialDataset(torch.utils.data.Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):return self.base[i]


def digest(batch):
    h=hashlib.sha256()
    for key in sorted(batch['obs']):h.update(key.encode());h.update(batch['obs'][key].numpy().tobytes())
    h.update(batch['actions'].numpy().tobytes());return h.hexdigest()


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);ObsUtils.initialize_obs_modality_mapping_from_dict(dict(rgb=['rgb'],depth=['depth'],scan=['scan'],low_dim=['proprio']))
    modes=[(mode.split(':')[0],int(mode.split(':')[1])) for mode in a.modes];records={}
    a.output.mkdir(parents=True)
    for bs in a.batch_sizes:
        probe=ZarrNoHistorySequenceDataset(a.dataset,['rgb','depth','proprio','scan'],9,2,20)
        rng=np.random.default_rng(100);batches=[rng.integers(0,len(probe),bs).tolist() for _ in range(a.steps+a.warmup)]
        expected=None
        for mode,workers in modes:
            data=ZarrNoHistorySequenceDataset(a.dataset,['rgb','depth','proprio','scan'],9,2,20)
            if mode=='serial':data=SerialDataset(data)
            loader=torch.utils.data.DataLoader(data,batch_sampler=batches,num_workers=workers,pin_memory=False,persistent_workers=workers>0)
            started=time.perf_counter();iterator=iter(loader);times=[];hashes=[];timed_start=None;warmup_seconds=None
            for step in range(a.steps+a.warmup):
                begin=time.perf_counter();batch=next(iterator);seconds=time.perf_counter()-begin
                hashes.append(digest(batch))
                if step>=a.warmup:times.append(seconds)
                if step==a.warmup-1:warmup_seconds=time.perf_counter()-started;timed_start=time.perf_counter()
            wall=time.perf_counter()-timed_start
            if expected is None:expected=hashes
            assert hashes==expected,'Raw input bytes/action labels differ'
            name=f'bs{bs}_{mode}_workers{workers}';record=dict(batch_size=bs,mode=mode,workers=workers,steps=a.steps,
                median_batch_seconds=float(np.median(times)),mean_batch_seconds=float(np.mean(times)),samples_per_second=bs*a.steps/wall,
                timed_wall_seconds=wall,warmup_seconds=warmup_seconds,warmup_batches=a.warmup,wall_includes_hash_verification=True,all_batch_hashes_equal=True,hashes=hashes)
            records[name]=record;(a.output/'results.json').write_text(json.dumps(dict(records=records,
                scope='Real main-view RGB/depth/proprio/scan and action I/O only; no selector inference, no optimizer, no model-speed or task-success claim',
                dataset=str(a.dataset),sampling='Identical fixed random batches, seed100; declared warmup; fresh dataset/loader per variant; throughput includes consumer hashing, unlike batch wait time'),indent=2))
            print(name,json.dumps({k:v for k,v in record.items() if k!='hashes'}),flush=True)
            del iterator,loader,data
    print('COMPLETE',len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[64]);p.add_argument('--steps',type=int,default=20)
    p.add_argument('--warmup',type=int,default=4);p.add_argument('--modes',nargs='+',default=['serial:1','batched:1','batched:4'])
    main(p.parse_args())
