"""Actual main-view diffusion updates with fixed equal training batches.

This benchmarks the shared action-policy skeleton, not the unfinished one-box
adapter. It does not write policy checkpoints or start a formal training run.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from omegaconf import OmegaConf
import robomimic.utils.obs_utils as ObsUtils
from cleandiffuser.dataset.dataset_utils import ImageNormalizer,MinMaxNormalizer
from momart_no_history_baseline.dataset import ZarrNoHistorySequenceDataset
from momart_no_history_baseline.model import build_agent
from momart_boxed_no_history.train import split_batch
from benchmark_training_io import SerialDataset,digest


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(2)
    cfg=OmegaConf.load(a.config);cfg.device=a.device;cfg.physical_batch_size=a.batch_size
    cfg.shape_meta.obs={k:v for k,v in cfg.shape_meta.obs.items() if k in ['rgb','depth','proprio','scan']}
    assert cfg.horizon==9 and cfg.obs_steps==2
    ObsUtils.initialize_obs_modality_mapping_from_dict(dict(rgb=['rgb'],depth=['depth'],scan=['scan'],low_dim=['proprio']))
    probe=ZarrNoHistorySequenceDataset(a.dataset,list(cfg.shape_meta.obs),9,2,20)
    proprio=np.asarray(probe.replay_buffer['proprio'][:]);action=np.asarray(probe.replay_buffer['action'][:])
    normalizer={'obs':{k:ImageNormalizer(device=a.device) for k in ['rgb','depth','scan']},
        'action':MinMaxNormalizer(np.stack([action.min(0),action.max(0)]),device=a.device)}
    normalizer['obs']['proprio']=MinMaxNormalizer(np.stack([proprio.min(0),proprio.max(0)]),device=a.device)
    rng=np.random.default_rng(100);batches=[rng.integers(0,len(probe),a.batch_size).tolist() for _ in range(a.warmup+a.steps)]
    records={};reference_hashes=None;reference_losses=None
    (a.output/'protocol.json').write_text(json.dumps(dict(dataset=str(a.dataset),config=str(a.config),batch=a.batch_size,seed=100,
        modes=a.modes,warmup=a.warmup,steps=a.steps,keys=list(cfg.shape_meta.obs),physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        semantics='Same raw sample indices, model architecture, optimizer, AMP behavior and initialization for every mode; original baseline agent; no one-box adapter/selector cache yet',
        measurement='Preflight all raw batch hashes in a separate loader pass; timed pass contains no hashing. End-to-end iterator wait, host-to-device normalization, forward/backward/optimizer/EMA; per-step CUDA synchronization',
        limitation='Short shared main-view skeleton benchmark on one data source and shared GPU; not full new experiment ETA or task success'),indent=2))
    for recipe in a.modes:
        mode,workers=recipe.split(':');workers=int(workers)
        data=ZarrNoHistorySequenceDataset(a.dataset,list(cfg.shape_meta.obs),9,2,20);data.normalizer=normalizer
        loader=torch.utils.data.DataLoader(SerialDataset(data) if mode=='serial' else data,batch_sampler=batches,
            num_workers=workers,pin_memory=True,persistent_workers=workers>0)
        hashes=[digest(batch) for batch in loader]
        if reference_hashes is None:reference_hashes=hashes
        assert hashes==reference_hashes
        iterator=iter(loader);torch.manual_seed(100);torch.cuda.manual_seed_all(100)
        agent=build_agent(cfg,normalizer);losses=[];times=[];waits=[];compute=[];torch.cuda.reset_peak_memory_stats()
        for step in range(a.warmup+a.steps):
            torch.cuda.synchronize();start=time.perf_counter();batch=next(iterator);loaded=time.perf_counter()
            condition,target=split_batch(cfg,data,a.task_id,batch);result=agent.update(target,condition,None,None)
            torch.cuda.synchronize();end=time.perf_counter();losses.append(float(result['loss']))
            if step>=a.warmup:times.append(end-start);waits.append(loaded-start);compute.append(end-loaded)
            if (step+1)%5==0:print('STEP',recipe,step+1,round(end-start,4),flush=True)
        if reference_losses is None:reference_losses=losses
        records[recipe]=dict(mean_step_seconds=float(np.mean(times)),median_step_seconds=float(np.median(times)),
            mean_loader_wait_seconds=float(np.mean(waits)),mean_update_seconds=float(np.mean(compute)),samples_per_second=a.batch_size/float(np.mean(times)),
            all_input_hashes_equal=True,max_loss_difference=float(np.max(np.abs(np.array(losses)-reference_losses))),losses=losses,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),steps=a.steps,warmup=a.warmup)
        (a.output/'results.json').write_text(json.dumps(dict(records=records,no_training_checkpoint=True,one_box_adapter_included=False),indent=2))
        print('RESULT',recipe,json.dumps({k:v for k,v in records[recipe].items() if k!='losses'}),flush=True)
        del agent,iterator,loader,data,batch,condition,target;gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['dataset','config','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=64);p.add_argument('--task-id',type=int,default=3)
    p.add_argument('--steps',type=int,default=16);p.add_argument('--warmup',type=int,default=4)
    p.add_argument('--modes',nargs='+',default=['serial:1','batched:1','batched:4']);main(p.parse_args())
