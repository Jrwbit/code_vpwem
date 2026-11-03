"""Matched BS256/250k training; one GPU per arm, no auxiliary/offline search."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from cleandiffuser.dataset.multitask import BalancedTaskBatchMux
from momart_boxed_no_history.train import initialize_modalities,build_suite as base_suite
from momart_target_no_history.dataset import STANDARD
from .dataset import HistoryTargetDataset,VARIANTS
from .model import build_agent


def atomic_json(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def worker_init(_):
    import numcodecs.blosc
    numcodecs.blosc.set_nthreads(1)


class EpochBatchSampler:
    """Legacy length grouping, deterministic dataset/epoch RNG and resume seek.

    DataLoader prefetch cannot change the resume position: the latter is derived
    from the committed optimizer step and the balanced task schedule.
    """
    def __init__(self,dataset,batch_size,seed,consumed=0):
        self.batch_size=int(batch_size);self.seed=int(seed);self.batches=len(dataset)//self.batch_size
        if not self.batches:raise ValueError('Dataset cannot produce a BS256 batch')
        self.epoch,self.skip=divmod(consumed,self.batches)
        ends=np.asarray(dataset.replay_buffer.episode_ends);starts=np.r_[0,ends[:-1]]
        indices=np.arange(len(dataset));episode=np.searchsorted(ends,indices,side='right')
        lengths=np.maximum(1,(indices-starts[episode]+19)//20);groups=(lengths//5)*5
        order=np.argsort(lengths)
        self.groups=[order[groups[order]==g] for g in np.unique(groups)]

    def __iter__(self):
        rng=np.random.RandomState(self.seed+self.epoch);self.epoch+=1
        groups=[v.copy() for v in self.groups]
        for group in groups:rng.shuffle(group)
        order=np.arange(len(groups));rng.shuffle(order);indices=np.concatenate([groups[i] for i in order])
        skip=self.skip;self.skip=0
        for i in range(skip,self.batches):yield indices[i*self.batch_size:(i+1)*self.batch_size].tolist()

    def __len__(self):return self.batches


def consumed_batches(names,seed,step):
    cycle,offset=divmod(step,len(names));counts={name:cycle for name in names}
    order=list(names);random.Random(seed+cycle).shuffle(order)
    for name in order[:offset]:counts[name]+=1
    return counts


def make_suite(args,metadata,step):
    def dataset_type(**kwargs):return HistoryTargetDataset(**kwargs,variant=args.history_variant,target_metadata=metadata)
    # The historical trainer computes normalizers over precisely the same ten
    # logical training views. Its temporary loaders are never iterated.
    datasets,task_ids,normalizer,_,stats=base_suite(args,dataset_type=dataset_type)
    counts=consumed_batches(list(datasets),int(args.seed),step);loaders={}
    for i,(name,data) in enumerate(datasets.items()):
        loaders[name]=torch.utils.data.DataLoader(data,batch_sampler=EpochBatchSampler(data,args.physical_batch_size,args.seed+1009*i,counts[name]),
            num_workers=int(args.num_workers),pin_memory=True,persistent_workers=args.num_workers>0,worker_init_fn=worker_init,
            generator=torch.Generator().manual_seed(args.seed+30001+i))
    return datasets,task_ids,normalizer,loaders,stats


def split_batch(args,dataset,task_id,batch):
    obs={}
    for key,value in batch['obs'].items():
        tensor=value.to(args.device,non_blocking=True)
        if key=='_history_rgb':obs[key]=tensor.float()/127.5-1
        elif key.startswith('_history_'):obs[key]=tensor.bool() if key=='_history_valid' else tensor.float()
        else:obs[key]=dataset.normalizer['obs'][key].normalize(tensor.float())
    obs['_task_id']=torch.full((len(batch['actions']),),task_id,dtype=torch.long,device=args.device)
    return obs,dataset.normalizer['action'].normalize(batch['actions'].to(args.device,non_blocking=True))


def parameter_hash(module,only_scene=False):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        if only_scene and not name.startswith('condition.scene.'):continue
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main(args):
    if args.history_variant not in VARIANTS:raise ValueError(args.history_variant)
    if args.physical_batch_size!=256 or args.effective_batch_size!=256:raise ValueError('Both physical and effective BS must be256')
    if args.gradient_steps!=250000 or args.save_freq!=25000 or args.keep_checkpoints!=10:raise ValueError('Expected250k,25k saving,all10 milestones')
    if args.horizon!=9 or args.obs_steps!=2:raise ValueError('Original PTP observation/action contract required')
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    work=Path(args.work_dir);work.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(work/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metadata_path=Path(args.bbox_root)/'target_metadata.json';metadata=json.loads(metadata_path.read_text())
    if metadata['annotation_standard']!=STANDARD or metadata['bundle_manifest_sha256']!=args.selector_bundle_manifest_sha256:raise ValueError('Wrong frozen new-box cache')
    if {str(s.name) for s in args.task_datasets}!=set(metadata['supported_sources']):raise ValueError('Ten-source cache mismatch')
    existing=work/'train_config.yaml'
    if existing.exists() and OmegaConf.to_container(OmegaConf.load(existing),resolve=True)!=OmegaConf.to_container(args,resolve=True):raise ValueError('Existing run configuration differs')
    OmegaConf.save(args,existing);models=work/'models';models.mkdir(exist_ok=True)
    available=[]
    for p in models.glob('model_*.pt'):
        step=int(p.stem.split('_')[-1])
        if (models/f'scheduler_{step}.pt').exists():available.append(step)
    step=max(available,default=0)
    initialize_modalities(args)
    datasets,task_ids,normalizer,loaders,stats=make_suite(args,metadata,step)
    agent=build_agent(args,normalizer);scheduler=CosineAnnealingLR(agent.optimizer,T_max=args.gradient_steps)
    np.savez(work/'normalizers.npz',**stats)
    if step:
        agent.load(str(models/f'model_{step}.pt'))
        saved=torch.load(models/f'scheduler_{step}.pt',map_location='cpu',weights_only=False)
        scheduler.load_state_dict(saved['scheduler']);agent.scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    else:
        atomic_json(work/'initialization.json',dict(all_parameters_sha256=parameter_hash(agent.model),scene_parameters_sha256=parameter_hash(agent.model,True),seed=args.seed))
    atomic_json(work/'input_protocol.json',dict(source_commit=args.source_commit,history_variant=args.history_variant,
        shared_new_box_cache=metadata,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),physical_batch=256,effective_batch=256,
        scene_inputs='original main/wrist RGB/depth,proprio,scan; main RGB alone painted',
        crop='same main crop for t0/t1/main depth/history; random train,center eval; offsets seed independently by optimizer step',
        history='strictly before newest observation,8 slots; dense consecutive or episode-clock0,20,40,...; no repeated initial frames',
        roi='unpainted historical main RGB,clipped to the shared crop then ROIAlign75x100,shared trainable main ResNet18',
        coordinates='four corners after SAME crop; sin/cos frequencies1,2,4,8 then MLP256',
        pooling='task-conditioned learned query; shared temporal attention weights for image/position; actual frame-age sin/cos; partial visibility supplied',
        condition='[diffusion_step,history_image256,history_position256,obs_t0,obs_t1],box arm has only step/obs; invalid histories attention-masked',
        checkpoint_roi=bool(args.checkpoint_roi),short_dropout='same0.3 applied jointly to condition; no separate history dropout',
        sampler='same original length grouping; dataset/epoch-local RNG shared across all arms; resume seek from committed optimizer step',
        milestone_policy='model/EMA/optimizer + scheduler/AMP/RNG saved atomically every25k,all10 retained'))
    mux=BalancedTaskBatchMux(loaders,seed=args.seed);started=time.monotonic();initial_step=step;last=started;last_step=step;skipped=0
    batch=None;losses=[]
    atomic_json(work/'status.json',dict(stage='training',step=step,total_steps=250000,pid=os.getpid(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    while step<args.gradient_steps:
        if batch is None:name,batch=mux.next(step)
        agent.model['condition'].set_training_step(step)
        condition,action=split_batch(args,datasets[name],task_ids[name],batch)
        old_scale=agent.scaler.get_scale();result=agent.update(action,condition,None,None,update_ema=False)
        if not math.isfinite(result['loss']):raise ValueError('Nonfinite training loss')
        if agent.scaler.get_scale()<old_scale:
            skipped+=1;continue  # Retry this same physical batch; no fake optimizer step.
        agent.ema_update();scheduler.step();step+=1;losses.append(result['loss'])
        if step<=20:
            with (work/'batch_trace.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,source=name,index_sha256=hashlib.sha256(batch['sample_index'].numpy().tobytes()).hexdigest()))+'\n')
        batch=None
        if step==initial_step+1 or step%args.log_freq==0:
            now=time.monotonic();seconds=(now-last)/max(step-last_step,1)
            record=dict(step=step,total_steps=250000,loss=float(np.mean(losses)),seconds_per_step=seconds,
                elapsed_seconds=now-started,estimated_remaining_hours=(250000-step)*seconds/3600,physical_batch=256,effective_batch=256,
                lr=agent.optimizer.param_groups[0]['lr'],amp_skipped_attempts=skipped,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            with (work/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            atomic_json(work/'status.json',dict(record,stage='training',pid=os.getpid(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
            print(json.dumps(record),flush=True);last=now;last_step=step;losses=[]
        if step%25000==0:
            model=models/f'model_{step}.pt';tmp=Path(str(model)+'.tmp');agent.save(str(tmp));tmp.replace(model)
            state=models/f'scheduler_{step}.pt';tmp=Path(str(state)+'.tmp')
            torch.save(dict(scheduler=scheduler.state_dict(),scaler=agent.scaler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),tmp);tmp.replace(state)
            print('SAVED',step,flush=True)
    atomic_json(work/'status.json',dict(stage='complete',step=250000,pid=os.getpid(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    args=OmegaConf.load(p.parse_args().config)
    args.source_commit=os.environ['MOMART_SOURCE_COMMIT']
    main(args)
