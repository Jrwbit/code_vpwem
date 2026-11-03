"""Fixed-clock main-camera public contexts, independent of annotation filtering."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from cleandiffuser.nn_condition.multi_image_condition import get_resnet,replace_submodules
from growth_data import load_growth_data
from robot_geometry import RobotGeometry
from history_contract import public_history_contract


def observation_schedule(rows,lengths):
    episodes={}
    for row in rows:
        ep=row['episode_id'];identity=(str(Path(row['source']).resolve()),row['demo'],row['task'])
        if ep in episodes and episodes[ep]['identity']!=identity:raise ValueError('Episode sensor-source identity mismatch')
        episodes.setdefault(ep,dict(identity=identity,queries=set()))['queries'].add(int(row['step']))
    result=[]
    for ep,value in sorted(episodes.items()):
        source,demo,task=value['identity'];n=lengths[ep];times=set(range(0,n,8))|value['queries']
        if not all(0<=t<n for t in times):raise ValueError('Observation timestamp outside episode')
        result.extend(dict(episode_id=ep,source=source,demo=demo,task=task,step=t,is_grid=t%8==0) for t in sorted(times))
    return result


def load_history(path):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    if meta['protocol']!='fixed_clock_main_context_v1':raise ValueError('Unexpected main history contract')
    values=np.load(path,allow_pickle=False)['context']
    assert len(values)==len(meta['rows']) and values.shape[1]==294
    return dict(context=values,rows=meta['rows'],metadata=meta)


def query_context(data,history):
    lookup={(r['episode_id'],r['step']):i for i,r in enumerate(history['rows'])}
    if len(lookup)!=len(history['rows']):raise ValueError('Duplicate public timeline observation')
    ids=[lookup[(r['episode_id'],r['step'])] for r in data['rows']]
    return history['context'][ids]


def main(a):
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    research=a.root/'momart_target_research_20261101';reference=None
    if a.corpus is None:
        data=load_growth_data(a.root)
        reserve=json.loads((research/'confirm8_corpus/manifest.json').read_text());assert not {r['episode_id'] for r in data['rows']}&{r['episode_id'] for r in reserve['episodes']}
    else:
        if a.reference_history is None:raise ValueError('Confirmation history requires frozen training history reference')
        reference=json.loads(a.reference_history.with_suffix('.json').read_text());public_history_contract(reference)
        manifest=json.loads((a.corpus/'manifest.json').read_text());data=dict(rows=manifest['records'],tasks=reference['tasks'])
    cfg=OmegaConf.load(a.config).object_memory;files={};lengths={};sensors={};configs={}
    for row in data['rows']:
        source=str(Path(row['source']).resolve())
        if source not in files:files[source]=h5py.File(source,'r');configs[source]=json.loads(files[source]['data'].attrs['env_args'])['env_kwargs']['ig_config']
        lengths[row['episode_id']]=len(files[source]['data'][row['demo']]['actions'])
    rows=observation_schedule(data['rows'],lengths);layouts={str(Path(k).resolve()):v for k,v in json.loads((research/'geometry_growth.json').read_text())['layouts'].items()}
    model=get_resnet('resnet18');model=replace_submodules(model,lambda x:isinstance(x,nn.BatchNorm2d),lambda x:nn.GroupNorm(x.num_features//16,x.num_features))
    checkpoint=torch.load(cfg.feature_checkpoint,map_location='cpu',weights_only=False);prefix='condition.encoder.key_model_map.rgb.'
    model.load_state_dict({k[len(prefix):]:v for k,v in checkpoint['model_ema'].items() if k.startswith(prefix)},strict=True);del checkpoint
    visual=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2).to(a.device).eval().requires_grad_(False)
    geometry=RobotGeometry(a.urdf,float(next(iter(configs.values())).get('robot_scale',1.)));contexts=[];rgb_sha=[]
    try:
        for source in files:
            layout=layouts[source];assert layout['eef_median']<.03 and layout['eef_p95']<.1
            qcol=layout['offset']+13+geometry.movable.index(geometry.joints['torso_lift_joint'])
            for row in rows:
                if row['source']!=source or (source,row['demo']) in sensors:continue
                ep=files[source]['data'][row['demo']];sensors[source,row['demo']]=(np.asarray(ep['obs/proprio']),np.asarray(ep['states'][:,qcol]))
        for start in range(0,len(rows),a.batch_size):
            batch=rows[start:start+a.batch_size];images=[];past=[];parts=[]
            for row in batch:
                source=row['source'];t=row['step'];previous=max(0,t-8);obs=files[source]['data'][row['demo']]['obs']
                image=np.asarray(obs['rgb'][t]);images.append(image);past.append(np.asarray(obs['rgb'][previous]));rgb_sha.append(hashlib.sha256(image.tobytes()).hexdigest())
                proprio,torso=sensors[source,row['demo']];current=np.asarray(proprio[t],np.float32);before=np.asarray(proprio[previous],np.float32);config=configs[source]
                fov=float(config.get('vertical_fov',90.));depth_high=float(config.get('depth_high',5.))
                ee,(uv,forward)=geometry.project(current,float(torso[t]),fov);old_ee,(old_uv,_)=geometry.project(before,float(torso[previous]),fov)
                geo=np.r_[ee,uv,torso[t],float(forward),fov/90,depth_high,ee-old_ee,uv-old_uv].astype(np.float32)
                task=np.eye(len(data['tasks']),dtype=np.float32)[data['tasks'].index(row['task'])];parts.append(np.r_[current,current-before,task,geo])
            with torch.inference_mode():
                tensor=torch.from_numpy(np.stack(images+past)).to(a.device).permute(0,3,1,2).float()/127.5-1
                values=visual(tensor).mean((-1,-2)).cpu().numpy()
            contexts.extend(np.r_[values[i],values[i+len(batch)],parts[i]].astype(np.float32) for i in range(len(batch)))
            print(json.dumps(dict(processed=min(start+a.batch_size,len(rows)),total=len(rows))),flush=True)
    finally:
        geometry.close()
        for f in files.values():f.close()
    result=dict(context=np.asarray(contexts),rows=rows);shared=query_context(data,result)
    old=np.c_[data['context'],data['geometry_context']] if a.corpus is None else None
    nonvisual=float(np.max(np.abs(shared[:,256:]-old[:,256:]))) if old is not None else None
    if nonvisual is not None and nonvisual>1e-5:raise ValueError('Public proprio/task/robot geometry failed shared-frame parity: '+str(nonvisual))
    lookup={(r['episode_id'],r['step']):i for i,r in enumerate(rows)}
    assert all(rgb_sha[lookup[r['episode_id'],r['step']]]==r['sha256'] for r in data['rows'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    meta=dict(protocol='fixed_clock_main_context_v1',rows=rows,rgb_sha256=rgb_sha,grid_stride=8,query_only_rows=sum(not r['is_grid'] for r in rows),
        episode_lengths=lengths,tasks=data['tasks'],context_dim=294,feature_checkpoint_sha256=hashlib.sha256(Path(cfg.feature_checkpoint).read_bytes()).hexdigest(),
        urdf_sha256=hashlib.sha256(a.urdf.read_bytes()).hexdigest(),tf32=False,physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        shared_query_frames=len(shared),shared_nonvisual_max_abs=nonvisual,shared_old_visual_max_abs=float(np.max(np.abs(shared[:,:256]-old[:,:256]))) if old is not None else None,
        contract='Uniform public grid from episode start, all alignment-rejected/unlabelled observations included. Off-grid queries do not advance persistent state. Only main RGB/proprio/measured torso/task, no detector pool/GT objects/future commands; inference must maintain this fixed clock.')
    if a.corpus is not None:
        meta['corpus_manifest_sha256']=hashlib.sha256((a.corpus/'manifest.json').read_bytes()).hexdigest()
        if public_history_contract(meta)!=public_history_contract(reference):raise ValueError('Confirmation exporter differs from frozen training sensor contract')
    np.savez_compressed(a.output,context=result['context'])
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k not in ['rows','rgb_sha256','episode_lengths']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','config','urdf','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--corpus',type=Path);p.add_argument('--reference-history',type=Path);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=32);main(p.parse_args())
