"""Frozen raw-image action-policy observation features; no actions or future data.

Use the saved training preprocessing/normalizers and EMA condition weights.
These are observation representations, not a claim of predicted target intent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from robot_history import frame_key


def prepare_observations(obs,step):
    past=max(0,step-1);indices=[past,step];result={}
    for key in ['rgb','rgb_wrist','depth','depth_wrist','proprio','scan']:
        values=np.stack([np.asarray(obs[key][t]) for t in indices]).astype(np.float32)
        if key.startswith('rgb'):
            if values.shape!=(2,90,120,3):raise ValueError('Expected raw HWC RGB')
            values=np.moveaxis(values,-1,1)/255.
        elif key.startswith('depth'):
            if values.shape!=(2,90,120,1):raise ValueError('Expected normalized HWC depth')
            values=np.moveaxis(values,-1,1)
        elif key=='scan':
            if values.shape!=(2,1,100):raise ValueError('Expected scan[2,1,100]')
        elif values.shape!=(2,10):raise ValueError('Expected proprio[2,10]')
        result[key]=values
    return result


def attach_policy(data,path):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['protocol']=='frozen_raw_policy_observations_v1' and meta['no_actions_or_future_inputs']
    lookup={key:i for i,key in enumerate(meta['keys'])};values=np.load(path,allow_pickle=False)['context']
    assert len(lookup)==len(meta['keys'])==len(values)
    indices=[lookup[frame_key(row)] for row in data['rows']]
    return dict(data,policy_context=values[indices])


def policy_view(data,view):
    from data import feature_view
    c,x=feature_view(data,rich=view=='rich_geometry',geometry=True)
    return c,np.c_[x,data['policy_context']].astype(np.float32)


def main(a):
    from omegaconf import OmegaConf
    from momart_no_history_baseline.model import TaskConditionedObsEncoder
    from cleandiffuser.nn_condition.multi_image_condition import MultiImageObsConditionMomart
    from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer,ImageNormalizer
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    torch.set_num_threads(2)
    if a.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    cfg=OmegaConf.load(a.config);cfg=OmegaConf.to_container(cfg,resolve=True)
    condition=TaskConditionedObsEncoder(MultiImageObsConditionMomart(shape_meta=cfg['shape_meta'],emb_dim=256,
        rgb_model_name=cfg['rgb_model'],resize_shape=cfg['resize_shape'],crop_shape=cfg['crop_shape'],random_crop=cfg['random_crop'],
        use_group_norm=cfg['use_group_norm'],use_seq=True,keep_horizon_dims=True),cfg['num_tasks'],emb_dim=256)
    loaded=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state={key[len('condition.'):]:value for key,value in loaded['model_ema'].items() if key.startswith('condition.')}
    condition.load_state_dict(state,strict=True);del loaded,state
    condition=condition.to(a.device).eval().requires_grad_(False)
    with np.load(a.normalizers) as stats:
        proprio=MinMaxNormalizer(np.stack([stats['proprio_min'],stats['proprio_max']]),device=a.device)
    image=ImageNormalizer(device=a.device)
    task_ids={row['name']:row['task_id'] for row in cfg['task_datasets']}
    rows={};manifests={}
    for corpus in a.corpora:
        p=corpus/'manifest.json';manifests[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        for row in json.loads(p.read_text())['records']:rows.setdefault(frame_key(row),row)
    keys=list(rows);files={};features=[]
    try:
        for start in range(0,len(keys),a.batch_size):
            batch=keys[start:start+a.batch_size];prepared=[];tasks=[]
            for key in batch:
                row=rows[key]
                if row['source'] not in files:files[row['source']]=h5py.File(row['source'],'r')
                prepared.append(prepare_observations(files[row['source']]['data'][row['demo']]['obs'],row['step']))
                tasks.append(task_ids[row['task']])
            tensor={key:torch.as_tensor(np.stack([v[key] for v in prepared]),device=a.device) for key in prepared[0]}
            tensor={key:(proprio if key=='proprio' else image).normalize(value).contiguous() for key,value in tensor.items()}
            tensor['_task_id']=torch.tensor(tasks,dtype=torch.long,device=a.device)
            with torch.inference_mode():
                output=condition(tensor)
                if output.shape!=(len(batch),2,256):raise ValueError('Unexpected frozen condition shape')
                features.append(output.flatten(1).cpu().numpy())
            if start%256==0:print(json.dumps(dict(processed=start+len(batch),total=len(keys),device=a.device)),flush=True)
    finally:
        for f in files.values():f.close()
    a.output.parent.mkdir(parents=True,exist_ok=True);values=np.concatenate(features)
    assert np.isfinite(values).all()
    np.savez_compressed(a.output,context=values)
    a.output.with_suffix('.json').write_text(json.dumps(dict(protocol='frozen_raw_policy_observations_v1',keys=keys,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
        normalizers_sha256=hashlib.sha256(a.normalizers.read_bytes()).hexdigest(),manifest_sha256=manifests,
        checkpoint=str(a.checkpoint),config=str(a.config),normalizers=str(a.normalizers),task_ids=task_ids,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        device=a.device,cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),batch_size=a.batch_size,
        inputs='Raw RGB/wrist/depth/wrist-depth/proprio/scan at max(0,t-1),t; native5-task ID. Saved normalization and eval center crop.',
        no_actions_or_future_inputs=True,trainable=False,weight_source='model_ema.condition.*',
        caveat='Frozen policy trained on upstream MoMaRT source train episodes. Selector-unseen source-nonvalid cohorts are not a full policy holdout. Observation features, not predicted plans or a closed-loop improvement claim.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpora',type=Path,nargs='+',required=True)
    for key in ['checkpoint','config','normalizers','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--disable-tf32',action='store_true',help='Use strict float32 convolutions/matmul; record numerical backend in metadata')
    main(p.parse_args())
