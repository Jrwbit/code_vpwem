"""CPU-only wrist/depth enrichment from causal raw observations, with no target inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from cleandiffuser.nn_condition.multi_image_condition import get_resnet,replace_submodules
from momart_object_memory.perception import file_sha256


def depth_summary(image):
    x=np.asarray(image).reshape(-1);good=x[np.isfinite(x)&(x>0)]
    if not len(good):return np.zeros(4,np.float32)
    return np.r_[np.quantile(good,[.1,.5,.9]),len(good)/len(x)].astype(np.float32)


def main(a):
    torch.set_num_threads(2)
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.corpus/'manifest.json').read_text());rows=manifest['records']
    base_meta=json.loads(a.features.with_suffix('.json').read_text());base=np.load(a.features,allow_pickle=False)
    assert base_meta['ids']==[r['id'] for r in rows]
    assert base_meta['manifest_sha256']==file_sha256(a.corpus/'manifest.json')
    cfg=OmegaConf.load(a.config).object_memory
    assert file_sha256(cfg.feature_checkpoint)==base_meta['perception']['feature_checkpoint_sha256']
    resnet=get_resnet('resnet18');resnet=replace_submodules(resnet,lambda x:isinstance(x,nn.BatchNorm2d),lambda x:nn.GroupNorm(x.num_features//16,x.num_features))
    ckpt=torch.load(cfg.feature_checkpoint,map_location='cpu',weights_only=False);prefix='condition.encoder.key_model_map.rgb.'
    resnet.load_state_dict({key[len(prefix):]:value for key,value in ckpt['model_ema'].items() if key.startswith(prefix)},strict=True)
    del ckpt
    visual=nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu,resnet.maxpool,resnet.layer1,resnet.layer2).eval().requires_grad_(False)
    contexts=[];candidates=[];files={}
    try:
        for start in range(0,len(rows),16):
            batch=rows[start:start+16];images=[];states=[]
            for row in batch:
                if row['source'] not in files:files[row['source']]=h5py.File(row['source'],'r')
                obs=files[row['source']]['data'][row['demo']]['obs'];t=row['step'];previous=max(0,t-8)
                rgb=np.asarray(obs['rgb'][t]);assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['sha256']
                images.extend([np.asarray(obs['rgb_wrist'][t]),np.asarray(obs['rgb_wrist'][previous])])
                states.append((np.asarray(obs['depth'][t]),np.asarray(obs['depth_wrist'][t])))
            with torch.inference_mode():
                tensor=torch.from_numpy(np.stack(images)).permute(0,3,1,2).float()/127.5-1
                wrist=visual(tensor).mean((-1,-2)).numpy()
            for j,(main_depth,wrist_depth) in enumerate(states):
                i=start+j;current=wrist[2*j];past=wrist[2*j+1]
                contexts.append(np.r_[current,past,depth_summary(main_depth),depth_summary(wrist_depth)])
                extra=np.zeros((8,11),np.float32)
                for slot in np.flatnonzero(base['valid'][i]):
                    raw=base['candidates'][i,slot];box=raw[128:132];x1,y1,x2,y2=(box*[120,90,120,90]).astype(int)
                    x1,y1=max(0,min(119,x1)),max(0,min(89,y1));x2,y2=max(x1+1,min(120,x2)),max(y1+1,min(90,y2))
                    w,h=box[2:]-box[:2];center=(box[:2]+box[2:])/2
                    cosine=np.dot(raw[:128],current)/max(np.linalg.norm(raw[:128])*np.linalg.norm(current),1e-8)
                    extra[slot]=np.r_[depth_summary(main_depth[y1:y2,x1:x2]),center,w,h,w*h,w/max(h,.001),cosine]
                candidates.append(extra)
            print(json.dumps(dict(processed=min(start+16,len(rows)),total=len(rows),device='cpu')),flush=True)
    finally:
        for f in files.values():f.close()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,context=np.asarray(contexts,np.float32),candidates=np.asarray(candidates,np.float32))
    a.output.with_suffix('.json').write_text(json.dumps(dict(ids=base_meta['ids'],manifest_sha256=base_meta['manifest_sha256'],
        feature_checkpoint_sha256=file_sha256(cfg.feature_checkpoint),device='cpu',
        inputs='current/past wrist RGB, current normalized main/wrist depth; no geometry GT, actions, events or future frames',
        caution='depth retained in stored normalized units, not asserted to be metres'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['corpus','features','config','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
