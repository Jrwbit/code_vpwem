"""Re-run frozen main perception on aligned train-only pixel crops.

Annotations choose training augmentation and match labels only. They never enter
perception, visual embeddings, proprio, own-robot geometry or public candidates.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
from growth_data import load_growth_data,growth_partitions
from sensor_crop import choose_crop,apply_crop,mask_box,pixel_affine,transform_geometry_context
from robot_geometry import box_geometry
from proxy import target_candidate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from momart_object_memory.perception import FrozenPerception
from momart_object_memory.batch_perception import perceive_batch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_key(row):return str(Path(row['source']).resolve()),row['demo'],int(row['step'])


def training_plan(data,research,maximum=0):
    sources={};hashes={}
    for name in ['dense','confirmation','confirm2','growth']:
        corpus=research/(name+'_corpus');manifest=json.loads((corpus/'manifest.json').read_text())
        annotation=json.loads((corpus/'annotations_all.json').read_text());lookup={r['id']:r['objects'] for r in annotation['annotations']}
        hashes[name]=dict(manifest=sha(corpus/'manifest.json'),annotation=sha(corpus/'annotations_all.json'))
        for row in manifest['records']:
            key=frame_key(row)
            if key in sources:raise ValueError('Duplicate source frame across training corpora')
            sources[key]=dict(corpus=corpus,row=row,truth=lookup[row['id']])
    train=growth_partitions(data)['train'];train=train[:maximum] if maximum else train
    rng=np.random.default_rng(20261103);plan=[]
    for i in train:
        row=data['rows'][i];source=sources[frame_key(row)];path=source['corpus']/'masks'/(source['row']['id']+'.npz')
        masks=np.load(path,allow_pickle=False)['masks'];truth=source['truth'];assert len(masks)==len(truth)
        target=row['label']['target']['instance_id'];where=[j for j,obj in enumerate(truth) if obj['instance_id']==target]
        assert len(where)<=1;mask=masks[where[0]] if where else None;crop,targeted=choose_crop(mask,rng)
        transformed=apply_crop(mask,crop,True) if mask is not None else None
        before=mask_box(mask) if mask is not None else None;after=mask_box(transformed) if transformed is not None else None
        plan.append(dict(base_index=int(i),id=row['id'],episode_id=row['episode_id'],source=row['source'],demo=row['demo'],task=row['task'],step=row['step'],
            source_corpus=str(source['corpus']),source_id=source['row']['id'],source_rgb_sha256=source['row']['sha256'],mask_sha256=sha(path),crop=crop,
            targeted_edge=targeted,reference_target=target,original_target_box=before,cropped_target_box=after,original_weight=float(data['weights'][i])))
    return plan,sources,hashes


def main(a):
    if a.output.exists() or a.output.with_suffix('.plan.json').exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=a.root/'momart_target_research_20261101';data=load_growth_data(a.root);plan,sources,source_hashes=training_plan(data,r,a.max_rows)
    reserve=r/'confirm10_corpus/manifest.json';reserved={x['episode_id'] for x in json.loads(reserve.read_text())['episodes']}
    splits=growth_partitions(data);forbidden={data['rows'][i]['episode_id'] for name,idx in splits.items() if name not in ['train','old_train'] for i in idx}
    assert not {x['episode_id'] for x in plan}&(forbidden|reserved)
    counts=Counter(total=len(plan),targeted_edge=sum(x['targeted_edge'] for x in plan),cropped_out=sum(x['cropped_target_box'] is None for x in plan))
    counts['cropped_target_short_le8']=sum(x['cropped_target_box'] is not None and min(x['cropped_target_box'][2]-x['cropped_target_box'][0],x['cropped_target_box'][3]-x['cropped_target_box'][1])<=8 for x in plan)
    counts['original_target_short_le8']=sum(x['original_target_box'] is not None and min(x['original_target_box'][2]-x['original_target_box'][0],x['original_target_box'][3]-x['original_target_box'][1])<=8 for x in plan)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    protocol=dict(protocol='aligned_main_crop_training_v1',seed=20261103,counts=dict(counts),rows=plan,source_hashes=source_hashes,
        crop='96x72 viewport inside120x90, resized120x90; target-aware2/4/8px sliver if feasible, otherwise deterministic random crop; all four edges eligible',
        interpolation=dict(rgb='PIL bilinear',depth='PIL nearest',instance_mask='PIL nearest'),past_offset=8,same_crop_current_past=True,
        inputs='Cropped main RGB/depth, unchanged current/delta proprio/task, own-robot geometry with crop-corrected UV and inverse ray calibration',
        forbidden='No wrist/side camera, future frames/commands, GT object states or reference target identity in model features',
        train_only=True,all_original_training_rows_retained=True,confirmation10_sha256=sha(reserve),plan_only=a.plan_only,partial_smoke=bool(a.max_rows),
        implementation_sha256={n:sha(Path(__file__).with_name(n)) for n in ['export_crop_training.py','sensor_crop.py','robot_geometry.py']})
    a.output.with_suffix('.plan.json').write_text(json.dumps(protocol,indent=2));print('PLAN',json.dumps(dict(counts)),flush=True)
    if a.plan_only:return
    cfg=OmegaConf.load(a.config).object_memory;perception=FrozenPerception(cfg,a.device);files={};configs={};arrays={k:[] for k in ['candidates','context','valid','y','positive','base_index','weights']};records=[]
    try:
        for start in range(0,len(plan),a.batch_size):
            batch=plan[start:start+a.batch_size];images=[];past=[];depths=[];refs=[];extra=[]
            for row in batch:
                i=row['base_index'];source=str(Path(row['source']).resolve());source_row=sources[frame_key(row)]
                if source not in files:files[source]=h5py.File(source,'r');configs[source]=json.loads(files[source]['data'].attrs['env_args'])['env_kwargs']['ig_config']
                obs=files[source]['data'][row['demo']]['obs'];t=row['step'];crop=row['crop'];image=np.asarray(obs['rgb'][t]);assert hashlib.sha256(image.tobytes()).hexdigest()==row['source_rgb_sha256']
                images.append(apply_crop(image,crop));past.append(apply_crop(np.asarray(obs['rgb'][max(0,t-8)]),crop));depths.append(apply_crop(np.asarray(obs['depth'][t],np.float32),crop,True))
                masks=np.load(Path(row['source_corpus'])/'masks'/(row['source_id']+'.npz'),allow_pickle=False)['masks'];truth=[]
                for obj,mask in zip(source_row['truth'],masks):
                    transformed=apply_crop(mask,crop,True);box=mask_box(transformed)
                    if box is not None:truth.append(dict(obj,box=box,pixels=int(transformed.sum())))
                refs.append(truth);geo=transform_geometry_context(data['geometry_context'][i],crop);extra.append((geo,configs[source]))
            detected=perceive_batch(perception,np.stack(images))
            with torch.inference_mode():
                tensor=torch.from_numpy(np.stack(images+past)).to(a.device).permute(0,3,1,2).float()/127.5-1;visual=perception.visual(tensor).mean((-1,-2)).cpu().numpy()
            for j,(row,objects,truth) in enumerate(zip(batch,detected,refs)):
                i=row['base_index'];c=np.zeros((8,155),np.float32);valid=np.zeros(8,bool);public=[];geo,config=extra[j]
                for slot,obj in enumerate(objects):
                    c[slot,:139]=np.r_[obj.feature,obj.box,obj.score,np.eye(6)[obj.class_id]];valid[slot]=True
                    c[slot,139:]=box_geometry(obj.box,depths[j],geo[:3],float(config.get('vertical_fov',90.)),float(config.get('depth_high',5.)),pixel_affine=pixel_affine(row['crop']))
                    public.append(dict(class_id=obj.class_id,score=obj.score,box=(obj.box*np.array([120,90,120,90])).tolist()))
                target=data['rows'][i]['label']['target'];slot=target_candidate(public,truth,target);positive=np.zeros(8,bool)
                for k,obj in enumerate(public):positive[k]=target_candidate([obj],truth,target)==0
                pool=(c*valid[:,None]).sum(0)/max(int(valid.sum()),1)
                context=np.r_[visual[j],visual[j+len(batch)],data['context'][i,256:],geo,pool].astype(np.float32);assert context.shape==(449,)
                for key,value in dict(candidates=c,context=context,valid=valid,y=slot if slot>=0 else 8,positive=positive,base_index=i,weights=data['weights'][i]).items():arrays[key].append(value)
                records.append(dict(id=row['id'],crop=row['crop'],public_objects=public,target_visible=any(obj['instance_id']==row['reference_target'] for obj in truth),positive_slots=np.flatnonzero(positive).tolist(),label=int(slot if slot>=0 else 8),
                    cropped_rgb_sha256=hashlib.sha256(images[j].tobytes()).hexdigest(),cropped_depth_sha256=hashlib.sha256(depths[j].tobytes()).hexdigest()))
            print(json.dumps(dict(processed=len(records),total=len(plan))),flush=True)
    finally:
        for f in files.values():f.close()
    values={k:np.asarray(v) for k,v in arrays.items()};assert np.isfinite(values['candidates']).all() and np.isfinite(values['context']).all()
    assert np.array_equal(values['base_index'],[x['base_index'] for x in plan])
    np.savez_compressed(a.output,**values)
    summary=dict(protocol='aligned_main_crop_features_v1',plan_sha256=sha(a.output.with_suffix('.plan.json')),records=records,counts=dict(counts),
        tasks=data['tasks'],perception=perception.metadata,tf32=False,device=a.device,physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        available=int(values['positive'].any(-1).sum()),target_visible=sum(x['target_visible'] for x in records),missing_target_label8=int((values['y']==8).sum()),
        feature_view='155D geometry candidates;449D context including pooled augmented candidates',smoke=bool(a.max_rows),confirmation10_loaded=False,
        caveat='Training-only synthetic viewport shifts, not physical camera motion or human intention; predictions on development remain original unmodified sensor features')
    a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',json.dumps({k:v for k,v in summary.items() if k not in ['records','perception']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','config','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-rows',type=int,default=0);p.add_argument('--plan-only',action='store_true');main(p.parse_args())
