"""Read-only acceptance on real training prefixes; no policy training or actions.

Creates isolated, explicitly labelled excerpts, never overwrites training data.
Exercises prepare_dataset, ObjectDataset, train.split_batch, OnlineObjects and
production evaluation normalization with independent online detector inference.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import sqlite3
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from momart_no_history_baseline.cache import training_replay
from momart_boxed_no_history.train import initialize_modalities
from momart_object_memory.prepare_train import prepare_dataset
from momart_object_memory.perception import FrozenPerception, bounded_detections, raw_roi_features
from momart_object_memory.cache import select_mode, stream_contract
from momart_object_memory.dataset import ObjectDataset
from momart_object_memory.train import build_suite, split_batch
from momart_object_memory.evaluate import normalize_observations, validate_experiment_protocol
from momart_object_memory.online import OnlineObjects
from momart_object_memory.state import make_bank, ObservationStream, query_schedule
from momart_object_memory.main_visual import build_main_boxer
from momart_object_memory.model import OBJECT_KEYS, ObjectMemoryEncoder


def must_reject(call, label):
    try:
        call()
    except ValueError:
        return label
    raise AssertionError('Accepted incompatible protocol: '+label)


def verify(args):
    torch.set_num_threads(2);torch.manual_seed(100)
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    cfg=OmegaConf.load(args.config);original=copy.deepcopy(cfg)
    cfg.device=args.device;cfg.data_root=str(output/'source');cfg.bbox_root=str(output/'main_boxes')
    cfg.object_memory.mode='a3'
    cfg.object_memory.cache_root=str(output/'objects');cfg.work_dir=str(output/'work')
    cfg.num_workers=0;cfg.physical_batch_size=2
    names=['table_setup_from_dresser_expert','unload_dishwasher_to_dresser_suboptimal']
    cfg.task_datasets=[s for s in cfg.task_datasets if s.name in names]
    provenance=[]
    for directory in (cfg.data_root,cfg.bbox_root,cfg.object_memory.cache_root,cfg.work_dir):
        Path(directory).mkdir(parents=True)
    for spec in cfg.task_datasets:
        name=spec.name;source=Path(original.data_root)/(name+'.zarr')
        replay,demos,frame_map=training_replay(zarr.open_group(str(source),mode='r'))
        n=min(args.frames,int(replay.episode_ends[0]))
        if n<17:raise ValueError('Need at least 17 real frames per training prefix')
        subset=Path(cfg.data_root)/(name+'.zarr');root=zarr.open_group(str(subset),mode='w')
        root.attrs.update(vpwem_cache_complete=True,vpwem_cache_kind='momart',demo_names=[demos[0]],
            acceptance_scope='real_training_prefix_not_complete_episode',original_source=str(source),
            original_physical_frames=frame_map[:n].tolist())
        root.create_dataset('meta/episode_ends',data=np.array([n],np.int64))
        digests={}
        for key in replay.keys():
            value=np.asarray(replay[key][:n]);root.create_dataset('data/'+key,data=value)
            digests[key]=hashlib.sha256(value.tobytes()).hexdigest()
        old_boxes=zarr.open_group(str(Path(original.bbox_root)/(name+'.bbox.zarr')),mode='r')
        boxes=zarr.open_group(str(Path(cfg.bbox_root)/(name+'.bbox.zarr')),mode='w')
        signature=copy.deepcopy(old_boxes.attrs['signature'])
        signature.update(source=str(subset),shape=list(root['data/rgb'].shape),
            episode_ends_sha256=hashlib.sha256(np.array([n],np.int64).tobytes()).hexdigest())
        boxes.attrs.update(dict(old_boxes.attrs));boxes.attrs.update(signature=signature,complete=True,
            completed_frames=n,demo_names=[demos[0]])
        boxes.create_dataset('boxes',data=np.asarray(old_boxes['boxes'][:n]))
        boxes.create_dataset('valid',data=np.asarray(old_boxes['valid'][:n]))
        provenance.append(dict(dataset=name,demo=demos[0],frames=n,source=str(source),
            physical_frames=frame_map[:n].tolist(),array_sha256=digests))
    OmegaConf.save(cfg,output/'config.yaml')
    perceive=FrozenPerception(cfg.object_memory,args.device)
    assert not any(p.requires_grad for p in perceive.detector.model.parameters())
    assert not any(p.requires_grad for p in perceive.visual.parameters())
    # Independently construct the archived experiment model and postprocessor.
    # They are a reference only, not imported by the production front end.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/detector_comparison_20261030'))
    from train_mask import make_model as archived_model
    from benchmark import sanitize as archived_sanitize
    reference=archived_model().to(args.device).eval().requires_grad_(False)
    reference.load_state_dict(torch.load(cfg.object_memory.detector_checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
    reference.roi_heads.score_thresh=min(cfg.object_memory.thresholds)
    archived_comparisons=0
    for spec in cfg.task_datasets:
        source=zarr.open_group(str(Path(cfg.data_root)/(spec.name+'.zarr')),mode='r')
        for index in (0,len(source['data/rgb'])-1):
            image=np.asarray(source['data/rgb'][index]);actual=perceive(image)
            with torch.inference_mode():
                pred=reference([torch.from_numpy(image.copy()).to(args.device).permute(2,0,1).float()/255])[0]
                objects=[dict(class_id=int(c)-1,box=b.tolist(),score=float(s))
                    for b,s,c in zip(pred['boxes'].cpu(),pred['scores'].cpu(),pred['labels'].cpu())
                    if float(s)>=cfg.object_memory.thresholds[int(c)-1]]
                objects=archived_sanitize(image,objects)
                expected=bounded_detections([(o['class_id'],np.asarray(o['box'],np.float32)/[120,90,120,90],o['score']) for o in objects],8)
                assert len(actual)==len(expected)
                if expected:
                    grid=perceive.visual(torch.from_numpy(image.copy()).to(args.device).permute(2,0,1)[None].float()/127.5-1)
                    features=raw_roi_features(grid,torch.tensor(np.stack([o[1] for o in expected]),device=args.device,dtype=torch.float32)).cpu().numpy()
                    for observation,(cls,box,score),feature in zip(actual,expected,features):
                        assert observation.class_id==cls
                        np.testing.assert_allclose(observation.box,box,atol=1e-5,rtol=1e-5)
                        np.testing.assert_allclose(observation.score,score,atol=1e-5,rtol=1e-5)
                        np.testing.assert_allclose(observation.feature,feature,atol=1e-5,rtol=1e-5)
            archived_comparisons+=1
        prepare_dataset(cfg,spec.name,perceive,output/'prepare_status.json')
    del reference

    initialize_modalities(cfg)
    datasets,tasks,norm,loaders,_=build_suite(cfg)
    saved_protocol=json.loads((Path(cfg.work_dir)/'object_protocol.json').read_text())
    validate_experiment_protocol(cfg,saved_protocol)
    encoder=ObjectMemoryEncoder(128,6).to(args.device).eval()
    boxer=build_main_boxer(cfg,args.device,1)
    online=OnlineObjects(cfg,perceive,1,boxer)
    checks=dict(arrays=0,tokens=0,training_samples=0,main_rgb_offline=0,main_rgb_online=0,normalized_objects=0,all_phase_anchors=0)
    rejections=[];max_error={k:0. for k in OBJECT_KEYS};nonempty=0
    for name,ds in datasets.items():
        online.reset_all();rgb=np.asarray(ds.base.replay_buffer['rgb'][:]);online_inputs=[];online_scenes=[]
        root=zarr.open_group(str(Path(cfg.object_memory.cache_root)/(name+'.objects.zarr')),mode='r')
        for anchor in range(0,len(rgb),cfg.action_steps):
            frames=[max(0,anchor-1),anchor]
            raw=rgb[frames].copy();obs={'rgb':np.moveaxis(raw,-1,1)[None].astype(np.float32)/255}
            input_copy=obs['rgb'].copy();result=online(obs)
            np.testing.assert_array_equal(input_copy,obs['rgb'])
            online_inputs.append(obs);online_scenes.append(result['rgb'].copy())
            snap={k:root[k][anchor] for k in OBJECT_KEYS}
            nonempty+=int(snap['valid'].sum())
            for mode in ('a1','a2','a3'):
                offline=select_mode(snap,mode)
                current=select_mode({k:result['_object_'+k][0] for k in OBJECT_KEYS},mode)
                for k in OBJECT_KEYS:
                    # Same frozen FP32 runtime: exact discrete fields, prescribed float tolerance.
                    if offline[k].dtype.kind=='f':
                        np.testing.assert_allclose(offline[k],current[k],atol=1e-5,rtol=1e-5)
                        max_error[k]=max(max_error[k],float(np.max(np.abs(offline[k]-current[k]))))
                    else:np.testing.assert_array_equal(offline[k],current[k])
                    checks['arrays']+=1
                with torch.inference_mode():
                    left,_=encoder({k:torch.as_tensor(v,device=args.device)[None] for k,v in offline.items()})
                    right,_=encoder({k:torch.as_tensor(v,device=args.device)[None] for k,v in current.items()})
                torch.testing.assert_close(left,right,atol=1e-5,rtol=1e-5);checks['tokens']+=1
            normalized=normalize_observations(result,norm,args.device)
            for k in OBJECT_KEYS:
                np.testing.assert_array_equal(normalized['_object_'+k].cpu(),result['_object_'+k]);checks['normalized_objects']+=1
        # Every real training anchor, including all cadence augmentation phases.
        # Recompute the stateless frozen front end independently of SQLite, then
        # feed only the causal prefix for each anchor to a fresh production bank.
        independent=[perceive(image) for image in rgb]
        for anchor in range(len(rgb)):
            stream=ObservationStream(make_bank(cfg),lambda index:independent[int(index)])
            for steps in query_schedule(anchor,int(cfg.action_steps)):
                stream.observe(steps,steps,[s*float(cfg.object_memory.period_seconds) for s in steps])
            for mode in ('a1','a2','a3'):
                offline=select_mode({k:root[k][anchor] for k in OBJECT_KEYS},mode)
                current=stream.bank.snapshot(mode)
                for k in OBJECT_KEYS:
                    if offline[k].dtype.kind=='f':
                        np.testing.assert_allclose(offline[k],current[k],atol=1e-5,rtol=1e-5)
                        max_error[k]=max(max_error[k],float(np.max(np.abs(offline[k]-current[k]))))
                    else:np.testing.assert_array_equal(offline[k],current[k])
                    checks['arrays']+=1
                with torch.inference_mode():
                    left,_=encoder({k:torch.as_tensor(v,device=args.device)[None] for k,v in offline.items()})
                    right,_=encoder({k:torch.as_tensor(v,device=args.device)[None] for k,v in current.items()})
                torch.testing.assert_close(left,right,atol=1e-5,rtol=1e-5);checks['tokens']+=1
            checks['all_phase_anchors']+=1
        # Independent replay of the unchanged archived main boxer.
        boxer.detector.reset_all()
        for obs,scene in zip(online_inputs,online_scenes):
            np.testing.assert_array_equal(boxer(obs)['rgb'],scene);checks['main_rgb_online']+=1
        for index in range(len(ds)):
            sample=ds[index];base=ds.base[index]
            for k in base['obs']:np.testing.assert_array_equal(sample['obs'][k],base['obs'][k])
            np.testing.assert_array_equal(sample['actions'],base['actions'])
            checks['main_rgb_offline']+=1
            batch=torch.utils.data.default_collate([sample])
            condition,action=split_batch(cfg,ds,tasks[name],batch)
            for k in OBJECT_KEYS:
                np.testing.assert_array_equal(condition['_object_'+k].cpu(),batch['objects'][k]);checks['training_samples']+=1
        # Actual dataset admission rejects old detector and modified detector protocol.
        signature=copy.deepcopy(root.attrs['signature'])
        writable=zarr.open_group(str(Path(cfg.object_memory.cache_root)/(name+'.objects.zarr')),mode='a')
        for label,patch in [('gdino',{'version':'gdino_raw_layer2_roi_v1'}),
                            ('thresholds',dict(signature['perception'],thresholds=[.1]*6))]:
            changed=copy.deepcopy(signature);changed['perception']=patch;writable.attrs['signature']=changed
            rejections.append(must_reject(lambda:ObjectDataset(ds.base,cfg,name),name+':'+label))
        writable.attrs['signature']=signature
        changed=copy.deepcopy(cfg);changed.object_memory.association.method='appearance_motion'
        rejections.append(must_reject(lambda:ObjectDataset(ds.base,changed,name),name+':association'))
        database=Path(cfg.object_memory.cache_root)/(name+'.perception.sqlite')
        with sqlite3.connect(str(database)) as con:
            value=con.execute('SELECT signature FROM metadata WHERE id=1').fetchone()[0]
            legacy=json.loads(value);legacy['perception']={'version':'gdino_raw_layer2_roi_v1'}
            con.execute('UPDATE metadata SET signature=? WHERE id=1',(json.dumps(legacy),));con.commit()
            rejections.append(must_reject(lambda:prepare_dataset(cfg,name,perceive,output/'prepare_status.json'),name+':legacy_sqlite'))
            con.execute('UPDATE metadata SET signature=? WHERE id=1',(value,));con.commit()
    legacy_protocol=copy.deepcopy(saved_protocol);legacy_protocol.pop('stream')
    rejections.append(must_reject(lambda:validate_experiment_protocol(cfg,legacy_protocol),'evaluation:missing_stream'))
    changed_protocol=copy.deepcopy(saved_protocol);changed_protocol['stream']['slots']=16
    rejections.append(must_reject(lambda:validate_experiment_protocol(cfg,changed_protocol),'evaluation:changed_stream'))
    assert nonempty>0, 'Cannot accept an all-padding comparison'
    result=dict(scope='real_training_prefixes_not_policy_training_or_rollout',provenance=provenance,
        tolerance=dict(atol=1e-5,rtol=1e-5,discrete='exact'),checks=checks,max_abs_array_error=max_error,
        nonempty_slots_at_queries=nonempty,archived_frontend_frames=archived_comparisons,rejected_protocols=rejections,
        perception=perceive.metadata,stream=stream_contract(cfg),passed=True)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('perception','provenance')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='momart_object_memory/config.yaml')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--frames',type=int,default=25)
    verify(p.parse_args())
