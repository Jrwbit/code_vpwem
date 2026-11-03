"""Resumable ten-source target cache, reusing verified frozen perception only."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import h5py
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
RESEARCH=Path(__file__).resolve().parents[1]/'experiments/momart_target_research_20261101'
sys.path.insert(0,str(RESEARCH))
from runtime import FrameFeatures
from robot_geometry import RobotGeometry,box_geometry
from momart_no_history_baseline.dataset import ZarrNoHistorySequenceDataset
from momart_object_memory.prepare_train import decode
from momart_object_memory.perception import FrozenPerception,file_sha256
from .dataset import STANDARD,source_signature
from .inventory import inventory
from .selector import UnifiedTargetSelector


def save_json(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def dataset(config,name):
    return ZarrNoHistorySequenceDataset(Path(config.data_root)/(name+'.zarr'),['rgb','depth','proprio'],9,2,20)


def contract(a,cfg,audit):
    files={str(path):file_sha256(path) for path in [a.urdf,RESEARCH/'runtime.py',RESEARCH/'robot_geometry.py',Path(__file__),Path(__file__).with_name('selector.py')]}
    value=dict(perception=audit['perception'],selector_manifest_sha256=file_sha256(a.selector_bundle/'manifest.json'),implementation_and_robot_sha256=files,
        causal_past_offset=8,sample_policy='every logical training frame, same-episode t-8; no future/object labels',
        source_mapping='expert/suboptimal share one of five task IDs; all ten sources use ONE frozen five-task model and identical inference',
        camera='main RGB and depth only; own proprio and own torso; original policy wrist remains untouched',
        global_feature_dtype='float32',global_batch_size=a.batch_size,tf32=False)
    digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    meta=dict(annotation_standard=STANDARD,bundle_manifest_sha256=value['selector_manifest_sha256'],
        sensor_protocol_sha256=digest,supported_sources=[str(s.name) for s in cfg.task_datasets],protocol=value)
    return meta


def global_features(a,cfg,name,perceive,metadata):
    d=dataset(cfg,name);n=len(d);path=a.output/(name+'.global.npy');status=a.output/(name+'.global.json')
    signature=dict(source=source_signature(d,Path(cfg.data_root)/(name+'.zarr')),sensor_protocol_sha256=metadata['sensor_protocol_sha256'])
    if status.exists():
        old=json.loads(status.read_text())
        if old['signature']!=signature:raise ValueError('Global feature protocol differs')
        completed=old['completed_frames']
        grid=np.lib.format.open_memmap(path,mode='r+')
    else:
        if path.exists():raise ValueError('Uncommitted global feature file: '+str(path))
        completed=0;grid=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,128))
        save_json(status,dict(signature=signature,completed_frames=0,complete=False))
    started=time.monotonic();initial=completed
    with torch.inference_mode():
        for start in range(completed,n,a.batch_size):
            end=min(start+a.batch_size,n);images=d.replay_buffer['rgb'].read_frames(np.arange(start,end))
            tensor=torch.from_numpy(images).permute(0,3,1,2).to(a.device,dtype=torch.float32)/127.5-1
            grid[start:end]=perceive.visual(tensor).mean((-1,-2)).cpu().numpy();grid.flush()
            save_json(status,dict(signature=signature,completed_frames=end,complete=end==n))
            if end//4096 != start//4096 or end==n:
                print(json.dumps(dict(stage='global_features',source=name,completed=end,total=n,fps=(end-initial)/max(time.monotonic()-started,.001))),flush=True)
    return path


def public_frame(geometry,objects,current_global,past_global,proprio,past_proprio,torso,past_torso,depth,fov,depth_high):
    ee,(uv,forward)=geometry.project(proprio,torso,fov);old_ee,(old_uv,_)=geometry.project(past_proprio,past_torso,fov)
    c=np.zeros((8,139),np.float32);valid=np.zeros(8,bool);g=np.zeros((8,16),np.float32)
    for slot,obj in enumerate(objects):
        c[slot]=np.r_[obj.feature,obj.box,obj.score,np.eye(6)[obj.class_id]];valid[slot]=True
        g[slot]=box_geometry(obj.box,depth,ee,fov,depth_high)
    x=np.r_[current_global,past_global,proprio,proprio-past_proprio,[1,0,0,0]].astype(np.float32)
    gx=np.r_[ee,uv,torso,float(forward),fov/90,depth_high,ee-old_ee,uv-old_uv].astype(np.float32)
    return FrameFeatures(c,x,valid,'table_cleanup_to_dishwasher_expert',geometry_candidates=g,geometry_context=gx)


def write_source(config_path,name,output,perception_root,selector_bundle,urdf,metadata):
    torch.set_num_threads(1);cfg=OmegaConf.load(config_path);output=Path(output)
    source=Path(cfg.data_root)/(name+'.zarr');d=dataset(cfg,name);n=len(d);target=output/(name+'.bbox.zarr')
    with (output/(name+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return write_locked(d,source,target,name,output,perception_root,selector_bundle,urdf,metadata)


def write_locked(d,source,target,name,output,perception_root,selector_bundle,urdf,metadata):
    n=len(d);signature=source_signature(d,source)
    root=zarr.open_group(str(target),mode='a')
    if 'signature' in root.attrs:
        if root.attrs['signature']!=signature or root.attrs['target_metadata']!=metadata:raise ValueError('Sidecar resume protocol differs')
        completed=int(root.attrs['completed_frames'])
    else:
        completed=0;root.attrs.update(signature=signature,target_metadata=metadata,completed_frames=0,complete=False)
    if not 0<=completed<=n:raise ValueError('Invalid commit cursor')
    arrays={key:root.require_dataset(key,shape=shape,dtype=dtype,chunks=chunks) for key,shape,dtype,chunks in [
        ('boxes',(n,4),'f4',(512,4)),('valid',(n,),'bool',(512,)),('class_id',(n,),'i1',(512,)),('confidence',(n,),'f4',(512,))]}
    if root.attrs['complete']:
        if completed!=n:raise ValueError('Complete cursor differs')
        return dict(source=name,frames=n,already_complete=True)
    global_status=json.loads((output/(name+'.global.json')).read_text())
    if not global_status['complete'] or global_status['signature']['sensor_protocol_sha256']!=metadata['sensor_protocol_sha256']:raise ValueError('Global cache incomplete')
    global_grid=np.load(output/(name+'.global.npy'),mmap_mode='r');selector=UnifiedTargetSelector(selector_bundle)
    # This frozen protocol contains only feed-forward pointers, so a committed
    # frame is sufficient for resume; no hidden temporal state is discarded.
    for runtime in [selector.runtime]:
        def verify(component):
            if component['kind']=='parent':
                if runtime.load(Path(component['screen'])/(component['name']+'.pt'))['spec']['kind']=='gru':raise ValueError('Frame resume not valid for a recurrent pointer')
            elif component['kind']=='blend':
                for member in component['members']:verify(member)
            elif component['kind']=='availability':verify(component['parent'])
            else:raise ValueError('Unsupported frozen cache component')
        verify(runtime.component)
    db=Path(perception_root)/(name+'.perception.sqlite');con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
    zroot=zarr.open_group(str(source),mode='r');source_hdf=Path(zroot.attrs.get('source_path',str(Path('/home/jrw/vpwem_data/momart')/(name+'.hdf5'))))
    geometry=None;started=time.monotonic();initial=completed;cursor=0
    try:
        with h5py.File(source_hdf,'r') as hdf:
            ig=json.loads(hdf['data'].attrs['env_args'])['env_kwargs']['ig_config']
            geometry=RobotGeometry(urdf,float(ig.get('robot_scale',1.)));fov=float(ig['vertical_fov']);depth_high=float(ig['depth_high'])
            layout_file=output/(name+'.robot_layout.json')
            if layout_file.exists():layout=json.loads(layout_file.read_text())
            else:
                demo=next(iter(d._demo_id_to_start_indices));ep=hdf['data'][demo]
                layout=geometry.infer_robot_layout(np.asarray(ep['states']),np.asarray(ep['obs/proprio']));save_json(layout_file,layout)
            col=layout['offset']+13+geometry.movable.index(geometry.joints['torso_lift_joint'])
            for demo,end in zip(d._demo_id_to_start_indices,d.replay_buffer.episode_ends):
                end=int(end);start=cursor;cursor=end
                if end<=completed:continue
                ep=hdf['data'][demo];proprio=np.asarray(ep['obs/proprio']);torso=np.asarray(ep['states'][:,col])
                if len(proprio)!=end-start:raise ValueError('Source demo length disagrees with training view')
                for begin in range(max(start,completed),end,512):
                    stop=min(begin+512,end);frames=np.arange(begin,stop);local=frames-start;past=np.maximum(0,local-8)
                    depths=np.asarray(ep['obs/depth'][int(local[0]):int(local[-1])+1]);items=con.execute('SELECT frame,payload FROM observations WHERE frame>=? AND frame<? ORDER BY frame',(begin,stop)).fetchall()
                    if [v[0] for v in items]!=frames.tolist():raise ValueError('Perception frame gap')
                    boxes=np.zeros((len(frames),4),np.float32);valid=np.zeros(len(frames),bool);classes=np.full(len(frames),-1,np.int8);confidence=np.zeros(len(frames),np.float32)
                    for k,((frame,blob),t,old) in enumerate(zip(items,local,past)):
                        public=public_frame(geometry,decode(blob),global_grid[frame],global_grid[start+old],proprio[t],proprio[old],float(torso[t]),float(torso[old]),depths[k],fov,depth_high)
                        result=selector.select(public,name,name+'/'+demo,int(t));valid[k]=result['valid'];confidence[k]=result['confidence']
                        if valid[k]:boxes[k]=result['box'];classes[k]=result['class_id']
                    for key,value in [('boxes',boxes),('valid',valid),('class_id',classes),('confidence',confidence)]:arrays[key][begin:stop]=value
                    root.attrs['completed_frames']=stop
                    rate=(stop-initial)/max(time.monotonic()-started,.001)
                    save_json(output/(name+'.status.json'),dict(stage='target_cache',source=name,completed_frames=stop,total_frames=n,fps=rate,remaining_seconds=(n-stop)/max(rate,.001)))
    finally:
        con.close()
        if geometry is not None:geometry.close()
    hashes={key:hashlib.sha256(np.asarray(arrays[key]).tobytes()).hexdigest() for key in arrays}
    root.attrs.update(complete=True,completed_frames=n,array_sha256=hashes)
    result=dict(source=name,frames=n,valid_frames=int(np.asarray(arrays['valid']).sum()),seconds=time.monotonic()-started,array_sha256=hashes)
    save_json(output/(name+'.verified.json'),result);print(json.dumps(result),flush=True);return result


def main(a):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=OmegaConf.load(a.config);a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit=inventory(cfg,a.perception_root);save_json(a.output/'inventory.json',audit)
        metadata=contract(a,cfg,audit);metadata_path=a.output/'target_metadata.json'
        if metadata_path.exists() and json.loads(metadata_path.read_text())!=metadata:raise ValueError('Frozen target metadata differs')
        save_json(metadata_path,metadata)
        perceive=FrozenPerception(cfg.object_memory,a.device)
        for row in audit['sources']:global_features(a,cfg,row['source'],perceive,metadata)
        del perceive;torch.cuda.empty_cache()
        # Spawn avoids inheriting initialized CUDA or HDF5 handles into workers.
        import multiprocessing
        with concurrent.futures.ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs=[pool.submit(write_source,a.config,row['source'],a.output,a.perception_root,a.selector_bundle,a.urdf,metadata) for row in audit['sources']]
            results=[job.result() for job in jobs]
        save_json(a.output/'summary.json',dict(complete=True,sources=results,total_frames=sum(r['frames'] for r in results),policy_training_started=False))
        print('ALL_TEN_SOURCES_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','perception-root','selector-bundle','urdf','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=128);p.add_argument('--workers',type=int,default=4);main(p.parse_args())
