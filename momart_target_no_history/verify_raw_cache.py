"""Recompute sampled cache entries from raw causal main sensors and detector."""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
RESEARCH=Path(__file__).resolve().parents[1]/'experiments/momart_target_research_20261101'
sys.path.insert(0,str(RESEARCH))
from spatial_frontend import SpatialFrontEnd,RobotObservation
from momart_object_memory.state import iou
from .prepare_cache import dataset
from .selector import UnifiedTargetSelector

OLD_TASKS=['table_cleanup_to_dishwasher_expert','table_setup_from_dishwasher_expert','table_setup_from_dresser_expert','unload_dishwasher_to_dresser_suboptimal']


def main(a):
    cfg=OmegaConf.load(a.config);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    selector=UnifiedTargetSelector(a.selector_bundle);frontend=None;parameters=None;records=[]
    try:
        for spec in cfg.task_datasets:
            if a.sources and str(spec.name) not in a.sources:continue
            name=str(spec.name);d=dataset(cfg,name);root=zarr.open_group(str(a.cache_root/(name+'.bbox.zarr')),mode='r')
            if root.attrs['complete'] is not True:raise ValueError('Raw replay requires completed source: '+name)
            layout=json.loads((a.cache_root/(name+'.robot_layout.json')).read_text())
            with h5py.File(a.hdf_root/(name+'.hdf5'),'r') as hdf:
                ig=json.loads(hdf['data'].attrs['env_args'])['env_kwargs']['ig_config'];new=(float(ig['vertical_fov']),float(ig['depth_high']),float(ig.get('robot_scale',1.)))
                if parameters!=new:
                    if frontend is not None:frontend.close()
                    frontend=SpatialFrontEnd(cfg.object_memory,a.urdf,OLD_TASKS,a.device,*new,include_visual=True);parameters=new
                col=layout['offset']+13+frontend.geometry.movable.index(frontend.geometry.joints['torso_lift_joint'])
                for index in [0,len(d)//2,len(d)-1]:
                    episode=int(np.searchsorted(d.replay_buffer.episode_ends,index,side='right'))
                    start=0 if episode==0 else int(d.replay_buffer.episode_ends[episode-1]);demo=list(d._demo_id_to_start_indices)[episode]
                    local=index-start;ep=hdf['data'][demo];obs=ep['obs'];episode_id=name+'/'+demo
                    def read(step):return RobotObservation(np.asarray(obs['rgb'][step]),np.asarray(obs['depth'][step]),np.asarray(obs['proprio'][step]),float(ep['states'][step,col]),episode_id,step,OLD_TASKS[0])
                    frame=frontend.build(read(local),read(max(0,local-8)));actual=selector.select(frame,name,episode_id,local)
                    expected_valid=bool(root['valid'][index]);expected_class=int(root['class_id'][index]);box=np.asarray(root['boxes'][index]);overlap=float(iou(box,actual['box'])) if expected_valid and actual['valid'] else None
                    box_max_abs=float(abs(box-actual['box']).max()) if overlap is not None else None
                    records.append(dict(source=name,index=index,episode=episode_id,step=local,
                        valid_equal=actual['valid']==expected_valid,class_equal=actual['class_id']==expected_class if actual['valid'] else expected_class==-1,
                        box_iou=overlap,box_max_abs=box_max_abs))
                    print(json.dumps(records[-1]),flush=True)
    finally:
        if frontend is not None:frontend.close()
    value=dict(queries=len(records),valid_disagreements=sum(not r['valid_equal'] for r in records),class_disagreements=sum(not r['class_equal'] for r in records),
        min_box_iou=min([r['box_iou'] for r in records if r['box_iou'] is not None],default=None),
        max_box_abs=max([r['box_max_abs'] for r in records if r['box_max_abs'] is not None],default=None),records=records,
        scope='Three raw sensor samples/source, batch1 detector and original frontend; no object GT or future inputs; not all-frame parity or policy closed loop')
    a.output.write_text(json.dumps(value,indent=2));print(json.dumps({k:v for k,v in value.items() if k!='records'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','cache-root','hdf-root','selector-bundle','urdf','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--sources',nargs='+');main(p.parse_args())
