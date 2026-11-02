"""Compare lean raw sensors with archived spatial features on fixed, uniform samples."""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from spatial_frontend import SpatialFrontEnd,RobotObservation
from runtime import SingleBoxSelector,FrameFeatures,view


def main(a):
    torch.set_num_threads(2);r=a.root/'momart_target_research_20261101';corpus=r/'dense_corpus'
    meta=json.loads((corpus/'manifest.json').read_text());base=dict(np.load(r/'dense_features.npz'));geometry=dict(np.load(r/'geometry_dense.npz'));masks=dict(np.load(r/'mask_dense.npz'))
    layouts=json.loads((r/'geometry_dense.json').read_text())['layouts'];tasks=json.loads((r/'dense_features.json').read_text())['tasks']
    runtime=SingleBoxSelector(a.selection,a.root);component=runtime.component
    while component['kind']=='availability':component=component['parent']
    if component['kind']!='parent':raise ValueError('This stateless parity check requires a direct spatial checkpoint')
    ckpt=torch.load(Path(component['screen'])/(component['name']+'.pt'),map_location='cpu',weights_only=False)
    if ckpt['spec']['view'] not in (['base','geometry','spatial','mask_small','mask_spatial'] if a.visual else ['spatial','mask_small','mask_spatial']) or ckpt['spec']['kind']=='gru':raise ValueError('Expected stateless spatial checkpoint')
    cfg=OmegaConf.load(a.config).object_memory;files={};frontend=None;records=[]
    try:
        for i in np.linspace(0,len(meta['records'])-1,a.samples,dtype=int):
            row=meta['records'][i]
            if row['source'] not in files:files[row['source']]=h5py.File(row['source'],'r')
            f=files[row['source']];ep=f['data'][row['demo']];obs=ep['obs'];ig=json.loads(f['data'].attrs['env_args'])['env_kwargs']['ig_config']
            if frontend is None:frontend=SpatialFrontEnd(cfg,a.urdf,tasks,a.device,float(ig['vertical_fov']),float(ig['depth_high']),float(ig.get('robot_scale',1.)),include_visual=a.visual)
            column=layouts[row['source']]['offset']+13+frontend.geometry.movable.index(frontend.geometry.joints['torso_lift_joint'])
            def measurement(t):return RobotObservation(np.asarray(obs['rgb'][t]),np.asarray(obs['depth'][t]),np.asarray(obs['proprio'][t]),float(ep['states'][t,column]),row['episode_id'],t,row['task'])
            current=measurement(row['step']);past=measurement(max(0,row['step']-8));started=time.perf_counter();frame=frontend.build(current,past);seconds=time.perf_counter()-started
            new=runtime.select(frame,row['episode_id']+'::new',row['step']);pipeline_seconds=time.perf_counter()-started
            reference=FrameFeatures(base['candidates'][i],base['context'][i],base['valid'][i],row['task'],geometry_candidates=geometry['candidates'][i],geometry_context=geometry['context'][i],mask_candidates=masks['candidates'][i])
            old=runtime.select(reference,row['episode_id']+'::reference',row['step'])
            c,x=view(frame,ckpt['spec']['view']);rc,rx=view(reference,ckpt['spec']['view'])
            records.append(dict(id=row['id'],slot_equal=new['slot']==old['slot'],valid_slots_equal=bool(np.array_equal(frame.valid,reference.valid)),
                feature_max_error=float(max(np.max(abs(c-rc)),np.max(abs(x-rx)))),probability_max_error=float(np.max(abs(new['probability']-old['probability']))),frontend_ms=seconds*1000,pipeline_ms=pipeline_seconds*1000))
    finally:
        for f in files.values():f.close()
        if frontend is not None:frontend.close()
    result=dict(samples=len(records),sampling='uniform indices fixed before inference',device=a.device,records=records,
        slot_disagreements=sum(not row['slot_equal'] for row in records),valid_slot_disagreements=sum(not row['valid_slots_equal'] for row in records),
        include_visual=a.visual,median_frontend_ms=float(np.median([row['frontend_ms'] for row in records])),median_pipeline_ms=float(np.median([row['pipeline_ms'] for row in records])),p95_pipeline_ms=float(np.quantile([row['pipeline_ms'] for row in records],.95)),caveat='CPU/GPU and batching differ; predicted mask threshold may change near-boundary pixels. No human/scene-object GT enters raw frontend. Small engineering parity check, not reliability evaluation.')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','selection','config','urdf','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--samples',type=int,default=24);p.add_argument('--visual',action='store_true');main(p.parse_args())
