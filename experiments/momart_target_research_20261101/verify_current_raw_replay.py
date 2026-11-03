"""Recompute the video's fixed selector from causal raw main-camera sensors."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from runtime import FrameFeatures,SingleBoxSelector,view
from spatial_frontend import SpatialFrontEnd,RobotObservation


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=a.root/'momart_target_research_20261101';selection=r/'frozen_for_confirm6_incumbent_blend/selected.json'
    cfg=OmegaConf.load(a.config).object_memory;tasks=json.loads((a.root/'momart_target_selector_20261101/features.json').read_text())['tasks']
    runtime=SingleBoxSelector(selection,a.root);reference_runtime=SingleBoxSelector(selection,a.root);frontend=None;files={};records=[]
    a.output.mkdir(parents=True)
    (a.output/'protocol.json').write_text(json.dumps(dict(selection_sha256=sha(selection),urdf_sha256=sha(a.urdf),
        episodes=a.episodes,main_view_only=True,device=a.device,physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),tf32=False,
        scope='Recompute all archived video query steps from current and t-8 raw main RGB, current observed depth and own robot proprio/torso; no labels, target boxes, object GT or future frame enters online API. Recorded sensor replay, not closed-loop actions.'),indent=2))
    try:
        for prefix,episode in zip(a.prefixes,a.episodes):
            manifest=json.loads((r/(prefix+'_corpus')/'manifest.json').read_text());base_meta=json.loads((r/(prefix+'_features.json')).read_text());geometry_meta=json.loads((r/('geometry_'+prefix+'.json')).read_text())
            assert sha(r/(prefix+'_corpus')/'manifest.json')==base_meta['manifest_sha256']==geometry_meta['manifest_sha256']
            assert sha(a.urdf)==geometry_meta['urdf_sha256']
            assert base_meta['ids']==geometry_meta['ids']==[row['id'] for row in manifest['records']]
            base=dict(np.load(r/(prefix+'_features.npz')));geometry=dict(np.load(r/('geometry_'+prefix+'.npz')))
            indices=[i for i,row in enumerate(manifest['records']) if row['episode_id']==episode]
            if not indices:raise ValueError('Episode not found: '+episode)
            for i in indices:
                row=manifest['records'][i];source=row['source']
                if source not in files:files[source]=h5py.File(source,'r')
                f=files[source];ep=f['data'][row['demo']];obs=ep['obs'];ig=json.loads(f['data'].attrs['env_args'])['env_kwargs']['ig_config']
                parameters=(float(ig['vertical_fov']),float(ig['depth_high']),float(ig.get('robot_scale',1.)))
                if frontend is None or parameters!=(frontend.fov,frontend.depth_high,frontend.geometry_scale):
                    if frontend is not None:frontend.close()
                    frontend=SpatialFrontEnd(cfg,a.urdf,tasks,a.device,*parameters,include_visual=True);frontend.geometry_scale=parameters[-1]
                column=geometry_meta['layouts'][source]['offset']+13+frontend.geometry.movable.index(frontend.geometry.joints['torso_lift_joint'])
                def measurement(step):return RobotObservation(np.asarray(obs['rgb'][step]),np.asarray(obs['depth'][step]),np.asarray(obs['proprio'][step]),float(ep['states'][step,column]),episode,step,row['task'])
                current=measurement(row['step']);past=measurement(max(0,row['step']-8))
                assert hashlib.sha256(current.rgb.tobytes()).hexdigest()==row['sha256']
                started=time.perf_counter();frame=frontend.build(current,past);frontend_end=time.perf_counter()
                actual=runtime.select(frame,episode,row['step']);ended=time.perf_counter()
                reference=FrameFeatures(base['candidates'][i],base['context'][i],base['valid'][i],row['task'],geometry_candidates=geometry['candidates'][i],geometry_context=geometry['context'][i])
                expected=reference_runtime.select(reference,episode,row['step']);c,x=view(frame,'geometry');rc,rx=view(reference,'geometry')
                box_delta=float(np.max(abs(np.asarray(actual['box'])-np.asarray(expected['box'])))) if actual['valid'] and expected['valid'] else None
                records.append(dict(id=row['id'],episode=episode,step=row['step'],slot_equal=actual['slot']==expected['slot'],actual_slot=actual['slot'],expected_slot=expected['slot'],
                    candidate_valid_equal=bool(np.array_equal(frame.valid,reference.valid)),max_feature_abs=float(max(abs(c-rc).max(),abs(x-rx).max())),
                    probability_max_abs=float(abs(actual['probability']-expected['probability']).max()),box_max_abs=box_delta,
                    frontend_ms=1000*(frontend_end-started),pipeline_ms=1000*(ended-started)))
                (a.output/'progress.json').write_text(json.dumps(dict(queries=len(records),latest=row['id'])))
    finally:
        for f in files.values():f.close()
        if frontend is not None:frontend.close()
    result=dict(queries=len(records),episodes=len(a.episodes),slot_disagreements=sum(not q['slot_equal'] for q in records),
        candidate_valid_disagreements=sum(not q['candidate_valid_equal'] for q in records),max_probability_abs=max(q['probability_max_abs'] for q in records),
        max_box_abs=max([q['box_max_abs'] for q in records if q['box_max_abs'] is not None],default=None),
        median_pipeline_ms=float(np.median([q['pipeline_ms'] for q in records])),p95_pipeline_ms=float(np.quantile([q['pipeline_ms'] for q in records],.95)),
        first_pipeline_ms=records[0]['pipeline_ms'],records=records,
        caveat='Raw causal recorded sensor replay at archived query times, including unknown-intent observations. Not interactive environment execution, action-policy integration or closed-loop task success. Timings include perception and selector, not sensor I/O/actions.')
    result['same_accepted_slots']=result['slot_disagreements']==0
    (a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    if not result['same_accepted_slots']:raise ValueError('Raw online-style outputs differ from archived video; inspect every disagreement before claiming parity')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','config','urdf','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prefixes',nargs='+',default=['confirm7','confirm9']);p.add_argument('--episodes',nargs='+',default=['table_setup_from_dishwasher_expert/demo_109','unload_dishwasher_to_dresser_suboptimal/demo_42'])
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if len(a.prefixes)!=len(a.episodes):p.error('Each episode requires its matching corpus prefix')
    main(a)
