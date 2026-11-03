"""Evaluate fixed detector-weight interpolation without changing postprocessing."""
import argparse
import json
from pathlib import Path
import os
import torch
from omegaconf import OmegaConf
from detector_interpolation import interpolate_detector
from train_perception_adaptation import sha, state_digest, evaluate, detector_gate
from growth_data import load_growth_data
from perception_data import collect_detector_queries, DetectorFrames
from momart_object_memory.mask_detector import make_model


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    status=json.loads((a.study/'status.json').read_text())
    if status['stage']!='complete' or status['completed_checkpoints']!=6:
        raise ValueError('Parent paired study must finish all six real checkpoints')
    parent=json.loads((a.study/'protocol.json').read_text())
    if parent['smoke']:raise ValueError('Smoke checkpoints cannot be screened')
    registry=json.loads((a.study/'registry.json').read_text())
    cfg=OmegaConf.load(a.config).object_memory
    assert sha(Path(cfg.detector_checkpoint))==cfg.detector_checkpoint_sha256==parent['initializer_sha256']
    assert list(map(float,cfg.thresholds))==parent['thresholds']
    research=a.root/'momart_target_research_20261101';data=load_growth_data(a.root)
    queries,hashes=collect_detector_queries(data,research)
    assert hashes==parent['source_hashes']
    assert [x['row']['id'] for x in queries['dev']]==parent['query_ids']['dev']
    reserve=research/'confirm10_corpus'
    assert sha(reserve/'manifest.json')==parent['holdout_manifest_sha256']
    assert not any((reserve/name).exists() for name in ['annotations_all.json','candidate_manifest.json'])
    baseline=json.loads((a.study/'baseline_development.json').read_text())
    assert baseline['weak_targets']['cached_availability_disagreements']==0
    assert baseline['frames']==len(queries['dev'])
    parents={name:dict(path=v['checkpoint'],sha256=v['checkpoint_sha256']) for name,v in registry.items()}
    assert len(parents)==6
    for value in parents.values():assert sha(Path(value['path']))==value['sha256']
    a.output.mkdir(parents=True)
    protocol=dict(method='detector head weight interpolation toward shared production initializer',
        alphas=[.1,.25,.5],parents=parents,parent_protocol_sha256=sha(a.study/'protocol.json'),
        baseline_sha256=sha(a.study/'baseline_development.json'),initializer_sha256=parent['initializer_sha256'],
        source_hashes=hashes,development_ids=parent['query_ids']['dev'],physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        holdout_manifest_sha256=parent['holdout_manifest_sha256'],holdout_opened=False,
        acceptance='Unchanged detector feasibility gate; no selector or closed-loop acceptance',
        ranking='Eligible first, acquire availability, all target availability, edge-small matches, micro precision',
        thresholds=parent['thresholds'],postprocessing=parent['postprocessing'],
        note='Exploratory follow-up to observed FP inflation; all 18 fixed recipes evaluated on unchanged development, no threshold search')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    initial=torch.load(cfg.detector_checkpoint,map_location='cpu',weights_only=False)['model']
    backbone=state_digest(initial,'backbone.');dataset=DetectorFrames(queries['dev']);results={}
    model=make_model();model.roi_heads.score_thresh=min(parent['thresholds']);model.to(a.device)
    for name,item in parents.items():
        adapted=torch.load(item['path'],map_location='cpu',weights_only=False)['model']
        for alpha in protocol['alphas']:
            recipe=f'{name}_alpha{alpha}'
            state=interpolate_detector(initial,adapted,alpha)
            assert state_digest(state,'backbone.')==backbone
            model.load_state_dict(state,strict=True)
            value=evaluate(model,dataset,data,parent['thresholds'],a.device,16);gate=detector_gate(value,baseline)
            (a.output/(recipe+'_development.json')).write_text(json.dumps(value,indent=2))
            results[recipe]=dict(name=recipe,parent=name,alpha=alpha,state_sha256=state_digest(state),gate=gate,
                metrics={k:v for k,v in value.items() if k not in ['records','frame_availability']})
            (a.output/'registry.json').write_text(json.dumps(results,indent=2))
            (a.output/'status.json').write_text(json.dumps(dict(stage='evaluating',completed=len(results),total=18),indent=2))
            print('RECIPE',recipe,json.dumps(results[recipe]),flush=True)
    eligible=[v for v in results.values() if v['gate']['eligible']]
    winner=max(eligible,key=lambda v:(v['metrics']['weak_targets']['acquire_available'],v['metrics']['weak_targets']['available'],
        v['metrics']['edge_small']['matched'],v['metrics']['bounded8']['micro']['precision'])) if eligible else None
    if winner:
        adapted=torch.load(parents[winner['parent']]['path'],map_location='cpu',weights_only=False)['model']
        torch.save(dict(model=interpolate_detector(initial,adapted,winner['alpha']),recipe=winner,
            protocol_sha256=sha(a.output/'protocol.json')),a.output/'selected_detector.pt')
    (a.output/'selected.json').write_text(json.dumps(dict(result=winner,no_promotion=True,selector_evaluated=False,confirmation_evaluated=False),indent=2))
    (a.output/'status.json').write_text(json.dumps(dict(stage='complete',completed=len(results),winner=winner['name'] if winner else 'original_frozen_detector'),indent=2))
    print('COMPLETE',winner['name'] if winner else 'original_frozen_detector',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','study','config','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');main(p.parse_args())
