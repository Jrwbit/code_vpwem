"""Paired detector-head adaptation on train108, original fixed development only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from growth_data import load_growth_data,growth_partitions
from perception_data import collect_detector_queries,DetectorFrames,key
from detector_metrics import instance_counts,aggregate_counts,iou
from proxy import target_candidate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from momart_object_memory.mask_detector import make_model,select_boxes
from momart_object_memory.perception import bounded_detections


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def state_digest(state,prefix=''):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        if k.startswith(prefix):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def public(items):return [dict(class_id=int(c),score=float(s),box=(np.asarray(b)*[120,90,120,90]).tolist()) for c,b,s in items]


def evaluate(model,dataset,data,thresholds,device,batch_size):
    model.eval();records=[];object_scores=[];full_scores=[];weak=defaultdict(int);edge=defaultdict(int);frame_availability=[]
    with torch.inference_mode():
        for start in range(0,len(dataset),batch_size):
            batch=[dataset[i] for i in range(start,min(start+batch_size,len(dataset)))];images,targets=zip(*batch)
            predictions=model([x.to(device) for x in images])
            for j,(image,target,prediction) in enumerate(zip(images,targets,predictions)):
                item=dataset.queries[start+j];idx=item['base_index'];row=item['row'];rgb=(image.permute(1,2,0).numpy()*255).round().astype(np.uint8)
                truth=[dict(class_id=int(c)-1,box=b.tolist()) for b,c in zip(target['boxes'],target['labels'])]
                detections=select_boxes(rgb,prediction,thresholds);objects=public(bounded_detections(detections,8))
                counts,matched=instance_counts(objects,truth,return_matches=True);object_scores.append(counts);full_scores.append(instance_counts(public(detections),truth))
                for truth_index,t in enumerate(truth):
                    b=t['box'];thin=min(b[2]-b[0],b[3]-b[1])<=8;touch=b[0]<=1 or b[1]<=1 or b[2]>=119 or b[3]>=89
                    if not thin or not touch:continue
                    edge['visible']+=1;edge['matched']+=truth_index in matched
                available=None
                if data['y'][idx]>=0:
                    weak['known']+=1;weak['visible']+=int(data['visible'][idx])
                    refs=item['source']['objects'];available=target_candidate(objects,refs,row['label']['target'])>=0;weak['available']+=available
                    weak['cached_original_available']+=bool(data['positive'][idx].any());weak['cached_availability_disagreements']+=available!=bool(data['positive'][idx].any())
                    if data['roles'][idx]==0:weak['acquire_queries']+=1;weak['acquire_available']+=available
                    frame_availability.append(dict(id=row['id'],available=bool(available),cached_available=bool(data['positive'][idx].any())))
                records.append(dict(id=row['id'],episode_id=row['episode_id'],objects=objects))
    return dict(frames=len(dataset),bounded8=aggregate_counts(object_scores),pre_capacity=aggregate_counts(full_scores),weak_targets=dict(weak),edge_small=dict(edge),records=records,frame_availability=frame_availability)


def detector_gate(value,baseline):
    v=value['bounded8'];b=baseline['bounded8'];w=value['weak_targets'];old=baseline['weak_targets']
    floors={c:max(0.,(x['recall'] or 0)-.02) for c,x in b['classes'].items() if x['tp']+x['fn']>=20}
    checks=dict(target_proposals_retained=w['available']>=old['available'],acquire_proposals_retained=w.get('acquire_available',0)>=old.get('acquire_available',0),
        false_boxes_not_increased=v['micro']['fp']<=b['micro']['fp'],false_bowls_not_increased=v['classes']['0']['fp']<=b['classes']['0']['fp'],
        per_class_recall_retained=all((v['classes'][c]['recall'] or 0)>=floor for c,floor in floors.items()),
        actual_proposal_gain=w['available']>old['available'] or value['edge_small'].get('matched',0)>baseline['edge_small'].get('matched',0))
    return dict(checks,eligible=all(checks.values()),recall_floors=floors,caveat='Detector feasibility only; no selector or closed-loop success claim')


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=a.root/'momart_target_research_20261101';data=load_growth_data(a.root);queries,source_hashes=collect_detector_queries(data,r);reserve=r/'confirm10_corpus'
    assert not any((reserve/n).exists() for n in ['annotations_all.json','candidate_manifest.json'])
    reserved={x['episode_id'] for x in json.loads((reserve/'manifest.json').read_text())['episodes']}
    assert not {x['row']['episode_id'] for part in queries.values() for x in part}&reserved
    cfg=OmegaConf.load(a.config).object_memory;thresholds=list(map(float,cfg.thresholds));assert sha(Path(cfg.detector_checkpoint))==cfg.detector_checkpoint_sha256
    a.output.mkdir(parents=True);steps=2 if a.smoke else 1000;checkpoints=[2] if a.smoke else [250,500,1000]
    train=DetectorFrames(queries['train']);dev=DetectorFrames(queries['dev'][:16] if a.smoke else queries['dev'])
    rng=np.random.default_rng(20261104);schedule=[dict(indices=rng.integers(0,len(train),2).tolist(),cropped=(rng.random(2)<.5).tolist(),crops=[[int(rng.integers(0,25)),int(rng.integers(0,19)),96,72] for _ in range(2)]) for _ in range(steps)]
    protocol=dict(protocol='paired_detector_head_adaptation_v1',smoke=a.smoke,steps_per_arm=steps,arms=['original','random_crop_half'],seed=100,
        training_frames=len(train),development_frames=len(dev),training_episodes=len({x['row']['episode_id'] for x in queries['train']}),development_episodes=len({x['row']['episode_id'] for x in queries['dev']}),
        original_known_train=sum(data['y'][x['base_index']]>=0 for x in queries['train']),unknown_intent_train=sum(data['y'][x['base_index']]<0 for x in queries['train']),
        targets='All visible six-class CAD masks, including unknown-intent queries; empty visible masks retained; no intent/future labels in training',
        crop='Label-independent random96x72 viewport resized120x90 on half scheduled samples; same schedule drawn for both arms',
        optimizer=dict(name='SGD',lr=.0005,momentum=.9,weight_decay=.0001,batch_size=2,clip_grad_norm=5),
        backbone='All backbone parameters frozen; RPN and ROI box/mask heads trainable; original network checkpoint loaded strictly',
        initializer_sha256=sha(Path(cfg.detector_checkpoint)),thresholds=thresholds,postprocessing='Existing red support, NMS and class-balanced8 unchanged',
        checkpoint_steps=checkpoints,source_hashes=source_hashes,query_ids={s:[x['row']['id'] for x in part] for s,part in queries.items()},
        episodes={s:sorted({x['row']['episode_id'] for x in part}) for s,part in queries.items()},schedule=schedule,
        holdout='confirm10',holdout_manifest_sha256=sha(reserve/'manifest.json'),holdout_features_loaded=False,
        acceptance='known/acquire proposal coverage nonregression; no FP or bowl FP increase; per-class recall floor minus.02; strict gain in proposal coverage or edge-small matched instances',
        limitation='Bounded detector pilot, not full data epochs; development proxies/CAD labels not independent human truth; any detector winner needs full selector rerun and prospective confirmation',
        physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),tf32=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else x))
    initial=torch.load(cfg.detector_checkpoint,map_location='cpu',weights_only=False)['model'];initial_digest=state_digest(initial);backbone_digest=state_digest(initial,'backbone.');registry={};baseline=None
    for arm in protocol['arms']:
        torch.manual_seed(100);model=make_model();model.load_state_dict(initial,strict=True);assert state_digest(model.state_dict())==initial_digest
        for p in model.backbone.parameters():p.requires_grad_(False)
        model.roi_heads.score_thresh=min(thresholds);model.to(a.device);parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.SGD(parameters,lr=.0005,momentum=.9,weight_decay=.0001)
        if baseline is None:
            baseline=evaluate(model,dev,data,thresholds,a.device,16);(a.output/'baseline_development.json').write_text(json.dumps(baseline,indent=2))
            if baseline['weak_targets']['cached_availability_disagreements']:raise ValueError('Original detector no longer reproduces cached proposal availability; audit baseline before adaptation')
            print('BASELINE',json.dumps({k:v for k,v in baseline.items() if k not in ['records','frame_availability']}),flush=True)
        started=time.monotonic();loss_log=[];torch.cuda.reset_peak_memory_stats()
        for step,entry in enumerate(schedule,1):
            model.train();model.backbone.eval();images=[];targets=[]
            for i,crop_flag,crop in zip(entry['indices'],entry['cropped'],entry['crops']):
                image,target=train.observation(i,crop if arm=='random_crop_half' and crop_flag else None)
                images.append(image.to(a.device));targets.append({k:v.to(a.device) for k,v in target.items()})
            optimizer.zero_grad(set_to_none=True);losses=model(images,targets);loss=sum(losses.values())
            if not torch.isfinite(loss):raise ValueError('Nonfinite detector loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5.);optimizer.step();loss_log.append(dict(step=step,loss=float(loss.detach()),components={k:float(v.detach()) for k,v in losses.items()}))
            if step%25==0:print('TRAIN',arm,step,round(loss_log[-1]['loss'],5),round(time.monotonic()-started,1),flush=True)
            if step not in checkpoints:continue
            assert state_digest(model.state_dict(),'backbone.')==backbone_digest
            optimizer.zero_grad(set_to_none=True);value=evaluate(model,dev,data,thresholds,a.device,16);gate=detector_gate(value,baseline);name=f'{arm}_step{step}'
            path=a.output/(name+'.pt');torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},step=step,seed=100,arm=arm,backbone_unchanged=True,
                initializer_sha256=protocol['initializer_sha256'],protocol_sha256=sha(a.output/'protocol.json'),thresholds=thresholds,smoke=a.smoke),path)
            (a.output/(name+'_development.json')).write_text(json.dumps(value,indent=2));record=dict(name=name,checkpoint=str(path),checkpoint_sha256=sha(path),gate=gate,
                metrics={k:v for k,v in value.items() if k not in ['records','frame_availability']},seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),backbone_unchanged=True)
            registry[name]=record;(a.output/'registry.json').write_text(json.dumps(registry,indent=2));(a.output/(arm+'_training.json')).write_text(json.dumps(loss_log,indent=2))
            (a.output/'status.json').write_text(json.dumps(dict(stage='training',arm=arm,step=step,completed_checkpoints=len(registry)),indent=2));print('CHECKPOINT',name,json.dumps(record),flush=True)
        del optimizer,model;torch.cuda.empty_cache()
    candidates=[x for x in registry.values() if x['gate']['eligible']]
    winner=max(candidates,key=lambda x:(x['metrics']['weak_targets'].get('acquire_available',0),x['metrics']['weak_targets']['available'],x['metrics']['edge_small'].get('matched',0),x['metrics']['bounded8']['micro']['precision'])) if candidates and not a.smoke else None
    (a.output/'selected.json').write_text(json.dumps(dict(name=winner['name'] if winner else 'original_frozen_detector',result=winner,smoke=a.smoke,selector_evaluated=False,confirmation_evaluated=False,no_promotion=True),indent=2))
    (a.output/'status.json').write_text(json.dumps(dict(stage='smoke_complete' if a.smoke else 'complete',completed_checkpoints=len(registry),winner=winner['name'] if winner else 'original_frozen_detector'),indent=2));print('COMPLETE',arm,'winner',winner['name'] if winner else 'original_frozen_detector',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','config','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');main(p.parse_args())
