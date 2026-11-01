"""Common subject labels, public feature views and non-leaking metrics for all methods."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'momart_target_selector_20261101'))
from proxy import target_candidate
from subject_labels import subject_label


def load_group(corpus,features,phases,tag,extra=None):
    corpus=Path(corpus);features=Path(features);phases=Path(phases)
    manifest=json.loads((corpus/'manifest.json').read_text());meta=json.loads(features.with_suffix('.json').read_text())
    source=json.loads(phases.read_text());sha=hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest()
    assert source['manifest_sha256']==meta['manifest_sha256']==sha
    assert meta['ids']==[r['id'] for r in manifest['records']]
    refs={r['id']:r['objects'] for r in json.loads((corpus/'annotations_all.json').read_text())['annotations']}
    source={r['id']:r for r in source['records']};detections={r['id']:r['objects'] for r in meta['detections']}
    arrays=np.load(features,allow_pickle=False);n=len(meta['ids'])
    labels=np.full(n,-100,np.int64);roles=np.full(n,-100,np.int64);positive=np.zeros((n,8),bool)
    visible=np.zeros(n,bool);weights=np.zeros(n,np.float32);rows=[]
    for i,row in enumerate(manifest['records']):
        label=subject_label(source[row['id']]);objects=detections[row['id']]
        if label is not None:
            role=label['role'];target=label['target'];roles[i]=role;weights[i]=label['weight']
            assert target is not None,'Unlabelled recipient must not become a negative subject'
            match=target_candidate(objects,refs[row['id']],target);labels[i]=match if match>=0 else 8
            for slot,obj in enumerate(objects):positive[i,slot]=target_candidate([obj],refs[row['id']],target)==0
            visible[i]=any(obj['instance_id']==target['instance_id'] for obj in refs[row['id']])
        rows.append(dict(row,tag=tag,label=label,objects=objects))
    result=dict(rows=rows,candidates=arrays['candidates'],context=arrays['context'],valid=arrays['valid'],
        y=labels,roles=roles,positive=positive,visible=visible,weights=weights,tasks=meta['tasks'])
    if extra is not None:
        extra=Path(extra);metadata=json.loads(extra.with_suffix('.json').read_text());enriched=np.load(extra,allow_pickle=False)
        assert metadata['ids']==meta['ids'] and metadata['manifest_sha256']==sha
        result['extra_candidates']=enriched['candidates'];result['extra_context']=enriched['context']
    return result


def combine(groups):
    tasks=groups[0]['tasks'];assert all(g['tasks']==tasks for g in groups)
    keys=[key for key in groups[0] if key not in ('rows','tasks')]
    data={key:np.concatenate([g[key] for g in groups]) for key in keys};data['tasks']=tasks
    data['rows']=[r for group in groups for r in group['rows']]
    seen={}
    for r in data['rows']:
        key=(r['tag'],r['split'])
        if r['episode_id'] in seen:assert seen[r['episode_id']]==key,'Episode crosses partitions'
        seen[r['episode_id']]=key
    return data


def feature_view(data,rich=False):
    candidates=data['candidates'];context=data['context']
    if rich:
        candidates=np.concatenate([candidates,data['extra_candidates']],-1)
        context=np.concatenate([context,data['extra_context']],-1)
    mask=data['valid'];pool=(candidates*mask[:,:,None]).sum(1)/mask.sum(1,keepdims=True).clip(1)
    return candidates.astype(np.float32),np.c_[context,pool].astype(np.float32)


def partitions(data):
    return {name:np.array([i for i,r in enumerate(data['rows']) if data['y'][i]>=0 and
       ((r['tag']=='old' and r['split']==name) if name in ('train','dev','test') else r['tag']==name)],dtype=int)
       for name in ('train','dev','test','regression','confirm')}


def metrics(pred,index,data):
    pred=np.asarray(pred);index=np.asarray(index,dtype=int);selected=pred<8
    hit=np.array([slot<8 and data['positive'][i,slot] for i,slot in enumerate(pred)])
    known=data['y']>=0;visible=data['visible'];available=data['positive'].any(-1)
    wrong_acquire=0;wrong_carry=0;wrong_class=0
    for i in index:
        if selected[i]:
            cls=data['rows'][i]['objects'][pred[i]]['class_id'];gold=data['rows'][i]['label']['target']['class_id']
            wrong_class+=cls!=gold;wrong_acquire+=data['roles'][i]==0 and cls!=0;wrong_carry+=data['roles'][i]==4 and cls!=0
    def rate(num,den):return float(num/den) if den else None
    precision=rate(hit[index].sum(),selected[index].sum());recall=rate(hit[index].sum(),visible[index].sum())
    per_episode=[]
    for episode in sorted({data['rows'][i]['episode_id'] for i in index}):
        group=np.array([i for i in index if data['rows'][i]['episode_id']==episode],int)
        den=visible[group].sum()
        if den:per_episode.append(float(hit[group].sum()/den))
    return dict(samples=len(index),visible=int(visible[index].sum()),available=int(available[index].sum()),
        selected=int(selected[index].sum()),correct=int(hit[index].sum()),precision=precision,recall=recall,
        f1=2*(precision or 0)*(recall or 0)/max((precision or 0)+(recall or 0),1e-8),
        coverage=rate((selected&visible)[index].sum(),visible[index].sum()),
        detector_ceiling=rate(available[index].sum(),visible[index].sum()),wrong_class=int(wrong_class),
        acquire_samples=int((data['roles'][index]==0).sum()),wrong_acquire=int(wrong_acquire),
        carry_samples=int((data['roles'][index]==4).sum()),wrong_carry=int(wrong_carry),
        episode_macro_recall=float(np.mean(per_episode)) if per_episode else None,
        unavailable_null_correct=int(((~available)&(~selected)&known)[index].sum()))


def calibrate(prob,index,data,minimum_selected=10):
    confidence=prob.max(-1);raw=prob.argmax(-1);points=[]
    for threshold in [0.,.25,.4,.5,.6,.7,.8,.85,.9,.95,.98]:
        prediction=np.where(confidence>=threshold,raw,8);value=metrics(prediction,index,data)
        if value['selected']>=minimum_selected:points.append(dict(threshold=threshold,metrics=value))
    feasible=[p for p in points if (p['metrics']['precision'] or 0)>=.9]
    if feasible:chosen=max(feasible,key=lambda p:(p['metrics']['recall'] or 0,p['metrics']['selected'],-p['threshold']))
    elif points:chosen=max(points,key=lambda p:(p['metrics']['f1'],p['metrics']['recall'] or 0,-p['threshold']))
    else:chosen=dict(threshold=0.,metrics=metrics(raw,index,data))
    return dict(chosen,precision_90_feasible=bool(feasible),selection_partition='old/dev only')
