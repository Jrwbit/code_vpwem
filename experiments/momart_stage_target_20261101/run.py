"""Train on old train episodes, select/calibrate on old dev, then evaluate new episodes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'momart_target_selector_20261101'))
from proxy import target_candidate
from selector import TargetSelector
from model import StagePointer,gated_logits
from phase_logic import ROLE_CLASS,ROLES


def load(corpus,features,labels,tag):
    m=json.loads((corpus/'manifest.json').read_text());meta=json.loads(features.with_suffix('.json').read_text())
    lab=json.loads(labels.read_text());digest=hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest()
    assert meta['manifest_sha256']==lab['manifest_sha256']==digest
    assert meta['ids']==[r['id'] for r in m['records']]
    labels={r['id']:r for r in lab['records']}
    gt={r['id']:r['objects'] for r in json.loads((corpus/'annotations_all.json').read_text())['annotations']}
    public={r['id']:r['objects'] for r in meta['detections']};arrays=np.load(features,allow_pickle=False)
    n=len(meta['ids']);y=np.full(n,-100,np.int64);roles=np.full(n,-100,np.int64)
    positive=np.zeros((n,8),bool);visible=np.zeros(n,bool);rows=[]
    for i,r in enumerate(m['records']):
        label=labels[r['id']]['label'];objs=public[r['id']]
        row=dict(r,tag=tag,label=label,objects=objs)
        if label is not None:
            roles[i]=label['role']
            if label['target'] is None:y[i]=8
            else:
                match=target_candidate(objs,gt[r['id']],label['target']);y[i]=match if match>=0 else 8
                for j,obj in enumerate(objs):positive[i,j]=target_candidate([obj],gt[r['id']],label['target'])==0
                visible[i]=any(o['instance_id']==label['target']['instance_id'] for o in gt[r['id']])
        rows.append(row)
    return dict(rows=rows,candidates=arrays['candidates'],context=arrays['context'],valid=arrays['valid'],
                y=y,roles=roles,positive=positive,visible=visible,tasks=meta['tasks'])


def measure(pred,indices,data):
    idx=np.asarray(indices,dtype=int);pred=np.asarray(pred);roles=data['roles'];positive=data['positive'];visible=data['visible']
    selected=pred<8;hit=np.array([slot<8 and positive[i,slot] for i,slot in enumerate(pred)])
    unsupported=roles==4;supported=(roles>=0)&(roles<4);available=positive.any(1)
    wrong_role=np.zeros(len(pred),bool);bowl_wrong=np.zeros(len(pred),bool)
    for i in idx:
        if selected[i]:
            cls=data['rows'][i]['objects'][pred[i]]['class_id']
            wrong_role[i]=ROLE_CLASS[roles[i]]!=cls
            bowl_wrong[i]=roles[i]==0 and cls!=0
    correct=hit | (unsupported & ~selected)
    macro=[]
    for role in range(5):
        group=idx[roles[idx]==role]
        if len(group):macro.append(float(correct[group].mean()))
    def ratio(num,den):return float(num/den) if den else None
    return dict(samples=len(idx),supported_visible=int(visible[idx].sum()),available_targets=int(available[idx].sum()),
        selected=int(selected[idx].sum()),correct_box=int(hit[idx].sum()),
        selection_precision=ratio(hit[idx].sum(),selected[idx].sum()),
        visible_recall=ratio(hit[idx].sum(),visible[idx].sum()),
        supported_coverage=ratio((selected&visible)[idx].sum(),visible[idx].sum()),
        unsupported_region_frames=int(unsupported[idx].sum()),
        emitted_box_on_unsupported_region=int((unsupported&selected)[idx].sum()),
        wrong_role_boxes=int(wrong_role[idx].sum()),acquire_frames=int((roles[idx]==0).sum()),
        non_bowl_box_on_acquire=int(bowl_wrong[idx].sum()),
        macro_decision_accuracy=float(np.mean(macro)) if macro else None)


def calibrate(prob,index,data,confidence=None):
    raw=prob.argmax(-1);confidence=prob.max(-1) if confidence is None else confidence;choices=[]
    for threshold in [0.,.5,.6,.7,.8,.9,.95,.99]:
        pred=np.where(confidence>=threshold,raw,8)
        metrics=measure(pred,index,data)
        if metrics['selected']>=10:choices.append((threshold,metrics))
    feasible=[item for item in choices if item[1]['selection_precision'] is not None and item[1]['selection_precision']>=.9]
    if feasible:
        chosen=max(feasible,key=lambda item:(item[1]['selected'],item[1]['selection_precision'],-item[0]))
    elif choices:
        chosen=max(choices,key=lambda item:(item[1]['selection_precision'] or 0,item[1]['selected'],-item[0]))
    else:chosen=(0.,measure(raw,index,data))
    return dict(threshold=chosen[0],dev=chosen[1],dev_precision_90_feasible=bool(feasible))


def main(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    a.output.mkdir(parents=True,exist_ok=False)
    old=load(a.old_corpus,a.old_features,a.old_labels,'old')
    new=load(a.new_corpus,a.new_features,a.new_labels,'new')
    assert old['tasks']==new['tasks']
    assert not {r['episode_id'] for r in old['rows']} & {r['episode_id'] for r in new['rows']}
    data={key:np.concatenate([old[key],new[key]]) for key in ('candidates','context','valid','y','roles','positive','visible')}
    data['rows']=old['rows']+new['rows'];raw=data['candidates'];valid=data['valid']
    pool=(raw*valid[:,:,None]).sum(1)/valid.sum(1,keepdims=True).clip(1)
    context=np.c_[data['context'],pool].astype(np.float32)
    splits={name:np.array([i for i,r in enumerate(data['rows']) if data['roles'][i]>=0 and
                 ((r['tag']=='old' and r['split']==name) if name!='new_test' else r['tag']=='new')])
            for name in ('train','dev','test','new_test')}
    assert min(map(len,splits.values()))>0
    train=splits['train'];dev=splits['dev']
    cm=raw[train][valid[train]].mean(0);cs=raw[train][valid[train]].std(0).clip(.05)
    xm=context[train].mean(0);xs=context[train].std(0).clip(.05)
    candidates=torch.tensor((raw-cm)/cs);contexts=torch.tensor((context-xm)/xs);mask=torch.tensor(valid)
    classes=torch.tensor(raw[:,:,133:].argmax(-1));labels=torch.tensor(data['y']);roles=torch.tensor(data['roles'])
    net=StagePointer(candidates.shape[-1],contexts.shape[-1]);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    best={name:dict(score=-1) for name in ('flat','stage')};history=[]
    for epoch in range(a.epochs):
        net.train()
        for batch in np.array_split(np.random.permutation(train),max(1,(len(train)+31)//32)):
            optimizer.zero_grad();pointer,role=net(candidates[batch],contexts[batch],mask[batch])
            loss=nn.functional.cross_entropy(pointer,labels[batch])+.5*nn.functional.cross_entropy(role,roles[batch])
            loss.backward();optimizer.step()
        net.eval()
        with torch.inference_mode():
            pointer,role=net(candidates,contexts,mask)
            outputs={'flat':pointer,'stage':gated_logits(pointer,role,classes,mask)}
        log=dict(epoch=epoch+1)
        for name,logits in outputs.items():
            metric=measure(logits.argmax(-1).numpy(),dev,data);score=metric['macro_decision_accuracy']
            if score>best[name]['score']:best[name]=dict(score=score,state=copy.deepcopy(net.state_dict()),epoch=epoch+1)
            log[name]=metric
        history.append(log)
    results={};predictions={};calibration={};stage_predictions=None
    for name in ('flat','stage'):
        net.load_state_dict(best[name]['state']);net.eval()
        with torch.inference_mode():
            pointer,role=net(candidates,contexts,mask)
            logits=pointer if name=='flat' else gated_logits(pointer,role,classes,mask)
            prob=logits.softmax(-1).numpy()
        confidence=prob.max(-1)
        if name=='stage':confidence=confidence*role.softmax(-1).max(-1).values.numpy()
        cal=calibrate(prob,dev,data,confidence);calibration[name]=cal
        for suffix,pred in [('',prob.argmax(-1)),('_reject',np.where(confidence>=cal['threshold'],prob.argmax(-1),8))]:
            method=name+suffix;predictions[method]=pred
            results[method]={split:measure(pred,index,data) for split,index in splits.items()}
        if name=='stage':
            stage_predictions=role.argmax(-1).numpy()
            role_accuracy={split:float((stage_predictions[index]==data['roles'][index]).mean()) for split,index in splits.items()}
        torch.save(dict(model=best[name]['state'],candidate_mean=cm,candidate_std=cs,context_mean=xm,context_std=xs,
            candidate_dim=candidates.shape[-1],context_dim=contexts.shape[-1],tasks=old['tasks'],calibration=cal,
            seed=a.seed,best_epoch=best[name]['epoch']),a.output/(name+'.pt'))
    # Previous model uses its unchanged old input/normalization, not stage labels.
    ckpt=torch.load(a.previous_model,map_location='cpu',weights_only=False)
    previous=TargetSelector(ckpt['candidate_dim'],ckpt['context_dim']).eval();previous.load_state_dict(ckpt['model'])
    with torch.inference_mode():
        pred=previous(torch.tensor((raw-ckpt['candidate_mean'])/ckpt['candidate_std']),
                      torch.tensor((data['context']-ckpt['context_mean'])/ckpt['context_std']),mask).argmax(-1).numpy()
    predictions['previous']=pred;results['previous']={split:measure(pred,index,data) for split,index in splits.items()}
    rows=[]
    for i,row in enumerate(data['rows']):
        rows.append(dict(id=row['id'],episode_id=row['episode_id'],step=row['step'],tag=row['tag'],split=row['split'],
            label=row['label'],predicted_role=int(stage_predictions[i]),objects=row['objects'],
            predictions={name:int(pred[i]) for name,pred in predictions.items()}))
    result=dict(protocol='stage_role_reject_pilot_v1',seed=a.seed,roles=ROLES,
        split_counts={key:len(value) for key,value in splits.items()},
        best_epochs={key:value['epoch'] for key,value in best.items()},role_accuracy=role_accuracy,
        calibration=calibration,results=results,predictions=rows,history=history,
        new_test_episodes=sorted({r['episode_id'] for r in new['rows']}),
        note='Geometry/future-derived supervision is scoring only. Carry/place rejects because recipient region is unlabelled; not a learned complete placement solution.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('predictions','history')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('old-corpus','old-features','old-labels','new-corpus','new-features','new-labels','previous-model','output'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--seed',type=int,default=100);p.add_argument('--epochs',type=int,default=200)
    main(p.parse_args())
