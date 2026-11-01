"""Episode-disjoint pilot. Privileged proxies supervise/scoring only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from proxy import target_candidate
from selector import TargetSelector


def main(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    manifest=json.loads((a.corpus/'manifest.json').read_text())
    truth={row['id']:row['objects'] for row in json.loads((a.corpus/'annotations_all.json').read_text())['annotations']}
    targets={row['id']:row for row in json.loads(a.targets.read_text())['records']}
    meta=json.loads(a.features.with_suffix('.json').read_text())
    expected=hashlib.sha256((a.corpus/'manifest.json').read_bytes()).hexdigest()
    assert meta['manifest_sha256']==expected
    assert json.loads(a.targets.read_text())['manifest_sha256']==expected
    assert meta['ids']==[row['id'] for row in manifest['records']]
    public={row['id']:row['objects'] for row in meta['detections']}
    arrays=np.load(a.features,allow_pickle=False)
    candidates=arrays['candidates'];contexts=arrays['context'];valid=arrays['valid'];n=len(candidates)
    y=np.full(n,-100,np.int64);positive=np.zeros((n,8),bool);visible=np.zeros(n,bool)
    splits={split:[] for split in ('train','dev','test')}
    episode_splits={}
    for i,row in enumerate(manifest['records']):
        if row['episode_id'] in episode_splits:assert episode_splits[row['episode_id']]==row['split']
        episode_splits[row['episode_id']]=row['split']
        target=targets[row['id']]['target'];objects=public[row['id']]
        assert len(objects)==int(valid[i].sum())
        if target is None:continue
        label=target_candidate(objects,truth[row['id']],target)
        y[i]=8 if label<0 else label
        for j,obj in enumerate(objects):
            positive[i,j]=target_candidate([obj],truth[row['id']],target)==0
        visible[i]=any(obj['instance_id']==target['instance_id'] for obj in truth[row['id']])
        splits[row['split']].append(i)
    if min(map(len,splits.values()))==0:raise ValueError('Proxy labels missing in a split: '+str({k:len(v) for k,v in splits.items()}))
    train=np.array(splits['train']);dev=np.array(splits['dev']);test=np.array(splits['test'])
    cmean=candidates[train][valid[train]].mean(0);cstd=candidates[train][valid[train]].std(0).clip(.05)
    xmean=contexts[train].mean(0);xstd=contexts[train].std(0).clip(.05)
    cs=torch.tensor((candidates-cmean)/cstd);xs=torch.tensor((contexts-xmean)/xstd)
    mask=torch.tensor(valid);labels=torch.tensor(y)
    model=TargetSelector(cs.shape[-1],xs.shape[-1]);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    def metrics(pred,indices):
        available=positive.any(1);pred=np.asarray(pred)
        hit=np.array([slot<8 and positive[i,slot] for i,slot in enumerate(pred)])
        index=np.asarray(indices,dtype=int)
        def rate(numerator,denominator):return float(numerator/denominator) if denominator else None
        per_episode=[]
        for episode in sorted({manifest['records'][i]['episode_id'] for i in index}):
            group=np.array([i for i in index if manifest['records'][i]['episode_id']==episode and available[i]])
            if len(group):per_episode.append(float(hit[group].mean()))
        return dict(samples=len(index),visible_targets=int(visible[index].sum()),available_targets=int(available[index].sum()),
            selected_targets=int((pred[index]<8).sum()),correct_selected=int(hit[index].sum()),
            conditional_selection_accuracy=rate(hit[index].sum(),available[index].sum()),
            visible_target_localization_recall=rate(hit[index].sum(),visible[index].sum()),
            selection_precision=rate(hit[index].sum(),(pred[index]<8).sum()),
            detector_ceiling=rate(available[index].sum(),visible[index].sum()),
            label_accuracy=float((pred[index]==y[index]).mean()),
            episode_macro_conditional_accuracy=float(np.mean(per_episode)) if per_episode else None)
    best=-1.;best_state=None;history=[]
    for epoch in range(a.epochs):
        model.train()
        for batch in np.array_split(np.random.permutation(train),max(1,(len(train)+31)//32)):
            opt.zero_grad();loss=nn.functional.cross_entropy(model(cs[batch],xs[batch],mask[batch]),labels[batch]);loss.backward();opt.step()
        model.eval()
        with torch.inference_mode():pred=model(cs,xs,mask).argmax(-1).numpy()
        value=metrics(pred,dev)
        score=value['conditional_selection_accuracy']
        if score is None:raise ValueError('No detectable proxy target in dev')
        if score>best:
            best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        history.append(dict(epoch=epoch+1,train_loss=float(loss),dev=value))
    model.load_state_dict(best_state);model.eval()
    with torch.inference_mode():
        t=time.perf_counter();prob=model(cs,xs,mask).softmax(-1).numpy();elapsed=time.perf_counter()-t
    learned=prob.argmax(-1)
    confidence=np.where(valid,candidates[:,:,132],-np.inf).argmax(-1)
    confidence[~valid.any(1)]=8
    areas=(candidates[:,:,130]-candidates[:,:,128])*(candidates[:,:,131]-candidates[:,:,129])
    largest=np.where(valid,areas,-np.inf).argmax(-1);largest[~valid.any(1)]=8
    # Task-conditioned target-class frequency is fitted on train labels only.
    tasks=meta['tasks'];prior=np.ones((len(tasks),6),np.float64)
    for i in train:
        row=manifest['records'][i];target=targets[row['id']]['target']
        prior[tasks.index(row['task']),target['class_id']]+=1
    prior/=prior.sum(-1,keepdims=True)
    task_prior=[]
    for i,row in enumerate(manifest['records']):
        scores=[prior[tasks.index(row['task']),obj['class_id']]*obj['score'] for obj in public[row['id']]]
        task_prior.append(int(np.argmax(scores)) if scores else 8)
    results={name:{split:metrics(pred,index) for split,index in splits.items()}
             for name,pred in [('learned',learned),('confidence',confidence),('largest_box',largest),('task_class_prior',task_prior)]}
    torch.save(dict(model=best_state,candidate_mean=cmean,candidate_std=cstd,context_mean=xmean,
        context_std=xstd,candidate_dim=cs.shape[-1],context_dim=xs.shape[-1],tasks=tasks,
        features_sha256=hashlib.sha256(a.features.read_bytes()).hexdigest(),seed=a.seed,
        best_epoch=best_epoch),a.output/'selector.pt')
    rows=[]
    for i,row in enumerate(manifest['records']):
        index=int(learned[i]);objs=public[row['id']]
        rows.append(dict(id=row['id'],episode_id=row['episode_id'],step=row['step'],split=row['split'],
            labelled=bool(y[i]!=-100),target=targets[row['id']]['target'],target_visible=bool(visible[i]),
            target_available=bool(positive[i].any()),selected=index if index<8 else None,
            selected_object=objs[index] if index<len(objs) else None,probabilities=prob[i].tolist(),
            correct=bool(index<8 and positive[i,index]),objects=objs))
    result=dict(protocol='episode_disjoint_motion_proximity_proxy_selector_v1',seed=a.seed,
        training_episodes=sum(split=='train' for split in episode_splits.values()),
        split_label_counts={k:len(v) for k,v in splits.items()},best_epoch=best_epoch,
        checkpoint_selection='dev conditional selection accuracy; earliest tied epoch',results=results,
        cpu_batched_scoring_ms_per_frame=elapsed/n*1000,history=history,predictions=rows,
        limitations='Proxy intent labels, sparse frames, small reused detector test split; no human intent GT, no continuous tracking or DP rollout; null means unavailable proxy target, not verified no manipulation intent')
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({key:value for key,value in result.items() if key not in ('history','predictions')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--seed',type=int,default=100);main(p.parse_args())
