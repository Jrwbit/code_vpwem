"""Dev-selected method search on a fixed subject protocol; confirmation remains unread."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from data import load_group,combine,feature_view,partitions,metrics,calibrate
from models import Pointer,RecurrentPointer,weighted_loss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'momart_target_selector_20261101'))
from selector import TargetSelector


def load_data(root):
    research=root/'momart_target_research_20261101';previous=root/'momart_target_selector_20261101';stage=root/'momart_stage_target_20261101'
    old=load_group(root/'momart_detector_drawer_augmented_20261030',previous/'features.npz',stage/'old_phases.json','old',research/'extra_old.npz')
    regression=load_group(stage/'new_corpus',stage/'new_features.npz',stage/'new_phases.json','regression',research/'extra_new.npz')
    return combine([old,regression])


def make_views(data,index):
    views={}
    for name,rich in [('base',False),('rich',True)]:
        raw,context=feature_view(data,rich);valid=data['valid']
        cm=raw[index][valid[index]].mean(0);cs=raw[index][valid[index]].std(0).clip(.05)
        xm=context[index].mean(0);xs=context[index].std(0).clip(.05)
        views[name]=dict(raw=raw,context=context,candidates=torch.tensor((raw-cm)/cs),contexts=torch.tensor((context-xm)/xs),
                         cm=cm,cs=cs,xm=xm,xs=xs)
    return views


def sequence_batch(data,view,episode_ids=None):
    ids=episode_ids or sorted({r['episode_id'] for r in data['rows']})
    groups=[sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==ep],key=lambda i:data['rows'][i]['step']) for ep in ids]
    length=max(map(len,groups));count=len(groups);c=view['candidates'];x=view['contexts']
    cb=torch.zeros(count,length,8,c.shape[-1]);xb=torch.zeros(count,length,x.shape[-1]);mask=torch.zeros(count,length,8,dtype=torch.bool)
    gap=torch.zeros(count,length);labels=torch.full((count,length),-100,dtype=torch.long);weights=torch.zeros(count,length)
    for row,index in enumerate(groups):
        n=len(index);cb[row,:n]=c[index];xb[row,:n]=x[index];mask[row,:n]=torch.tensor(data['valid'][index])
        steps=np.array([data['rows'][i]['step'] for i in index]);assert np.all(np.diff(steps)>0)
        gap[row,:n]=torch.tensor(np.log1p(np.r_[0,np.diff(steps)]*.05),dtype=torch.float32)
        labels[row,:n]=torch.tensor(data['y'][index]);weights[row,:n]=torch.tensor(data['weights'][index])
    return dict(c=cb,x=xb,mask=mask,gaps=gap,labels=labels,weights=weights,groups=groups)


def train_torch(spec,data,view,splits,seed,epochs):
    torch.manual_seed(seed);np.random.seed(seed);kind=spec['kind'];recurrent=kind=='gru'
    candidates=view['candidates'];context=view['contexts'];valid=torch.tensor(data['valid']);classes=torch.tensor(data['candidates'][:,:,133:].argmax(-1))
    labels=torch.tensor(data['y']);roles=torch.tensor(data['roles']);weights=torch.tensor(data['weights'] if spec.get('weighted') else (data['y']>=0).astype(np.float32))
    if recurrent:
        net=RecurrentPointer(candidates.shape[-1],context.shape[-1],spec['width'])
        train_ids=sorted({data['rows'][i]['episode_id'] for i in splits['train']});train_seq=sequence_batch(data,view,train_ids);all_seq=sequence_batch(data,view)
    else:net=Pointer(candidates.shape[-1],context.shape[-1],spec['width'],spec.get('dropout',.1),kind,spec.get('soft_role',0.))
    opt=torch.optim.AdamW(net.parameters(),lr=spec.get('lr',.001),weight_decay=spec.get('decay',.001))
    best=-1.;state=None;best_epoch=0;history=[];start=time.monotonic()
    def predict():
        net.eval()
        with torch.inference_mode():
            if recurrent:
                probability=np.zeros((len(labels),9),np.float32)
                result=net(all_seq['c'],all_seq['x'],all_seq['mask'],all_seq['gaps']).softmax(-1).numpy()
                for row,index in enumerate(all_seq['groups']):probability[index]=result[row,:len(index)]
                return probability
            return net(candidates,context,valid,classes)[0].softmax(-1).numpy()
    for epoch in range(epochs):
        net.train()
        if recurrent:
            opt.zero_grad();logits=net(train_seq['c'],train_seq['x'],train_seq['mask'],train_seq['gaps'])
            loss=weighted_loss(logits,train_seq['labels'],train_seq['weights'],train_seq['mask'],spec.get('smoothing',0.));loss.backward();opt.step()
        else:
            for batch in np.array_split(np.random.permutation(splits['train']),max(1,(len(splits['train'])+31)//32)):
                opt.zero_grad();logits,role=net(candidates[batch],context[batch],valid[batch],classes[batch])
                loss=weighted_loss(logits,labels[batch],weights[batch],valid[batch],spec.get('smoothing',0.))
                if role is not None:loss=loss+.2*nn.functional.cross_entropy(role,roles[batch])
                loss.backward();opt.step()
        probability=predict();dev=metrics(probability.argmax(-1),splits['dev'],data)
        history.append(dict(epoch=epoch+1,loss=float(loss),dev=dev))
        if dev['f1']>best:
            best=dev['f1'];state=copy.deepcopy(net.state_dict());best_epoch=epoch+1
        if epoch+1-best_epoch>=30:break
    net.load_state_dict(state);probability=predict()
    checkpoint=dict(state=state,spec=spec,seed=seed,best_epoch=best_epoch,candidate_dim=candidates.shape[-1],context_dim=context.shape[-1],
                    normalizer={name:view[name] for name in ('cm','cs','xm','xs')})
    return probability,checkpoint,dict(best_epoch=best_epoch,epochs=len(history),seconds=time.monotonic()-start,history=history)


def candidate_rows(data,view,index=None):
    xs=[];ys=[];weights=[];positions=[]
    index=range(len(data['rows'])) if index is None else index
    for i in index:
        valid=np.r_[np.flatnonzero(data['valid'][i]),8];count=len(valid)
        for slot in valid:
            candidate=view['raw'][i,slot] if slot<8 else np.zeros(view['raw'].shape[-1],np.float32)
            xs.append(np.r_[candidate,view['context'][i],slot==8])
            ys.append(int(data['positive'][i,slot]) if slot<8 else int(data['y'][i]==8))
            weights.append(data['weights'][i]/count);positions.append((i,int(slot)))
    return np.asarray(xs,np.float32),np.asarray(ys),np.asarray(weights),positions


def train_tree(spec,data,view,splits,seed):
    from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
    import sklearn
    options=dict(random_state=seed)
    if spec['kind']=='extra':model=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=spec['leaf'],max_features=.5,n_jobs=2,class_weight='balanced',**options)
    elif spec['kind']=='forest':model=RandomForestClassifier(n_estimators=256,max_depth=12,min_samples_leaf=spec['leaf'],n_jobs=2,class_weight='balanced',**options)
    else:model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=spec['leaves'],min_samples_leaf=10,l2_regularization=1.,early_stopping=False,max_bins=63,**options)
    train_x,train_y,train_w,_=candidate_rows(data,view,splits['train']);start=time.monotonic()
    model.fit(train_x,train_y,sample_weight=train_w)
    x,_,_,positions=candidate_rows(data,view);scores=model.predict_proba(x)[:,list(model.classes_).index(1)]
    prob=np.zeros((len(data['rows']),9),np.float32)
    for (i,slot),score in zip(positions,scores):prob[i,slot]=score
    prob/=prob.sum(-1,keepdims=True).clip(1e-8)
    return prob,dict(model=model,spec=spec,seed=seed,sklearn_version=sklearn.__version__),dict(seconds=time.monotonic()-start,rows=len(train_x))


def subject_memory(prob,data,strength):
    """Causal preference for a previously selected bowl while grasp heuristic remains positive."""
    result=prob.copy()
    for episode in sorted({r['episode_id'] for r in data['rows']}):
        indices=sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==episode],key=lambda i:data['rows'][i]['step'])
        last=None;last_step=None;previous_grasp=False
        for i in indices:
            row=data['rows'][i];grasp=data['context'][i,258]>0  # context: main current128,past128,proprio head2,grasp1
            if last is not None and grasp and previous_grasp and row['step']-last_step<=16:
                similarity=[]
                for slot,obj in enumerate(row['objects']):
                    if obj['class_id']!=0:continue
                    f=data['candidates'][i,slot,:128];old=last['feature']
                    cosine=np.dot(f,old)/max(np.linalg.norm(f)*np.linalg.norm(old),1e-8)
                    a=np.asarray(obj['box']);b=np.asarray(last['box']);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
                    overlap=inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter,1e-8)
                    if cosine>=.9 or overlap>=.3:similarity.append((cosine+overlap,slot))
                if similarity:
                    slot=max(similarity)[1];result[i,slot]*=np.exp(strength);result[i]/=result[i].sum()
            selected=int(result[i].argmax())
            if selected<len(row['objects']) and row['objects'][selected]['class_id']==0:
                last=dict(box=row['objects'][selected]['box'],feature=data['candidates'][i,selected,:128].copy());last_step=row['step']
            elif not grasp:last=None
            previous_grasp=grasp
    return result


def write_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def specifications():
    result=[]
    for view in ('base','rich'):
        result.append(dict(name=f'{view}_linear',view=view,kind='linear',width=32))
        for width in (64,128,256):result.append(dict(name=f'{view}_mlp{width}',view=view,kind='mlp',width=width))
        result.append(dict(name=f'{view}_weighted_smooth',view=view,kind='mlp',width=128,weighted=True,smoothing=.05,dropout=.3))
        for width in (64,128):result.append(dict(name=f'{view}_query{width}',view=view,kind='query',width=width))
        for strength in (.25,1.):result.append(dict(name=f'{view}_soft{strength}',view=view,kind='mlp',width=128,soft_role=strength,weighted=True))
        result.append(dict(name=f'{view}_gru64',view=view,kind='gru',width=64,weighted=True))
        for leaf in (2,5):result.append(dict(name=f'{view}_extra{leaf}',view=view,kind='extra',leaf=leaf))
        result.append(dict(name=f'{view}_forest',view=view,kind='forest',leaf=3))
        for leaves in (7,15):result.append(dict(name=f'{view}_boost{leaves}',view=view,kind='boost',leaves=leaves))
    return result


def main(a):
    torch.set_num_threads(2);data=load_data(a.root);splits=partitions(data)
    assert min(len(splits[x]) for x in ('train','dev','test','regression'))>0
    a.output.mkdir(parents=True,exist_ok=False);views=make_views(data,splits['train']);specs=specifications()
    write_json(a.output/'data_audit.json',dict(protocol='active_subject_v1',counts={k:len(v) for k,v in splits.items()},
        roles={split:{str(role):int((data['roles'][index]==role).sum()) for role in range(5)} for split,index in splits.items()},
        unknown=len(data['rows'])-int((data['y']>=0).sum()),confirmation_not_loaded=True,
        rows=[dict(id=r['id'],episode_id=r['episode_id'],step=r['step'],tag=r['tag'],split=r['split'],label=r['label']) for r in data['rows']]))
    write_json(a.output/'specifications.json',specs)
    registry={};probabilities={}
    def record(name,prob,info=None,checkpoint=None):
        assert np.isfinite(prob).all() and np.allclose(prob.sum(-1),1.,atol=1e-4)
        raw=prob.argmax(-1);cal=calibrate(prob,splits['dev'],data)
        reject=np.where(prob.max(-1)>=cal['threshold'],raw,8)
        result=dict(name=name,info=info or {},calibration=cal,
            raw={key:metrics(raw,index,data) for key,index in splits.items() if len(index)},
            reject={key:metrics(reject,index,data) for key,index in splits.items() if len(index)})
        registry[name]=result;probabilities[name]=prob
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob)
        if checkpoint is not None:
            checkpoint['calibration']=cal;checkpoint['subject_protocol']='active_subject_v1'
            torch.save(checkpoint,a.output/(name+'.pt'))
        write_json(a.output/'registry.json',registry)
        write_json(a.output/'status.json',dict(stage='searching',completed=len(registry),latest=name,updated_unix=time.time()))
        print(json.dumps(dict(name=name,dev=result['raw']['dev'],regression=result['raw']['regression'])),flush=True)
    previous_path=a.root/'momart_target_selector_20261101/pilot_seed100/selector.pt'
    previous_ckpt=torch.load(previous_path,map_location='cpu',weights_only=False)
    previous=TargetSelector(previous_ckpt['candidate_dim'],previous_ckpt['context_dim']).eval();previous.load_state_dict(previous_ckpt['model'])
    with torch.inference_mode():prob=previous(torch.tensor((data['candidates']-previous_ckpt['candidate_mean'])/previous_ckpt['candidate_std']),
        torch.tensor((data['context']-previous_ckpt['context_mean'])/previous_ckpt['context_std']),torch.tensor(data['valid'])).softmax(-1).numpy()
    record('previous_pointer',prob,dict(frozen_checkpoint=str(previous_path)))
    for kind in ('confidence','task_prior'):
        counts=np.ones((len(data['tasks']),6))
        for i in splits['train']:
            row=data['rows'][i];counts[data['tasks'].index(row['task']),row['label']['target']['class_id']]+=1
        counts/=counts.sum(-1,keepdims=True);prob=np.zeros((len(data['rows']),9),np.float32)
        for i,row in enumerate(data['rows']):
            for slot,obj in enumerate(row['objects']):prob[i,slot]=obj['score']*(counts[data['tasks'].index(row['task']),obj['class_id']] if kind=='task_prior' else 1.)
            if not row['objects']:prob[i,8]=1.
        prob/=prob.sum(-1,keepdims=True).clip(1e-8);record(kind,prob)
    for spec in specs:
        write_json(a.output/'status.json',dict(stage='training',method=spec['name'],completed=len(registry),updated_unix=time.time()))
        if spec['kind'] in ('extra','forest','boost'):
            prob,checkpoint,info=train_tree(spec,data,views[spec['view']],splits,a.seed)
        else:prob,checkpoint,info=train_torch(spec,data,views[spec['view']],splits,a.seed,a.epochs)
        record(spec['name'],prob,info,checkpoint)
    # Ensembles and temporal memory candidates are ranked/fitted using dev only.
    ranked=sorted(registry,key=lambda name:registry[name]['raw']['dev']['f1'],reverse=True)
    for count in (3,5):
        names=ranked[:count];record('ensemble'+str(count),np.mean([probabilities[name] for name in names],0),dict(members=names,selection='old/dev F1'))
    for name in ranked[:3]:
        for strength in (.5,1.,2.):record(name+'_memory'+str(strength),subject_memory(probabilities[name],data,strength),dict(parent=name,strength=strength))
    baseline_recall=registry['previous_pointer']['raw']['dev']['recall'] or 0.
    def merit(name):
        row=registry[name];value=row['calibration']['metrics'];joint=row['calibration']['precision_90_feasible'] and (value['recall'] or 0)>=baseline_recall
        return (joint,value['recall'] if joint else row['raw']['dev']['f1'],-row['raw']['dev']['wrong_acquire'])
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,criterion='dev: precision90 with baseline recall, else raw F1; no regression/confirmation selection',
        baseline_dev_recall=baseline_recall,development_joint_goal_met=bool(merit(winner)[0]),result=registry[winner],
        confirmation_evaluated=False,method_count=len(registry)))
    write_json(a.output/'status.json',dict(stage='screening_complete',completed=len(registry),winner=winner,confirmation_evaluated=False,updated_unix=time.time()))
    print(json.dumps(dict(winner=winner,methods=len(registry),dev_goal_met=bool(merit(winner)[0]))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=100);p.add_argument('--epochs',type=int,default=150)
    main(p.parse_args())
