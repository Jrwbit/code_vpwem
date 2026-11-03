"""Paired train-size comparison with separate old/new development gates."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from availability_features import replace_availability
from benchmark import write_json
from confirm import predict_component,apply_calibration,evaluation_gate
from data import feature_view,metrics
from expanded import calibrate_v4,baseline_for
from growth_data import load_growth_data,growth_partitions
from models import Pointer,weighted_loss,hard_negative_ranking_loss
from refine import role_breakdown
from sequence_metrics import sequence_metrics
from training_consistency import class_consistency_loss


def train_model(data,view,train,dev,seed,device,ranking_strength=0.,consistency_strength=0.,consistency_mode='temporal',consistency_predecessor=None,epoch_callback=None):
    torch.manual_seed(seed);np.random.seed(seed);raw,context=view;valid=data['valid']
    cm=raw[train][valid[train]].mean(0);cs=raw[train][valid[train]].std(0).clip(.05)
    xm=context[train].mean(0);xs=context[train].std(0).clip(.05)
    c=torch.tensor((raw-cm)/cs,device=device);x=torch.tensor((context-xm)/xs,device=device)
    mask=torch.tensor(valid,device=device);labels=torch.tensor(data['y'],device=device);weights=torch.tensor(data['weights'],device=device)
    positive=torch.tensor(data['positive'],device=device) if ranking_strength else None
    classes=torch.tensor(data['candidates'][:,:,133:].argmax(-1),device=device) if consistency_strength else None
    net=Pointer(c.shape[-1],x.shape[-1],width=64,kind='mlp').to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    best=-1.;best_epoch=0;state=None;history=[];started=time.monotonic()
    for epoch in range(150):
        net.train()
        for batch in np.array_split(np.random.permutation(train),max(1,(len(train)+31)//32)):
            opt.zero_grad();logits,_=net(c[batch],x[batch],mask[batch])
            loss=weighted_loss(logits,labels[batch],weights[batch],mask[batch])
            if ranking_strength:loss=loss+ranking_strength*hard_negative_ranking_loss(logits,labels[batch],positive[batch],weights[batch],mask[batch])
            if consistency_strength:
                linked=np.flatnonzero(consistency_predecessor[batch]>=0)
                if len(linked):
                    current=batch[linked];past=consistency_predecessor[current]
                    other=past if consistency_mode=='temporal' else current
                    other_logits=net(c[other],x[other],mask[other])[0]
                    pair_weight=torch.minimum(weights[current],weights[past])
                    loss=loss+consistency_strength*class_consistency_loss(logits[linked],other_logits,classes[current],classes[other],pair_weight)
            loss.backward();opt.step()
        net.eval()
        with torch.inference_mode():prob=net(c,x,mask)[0].softmax(-1).cpu().numpy()
        value=metrics(prob.argmax(-1),dev,data)
        history.append(dict(epoch=epoch+1,loss=float(loss.detach()),dev=value))
        if epoch_callback is not None:epoch_callback(epoch+1,net,prob,dict(cm=cm,cs=cs,xm=xm,xs=xs))
        if value['f1']>best:
            best=value['f1'];best_epoch=epoch+1;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch+1-best_epoch>=30:break
    net.load_state_dict(state);net.eval()
    with torch.inference_mode():prob=net(c,x,mask)[0].softmax(-1).cpu().numpy()
    return prob,dict(state=state,seed=seed,best_epoch=best_epoch,normalizer=dict(cm=cm,cs=cs,xm=xm,xs=xs)),dict(history=history,seconds=time.monotonic()-started)


def main(a):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=a.root/'momart_target_research_20261101';a.output.mkdir(parents=True,exist_ok=False)
    transition=a.weighting=='transition'
    cases=[('uniform',1.,1.),('transition2',2.,1.),('transition4',4.,1.),('acquire2',1.,2.),('combined2',2.,2.)] if transition else [('uniform',1.,1.)]
    populations=[('enlarged108','train')] if transition else [('old28','old_train'),('enlarged108','train')]
    views=['geometry'] if transition else ['base','geometry'];fits=len(cases)*len(populations)*len(views)*3
    write_json(a.output/'frozen_search_protocol.json',dict(method='train-only transition/acquisition loss emphasis' if transition else 'paired training-set growth',train_populations=[p[0] for p in populations],weighting_cases=cases,
        development='old8 + new16; each subgroup must independently pass original V4 in addition to combined24',
        views=views,kind='mlp64',seeds=[100,200,300],fit_count=fits,bias_controls=[0.,4.],frozen_current_weights_control=True,
        acceptance_revision=4,confirmation5_loaded=False,device=a.device,physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),tf32=False,
        manifest_sha256=hashlib.sha256((r/'growth_corpus/manifest.json').read_bytes()).hexdigest(),
        confirm5_manifest_sha256=hashlib.sha256((r/'confirm5_corpus/manifest.json').read_bytes()).hexdigest(),
        reserved_confirmation='confirm6' if transition else 'confirm5',
        reserved_manifest_sha256=hashlib.sha256((r/(('confirm6' if transition else 'confirm5')+'_corpus')/'manifest.json').read_bytes()).hexdigest(),
        selection='Combined V4 and old/new subgroup V4 feasibility, then maximum correct-box recall (else F1), then fewer raw acquisition wrong-class errors',
        caveat='New development source-train episodes are selector-unseen, not full upstream perception/policy holdout. Weak labels unchanged. Do not mix registry with old8-dev screens.'))
    data=load_growth_data(a.root);splits=growth_partitions(data);dev=splits['dev'];registry={}
    if transition:
        reserved={ep['episode_id'] for ep in json.loads((r/'confirm6_corpus/manifest.json').read_text())['episodes']}
        if reserved&{row['episode_id'] for row in data['rows']}:raise ValueError('Reserved confirmation6 overlaps examined development/training')
    raw_previous=predict_component(dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100')),data,a.root).argmax(-1)
    def summarize(pred,index):return dict(raw=metrics(pred,index,data),accepted=metrics(pred,index,data),roles=role_breakdown(pred,index,data),sequence=sequence_metrics(pred,data,index))
    baseline=baseline_for(np.eye(9,dtype=np.float32)[raw_previous],dev,data)
    subbaseline={key:summarize(raw_previous,splits[key]) for key in ['old_dev','new_dev']}
    write_json(a.output/'data_audit.json',dict(counts={k:len(v) for k,v in splits.items()},
        episodes={key:sorted({data['rows'][i]['episode_id'] for i in indices}) for key,indices in splits.items()},
        baseline={key:summarize(raw_previous,splits[key]) for key in ['dev','old_dev','new_dev']},labels='active_subject_v1, no frame exclusion or target relabelling'))
    def record(name,prob,component):
        assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
        cal=calibrate_v4(prob,dev,data,baseline);pred=apply_calibration(prob,cal)
        sub={key:summarize(pred,splits[key]) for key in subbaseline}
        gates={key:evaluation_gate(sub[key],subbaseline[key],4) for key in sub}
        feasible=cal['joint_goal_met'] and all(v['joint_goal_met'] for v in gates.values())
        registry[name]=dict(name=name,component=component,calibration=cal,joint_growth_goal_met=bool(feasible),subgroup_gates=gates,subgroups=sub,
            raw={key:metrics(prob.argmax(-1),idx,data) for key,idx in splits.items() if len(idx)},
            reject={key:metrics(pred,idx,data) for key,idx in splits.items() if len(idx)})
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob);write_json(a.output/'registry.json',registry)
        write_json(a.output/'status.json',dict(stage='fitting',completed=len(registry),latest=name))
        print(name,feasible,cal['metrics']['correct'],cal['metrics']['wrong_acquire'],flush=True)
    parent=json.loads((r/'frozen_for_confirm4/selected.json').read_text())['result']['component']
    record('frozen_current_control',predict_component(parent,data,a.root),parent)
    if transition:
        growth=json.loads((r/'frozen_for_confirm5_growth108/selected.json').read_text())['result']['component']
        record('frozen_growth_weights_control',predict_component(growth,data,a.root),growth)
    for view_name in views:
        view=feature_view(data,geometry=view_name=='geometry')
        for population,key,case,transition_factor,acquire_factor in [(p,k,*case) for p,k in populations for case in cases]:
            for seed in [100,200,300]:
                name=f'{view_name}_{population}_seed{seed}'+('_'+case if transition else '')
                fit_data=data
                if transition:
                    from transition_weights import training_weights
                    weights,emphasis=training_weights(data,splits[key],transition_factor,acquire_factor)
                    fit_data=dict(data,weights=weights)
                prob,checkpoint,info=train_model(fit_data,view,splits[key],dev,seed,a.device)
                if transition:info['emphasis']=emphasis
                checkpoint['spec']=dict(name=name,view=view_name,kind='mlp',width=64,weighted=True,training_population=population,loss_emphasis=case)
                torch.save(checkpoint,a.output/(name+'.pt'));write_json(a.output/(name+'_training.json'),info)
                component=dict(kind='parent',name=name,screen=str(a.output))
                for offset in [0.,4.]:record(name+'_bias'+str(offset),replace_availability(prob,prob[:,:8].sum(-1),0.,offset),dict(kind='availability',parent=component,model=None,mix=0.,offset=offset))
    def merit(name):
        v=registry[name];c=v['calibration'];return v['joint_growth_goal_met'],c['metrics']['recall'] if v['joint_growth_goal_met'] else c['metrics']['f1'],-v['raw']['dev']['wrong_acquire']
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=fits,selection_partition='growth combined24-dev with both subgroup gates',acceptance_revision=4,confirmation5_evaluated=transition,confirmation6_evaluated=False))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner))
    print(json.dumps(dict(winner=winner,joint_growth_goal_met=registry[winner]['joint_growth_goal_met']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--weighting',choices=['uniform','transition'],default='uniform');main(p.parse_args())
