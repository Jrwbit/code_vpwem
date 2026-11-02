"""Joint subject selection and future-command auxiliary supervision.

No teacher-forced commands: subject logits always consume a predicted plan.
Only training episode action labels affect gradients; original dev labels remain.
"""
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
from confirm import predict_component,apply_calibration
from data import partitions,feature_view,metrics
from expanded import load_expanded_data,calibrate_v4,baseline_for
from models import weighted_loss
from plan_model import ActionPlanPointer,plan_loss
from plan_labels import load_plan_labels


def main(a):
    torch.set_num_threads(2);r=a.root/'momart_target_research_20261101';a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'frozen_search_protocol.json',dict(method='self-predicted plan query with command auxiliary loss',
        train='unchanged expanded28 episodes; unknown subject frames can supervise action loss only',dev='unchanged expanded8 episodes; original full subject labels',
        view='geometry',seeds=[100,200,300],auxiliary_weights=[0.,.1,1.],fit_count=9,bias_controls=[0.,4.],frozen_parent_control=True,
        plan='8x10 predicted commands from public current/past image/robot features; predicted plan consumed even during training',
        target_file_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),targets_forbidden_as_inference_inputs=True,
        confirm5_manifest_sha256=hashlib.sha256((r/'confirm5_corpus/manifest.json').read_bytes()).hexdigest(),confirmation5_loaded=False,
        caveat='Auxiliary prediction is a selector head, not the deployed diffusion action policy. No closed-loop claims; weak subject labels unchanged.'))
    data=load_expanded_data(a.root);splits=partitions(data)
    train_all=np.array([i for i,row in enumerate(data['rows']) if row['tag']=='old' and row['split']=='train'],int)
    assert not set(train_all)&set(splits['dev'])
    target,action_valid=load_plan_labels(data,a.labels);target=torch.tensor(target);action_valid=torch.tensor(action_valid)
    raw,context=feature_view(data,geometry=True);train=splits['train'];valid=data['valid']
    cm=raw[train][valid[train]].mean(0);cs=raw[train][valid[train]].std(0).clip(.05)
    xm=context[train].mean(0);xs=context[train].std(0).clip(.05)
    c=torch.tensor((raw-cm)/cs);x=torch.tensor((context-xm)/xs);mask=torch.tensor(valid)
    labels=torch.tensor(data['y']);weights=torch.tensor(data['weights'])
    baseline=baseline_for(predict_component(dict(kind='parent',name='previous_pointer',screen=str(r/'screen_seed100')),data,a.root),splits['dev'],data)
    registry={}
    def record(name,prob,component):
        assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(-1),1,atol=1e-5)
        cal=calibrate_v4(prob,splits['dev'],data,baseline);pred=apply_calibration(prob,cal)
        registry[name]=dict(name=name,component=component,calibration=cal,
            raw={k:metrics(prob.argmax(-1),v,data) for k,v in splits.items() if len(v)},
            reject={k:metrics(pred,v,data) for k,v in splits.items() if len(v)})
        np.savez_compressed(a.output/(name+'_probabilities.npz'),probability=prob);write_json(a.output/'registry.json',registry)
        write_json(a.output/'status.json',dict(stage='fitting',completed=len(registry),latest=name))
        print(name,cal['joint_goal_met'],cal['metrics']['correct'],cal['metrics']['wrong_acquire'],flush=True)
    parent=json.loads((r/'frozen_for_confirm4/selected.json').read_text())['result']['component']
    record('frozen_parent_control',predict_component(parent,data,a.root),parent)
    write_json(a.output/'data_audit.json',dict(training_frames=len(train_all),training_subject_labels=len(train),dev_subject_labels=len(splits['dev']),
        gradient_episodes=sorted({data['rows'][i]['episode_id'] for i in train_all}),unknown_subject_frames_retained_for_action_loss=True))
    for auxiliary in [0.,.1,1.]:
        for seed in [100,200,300]:
            torch.manual_seed(seed);np.random.seed(seed);net=ActionPlanPointer(c.shape[-1],x.shape[-1])
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001);best=-1.;best_epoch=0;state=None;history=[];started=time.monotonic()
            for epoch in range(150):
                net.train()
                for batch in np.array_split(np.random.permutation(train_all),max(1,(len(train_all)+31)//32)):
                    opt.zero_grad();logits,plan=net(c[batch],x[batch],mask[batch])
                    subject=weighted_loss(logits,labels[batch],weights[batch],mask[batch])
                    commands=plan_loss(plan,target[batch],action_valid[batch]);loss=subject+auxiliary*commands
                    loss.backward();opt.step()
                net.eval()
                with torch.inference_mode():prob=net(c,x,mask)[0].softmax(-1).numpy()
                value=metrics(prob.argmax(-1),splits['dev'],data)
                history.append(dict(epoch=epoch+1,subject_loss=float(subject),plan_loss=float(commands),dev=value))
                if value['f1']>best:best=value['f1'];best_epoch=epoch+1;state=copy.deepcopy(net.state_dict())
                if epoch+1-best_epoch>=30:break
            net.load_state_dict(state);net.eval()
            with torch.inference_mode():prob=net(c,x,mask)[0].softmax(-1).numpy()
            name=f'plan_aux{auxiliary}_seed{seed}';spec=dict(name=name,kind='plan_query',view='geometry',width=64,horizon=8,auxiliary_weight=auxiliary)
            torch.save(dict(state=state,spec=spec,seed=seed,best_epoch=best_epoch,normalizer=dict(cm=cm,cs=cs,xm=xm,xs=xs)),a.output/(name+'.pt'))
            write_json(a.output/(name+'_training.json'),dict(history=history,seconds=time.monotonic()-started,best_epoch=best_epoch))
            component=dict(kind='parent',name=name,screen=str(a.output))
            for offset in [0.,4.]:record(name+'_bias'+str(offset),replace_availability(prob,prob[:,:8].sum(-1),0.,offset),dict(kind='availability',parent=component,model=None,mix=0.,offset=offset))
    def merit(name):
        cal=registry[name]['calibration'];return cal['joint_goal_met'],cal['metrics']['recall'] if cal['joint_goal_met'] else cal['metrics']['f1'],cal['metrics']['precision']
    winner=max(registry,key=merit)
    write_json(a.output/'selected.json',dict(name=winner,result=registry[winner],method_count=len(registry),fit_count=9,selection_partition='expanded/dev only',acceptance_revision=4,confirmation5_evaluated=False))
    write_json(a.output/'status.json',dict(stage='complete',completed=len(registry),winner=winner))
    print(json.dumps(dict(winner=winner,calibration=registry[winner]['calibration']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','labels','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
