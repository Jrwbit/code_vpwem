"""Fixed public clock recurrence, with temporary off-grid queries and no GT resets."""
import hashlib
import time
import numpy as np
import torch
from models import RecurrentPointer,weighted_loss
from data import feature_view,metrics
from checkpoint_selection import checkpoint_merit
from main_history import query_context,load_history


def grid_batch(rows,candidates,context,valid,history_context,history_rows):
    episodes=sorted({r['episode_id'] for r in rows});episode_index={ep:i for i,ep in enumerate(episodes)}
    indices={ep:[i for i,r in enumerate(history_rows) if r['episode_id']==ep and r['is_grid']] for ep in episodes}
    for ep,ids in indices.items():
        assert ids and [history_rows[i]['step'] for i in ids]==list(range(0,history_rows[ids[-1]]['step']+1,8))
    length=max(map(len,indices.values()));x=context.new_zeros((len(episodes),length,context.shape[-1]));gap=context.new_zeros((len(episodes),length))
    for ep,ids in indices.items():
        j=episode_index[ep];x[j,:len(ids)]=history_context[ids];gap[j,1:len(ids)]=float(np.log(2))
    qe=np.array([episode_index[r['episode_id']] for r in rows]);qt=np.array([r['step']//8 for r in rows]);remainder=np.array([r['step']%8 for r in rows])
    assert all(qt[i]<len(indices[r['episode_id']]) for i,r in enumerate(rows))
    return dict(grid=x,gaps=gap,c=candidates,x=context,mask=valid,query_episode=torch.as_tensor(qe,device=x.device),
        query_floor=torch.as_tensor(qt,device=x.device),query_remainder=torch.as_tensor(np.log1p(remainder/8),device=x.device,dtype=x.dtype),
        off_grid=torch.as_tensor(remainder!=0,device=x.device),episode_ids=episodes,query_episode_numpy=qe,query_indices=np.arange(len(rows)))


def subset_batch(batch,episode_indices):
    local={int(ep):i for i,ep in enumerate(episode_indices)};ids=np.flatnonzero(np.isin(batch['query_episode_numpy'],episode_indices))
    result={key:batch[key][ids] for key in ['c','x','mask','query_floor','query_remainder','off_grid']}
    result.update(grid=batch['grid'][episode_indices],gaps=batch['gaps'][episode_indices],query_indices=ids,
        query_episode=torch.as_tensor([local[int(batch['query_episode_numpy'][i])] for i in ids],device=batch['grid'].device))
    return result


def query_logits(net,batch):
    projected=net.project(torch.cat([batch['grid'],batch['gaps'][:,:,None]],-1))
    if net.reset_each_frame:
        encoded,_=net.gru(projected.reshape(-1,1,net.gru.input_size));encoded=encoded.reshape(*projected.shape)
    else:encoded,_=net.gru(projected)
    hidden=encoded[batch['query_episode'],batch['query_floor']]
    off=batch['off_grid']
    if off.any():
        inp=net.project(torch.cat([batch['x'][off],batch['query_remainder'][off,None]],-1))[:,None]
        before=None if net.reset_each_frame else hidden[off][None]
        temporary,_=net.gru(inp,before);hidden=hidden.clone();hidden[off]=temporary[:,0]
    return net.pointer(batch['c'],hidden,batch['mask'])[0]


def sequence_probability(net,batch):
    with torch.inference_mode():return query_logits(net,batch).softmax(-1).cpu().numpy()


def make_batch(data,history,c,x,valid,normalizer):
    h=torch.as_tensor((history['context']-normalizer['xm'])/normalizer['xs'],device=x.device)
    return grid_batch(data['rows'],c,x,valid,h,history['rows'])


def predict_grid_checkpoint(checkpoint,data):
    history=data.get('main_history')
    if history is None:history=load_history(checkpoint['spec']['main_history_path'])
    raw,_=feature_view(data,geometry=True);context=query_context(data,history);norm=checkpoint['normalizer']
    c=torch.tensor((raw-norm['cm'])/norm['cs']);x=torch.tensor((context-norm['xm'])/norm['xs']);mask=torch.tensor(data['valid'])
    net=RecurrentPointer(c.shape[-1],x.shape[-1],checkpoint['spec']['width'],checkpoint['spec']['reset_each_frame']).eval();net.load_state_dict(checkpoint['state'])
    return sequence_probability(net,make_batch(data,history,c,x,mask,norm))


def fit_recurrent(data,view,splits,seed,device,mode,assess_fixed,history):
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(2)
    raw,context=view;train=splits['train'];valid=data['valid'];started=time.monotonic()
    norm=dict(cm=raw[train][valid[train]].mean(0),cs=raw[train][valid[train]].std(0).clip(.05),xm=context[train].mean(0),xs=context[train].std(0).clip(.05))
    c=torch.as_tensor((raw-norm['cm'])/norm['cs'],device=device);x=torch.as_tensor((context-norm['xm'])/norm['xs'],device=device);mask=torch.as_tensor(valid,device=device)
    all_seq=make_batch(data,history,c,x,mask,norm)
    train_ids={data['rows'][i]['episode_id'] for i in train};used=np.array([i for i,ep in enumerate(all_seq['episode_ids']) if ep in train_ids])
    labels=torch.as_tensor(data['y'],device=device);weights=torch.as_tensor(data['weights'],device=device)
    net=RecurrentPointer(c.shape[-1],x.shape[-1],64,reset_each_frame=mode=='reset').to(device)
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in sorted(net.state_dict().items()))).hexdigest()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    best_f1=-1.;best_epoch=0;best_fixed=None;states={};log=[]
    for epoch in range(150):
        net.train()
        for episode_batch in np.array_split(np.random.permutation(used),max(1,(len(used)+7)//8)):
            batch=subset_batch(all_seq,episode_batch);ids=batch['query_indices'];opt.zero_grad()
            logits=query_logits(net,batch);loss=weighted_loss(logits,labels[ids],weights[ids],batch['mask']);loss.backward();opt.step()
        net.eval();probability=sequence_probability(net,all_seq);raw_metrics=metrics(probability.argmax(-1),splits['dev'],data)
        point=assess_fixed(probability);merit=checkpoint_merit(dict(joint_growth_goal_met=point['joint_goal_met'],calibration=point))
        log.append(dict(epoch=epoch+1,loss=float(loss.detach()),raw_dev=raw_metrics,fixed_metrics=point['metrics'],fixed_sequence=point['sequence'],fixed_goal_met=point['joint_goal_met']))
        def save(policy):states[policy]=dict(state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},seed=seed,best_epoch=epoch+1,normalizer=norm)
        if raw_metrics['f1']>best_f1:best_f1=raw_metrics['f1'];best_epoch=epoch+1;save('raw_f1')
        if best_fixed is None or merit>best_fixed:best_fixed=merit;save('fixed_gate')
        if (epoch+1)%10==0:print('TRAIN',mode,seed,epoch+1,raw_metrics['correct'],point['joint_goal_met'],flush=True)
        if epoch+1-best_epoch>=30:break
    predictions={}
    for policy,checkpoint in states.items():
        net.load_state_dict(checkpoint['state']);net.eval();predictions[policy]=sequence_probability(net,all_seq)
    observed={ (r['episode_id'],r['step']) for r in data['rows'] }
    background=sum(r['episode_id'] in train_ids and r['is_grid'] and (r['episode_id'],r['step']) not in observed for r in history['rows'])
    return predictions,states,dict(history=log,seconds=time.monotonic()-started,initial_parameter_sha256=initial,train_episodes=len(train_ids),
        background_training_observations=background,unknown_training_queries=int(sum(data['y'][i]<0 and data['rows'][i]['episode_id'] in train_ids for i in range(len(data['rows'])))))
