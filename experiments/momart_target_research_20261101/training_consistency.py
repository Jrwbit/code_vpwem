"""Training-only stable-subject pairs and slot-permutation-invariant class JS.

Reference identities only select loss pairs. Inference retains existing causal public features;
unknown, unavailable, cross-split and changed-subject boundaries are not bridged.
"""
import numpy as np
import torch


def stable_training_predecessors(data,train,max_gap=16):
    train=set(map(int,train));previous=np.full(len(data['rows']),-1,np.int64)
    episodes={data['rows'][i]['episode_id'] for i in train}
    for episode in sorted(episodes):
        ids=sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==episode],key=lambda i:data['rows'][i]['step'])
        for left,right in zip(ids,ids[1:]):
            a,b=data['rows'][left],data['rows'][right]
            if left not in train or right not in train:continue
            if a['label'] is None or b['label'] is None:continue
            if not data['positive'][left].any() or not data['positive'][right].any():continue
            if not 0<b['step']-a['step']<=max_gap:continue
            if a['label']['target']['instance_id']==b['label']['target']['instance_id']:previous[right]=left
    return previous


def class_distribution(logits,classes):
    probability=logits.softmax(-1)
    pooled=probability.new_zeros((len(probability),7))
    pooled.scatter_add_(1,classes,probability[:,:-1])
    pooled[:,6]=probability[:,-1]
    return pooled


def class_consistency_loss(current,other,current_classes,other_classes,weights):
    if len(current)==0:return current[:,-1].sum()*0
    p=class_distribution(current,current_classes);q=class_distribution(other,other_classes)
    mid=(p+q)*.5;log_mid=mid.clamp_min(1e-8).log()
    js=.5*((p*(p.clamp_min(1e-8).log()-log_mid)).sum(-1)+(q*(q.clamp_min(1e-8).log()-log_mid)).sum(-1))
    return (js*weights).sum()/weights.sum().clamp_min(1e-8)
