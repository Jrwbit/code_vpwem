"""Normalize train-only extra observations with original training statistics."""
import numpy as np
import torch


def prepare_augmentation(data,augmentation,train,normalizer,device):
    base=np.asarray(augmentation['base_index'],dtype=int);train=np.asarray(train,dtype=int)
    if len(set(base))!=len(base) or set(base)!=set(train):raise ValueError('Augmentation must cover exactly the known training queries once')
    if np.any(np.asarray(data['y'])[base]<0):raise ValueError('Unknown original targets cannot enter supervised augmentation')
    raw=np.asarray(augmentation['candidates'],np.float32);context=np.asarray(augmentation['context'],np.float32);valid=np.asarray(augmentation['valid'],bool);labels=np.asarray(augmentation['y'],np.int64)
    if raw.shape!=(len(base),8,len(normalizer['cm'])) or context.shape!=(len(base),len(normalizer['xm'])) or valid.shape!=(len(base),8) or labels.shape!=(len(base),):raise ValueError('Unexpected augmented sensor dimensions')
    if not np.isfinite(raw).all() or not np.isfinite(context).all() or np.any((labels<0)|(labels>8)):raise ValueError('Invalid augmented features or labels')
    ids=np.flatnonzero(labels<8)
    if not valid[ids,labels[ids]].all():raise ValueError('Augmented positive slot is padded')
    lookup=np.full(len(data['y']),-1,int);lookup[base]=np.arange(len(base))
    return dict(c=torch.tensor((raw-normalizer['cm'])/normalizer['cs'],device=device),x=torch.tensor((context-normalizer['xm'])/normalizer['xs'],device=device),
        mask=torch.tensor(valid,device=device),labels=torch.tensor(labels,device=device),lookup=lookup)
