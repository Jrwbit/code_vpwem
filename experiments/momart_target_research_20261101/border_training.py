"""Train-only emphasis on recorded small border targets; no invented crops."""
import numpy as np
from audit_border_visibility import border_flags


def border_training_mask(data,index,short_side_pixels=8,scope='all'):
    if scope not in ['all','bowl']:raise ValueError('Unknown border training scope')
    result=np.zeros(len(data['rows']),bool)
    for i in index:
        if data['y'][i]<0:continue
        if scope=='bowl' and data['roles'][i] not in [0,4]:continue
        result[i]=border_flags(data['rows'][i]['objects'],data['positive'][i],[90,120],short_side_pixels)
    return result


def emphasize_training(data,train,review,factor):
    if factor<1 or not np.isfinite(factor):raise ValueError('Training emphasis must be finite and at least one')
    weights=data['weights'].copy();ids=np.asarray(train,dtype=int);weights[ids]*=np.where(review[ids],factor,1.)
    return dict(data,weights=weights)
