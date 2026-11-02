"""Training-only confidence treatment for uniformly audited weak-label conflicts.

A review flag is not an alternative target label. Targets and evaluation remain
unchanged; downweight only known training frames and never delete a frame.
"""
import json
from pathlib import Path
import numpy as np


def audit_key(row):
    return str(Path(row['source']).resolve()),row['demo'],int(row['step'])


def conflict_mask(data,audit,index):
    lookup={audit_key(row):row for row in audit['records']}
    if len(lookup)!=len(audit['records']):raise ValueError('Duplicate physical audit frames')
    mask=np.zeros(len(data['rows']),bool)
    for i in index:
        row=lookup[audit_key(data['rows'][i])]
        if not row['old_holding_reproduced']:raise ValueError('Weak holding audit did not reproduce old labels')
        mask[i]=row['old_held_with_carrier_conflict'] or row['all_old_future_holds_carrier_conflicted']
    return mask


def downweight_training(data,train,review,factor):
    if not 0<factor<=1:raise ValueError('Use positive confidence; frame deletion is not this experiment')
    train=np.asarray(train,dtype=int);review=np.asarray(review,dtype=bool)
    if review.shape!=data['weights'].shape:raise ValueError('Review mask shape mismatch')
    weights=data['weights'].copy();weights[train]*=np.where(review[train],factor,1)
    return dict(data,weights=weights)
