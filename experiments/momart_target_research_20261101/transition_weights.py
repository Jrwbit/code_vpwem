"""Train-only loss emphasis on sparse subject changes and bowl acquisition.

Labels set loss weights only; inference inputs and every evaluation label and
denominator are unchanged. Unknown observations break a transition window.
"""
import numpy as np


def training_weights(data,train,transition_factor=1.,acquire_factor=1.,window=3):
    if transition_factor<1 or acquire_factor<1 or window<1:raise ValueError('Invalid loss emphasis')
    train=set(map(int,train));weights=data['weights'].copy();events=[];emphasized=[]
    episodes={data['rows'][i]['episode_id'] for i in train}
    for episode in sorted(episodes):
        ids=sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==episode],key=lambda i:data['rows'][i]['step'])
        previous=None;remaining=0
        for i in ids:
            row=data['rows'][i]
            if i not in train or row['label'] is None:
                previous=None;remaining=0;continue
            if previous is not None:
                last=data['rows'][previous]
                if not 0<row['step']-last['step']<=16:remaining=0
                elif row['label']['target']['instance_id']!=last['label']['target']['instance_id']:
                    remaining=window;events.append(i)
            if remaining:
                weights[i]*=transition_factor;remaining-=1;emphasized.append(i)
            if data['roles'][i]==0:weights[i]*=acquire_factor
            previous=i
    return weights,dict(transition_events=len(events),window_frames=len(emphasized),
        changed_training_frames=sum(weights[i]!=data['weights'][i] for i in train),
        transition_factor=transition_factor,acquire_factor=acquire_factor,window=window,
        caveat='Weak subject labels affect training loss only; unknown frames break windows; no intent relabelling')
