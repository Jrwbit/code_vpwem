"""Train-only uncertainty flags for far, future-motion acquisition proxies."""
from pathlib import Path
import numpy as np


def far_future_mask(data,audit,index):
    def key(row):return str(Path(row['source']).resolve()),row['demo'],int(row['step'])
    lookup={key(row):row for row in audit['records']}
    if len(lookup)!=len(audit['records']):raise ValueError('Duplicate contact audit query')
    mask=np.zeros(len(data['rows']),bool)
    for i in index:
        if data['y'][i]<0 or data['roles'][i]!=0:continue
        row=data['rows'][i];value=lookup.get(key(row))
        if value is None:raise ValueError('Missing acquire-bowl contact audit query')
        target=row['label']['target'];onset=target.get('onset');delta=onset-row['step'] if onset is not None else None
        if value['target']!=target['instance_id'] or value['future_motion_steps']!=delta:raise ValueError('Contact audit and unchanged target labels disagree')
        mask[i]=delta is not None and delta>0 and value['facts']['target_far']
    return mask
