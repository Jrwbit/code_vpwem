"""One-step causal proposal association using public appearance/class/box evidence.

No GT identity, target label, contact, or future frame is read. This is not an
occlusion-reidentification system; absent/uncertain matches have explicit zeros.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def associate(current,previous,current_valid,previous_valid):
    current=np.asarray(current);previous=np.asarray(previous)
    ci=np.flatnonzero(current_valid);pi=np.flatnonzero(previous_valid);matches=np.full(8,-1,int)
    if not len(ci) or not len(pi):return matches
    a=current[ci];b=previous[pi];abox=a[:,128:132];bbox=b[:,128:132]
    low=np.maximum(abox[:,None,:2],bbox[None,:,:2]);high=np.minimum(abox[:,None,2:],bbox[None,:,2:])
    intersection=np.maximum(high-low,0).prod(-1)
    area_a=np.maximum(abox[:,2:]-abox[:,:2],0).prod(-1);area_b=np.maximum(bbox[:,2:]-bbox[:,:2],0).prod(-1)
    iou=intersection/np.maximum(area_a[:,None]+area_b[None,:]-intersection,1e-8)
    appearance_a=a[:,:128];appearance_b=b[:,:128]
    cosine=(appearance_a@appearance_b.T)/np.maximum(np.linalg.norm(appearance_a,axis=-1)[:,None]*np.linalg.norm(appearance_b,axis=-1)[None,:],1e-8)
    cosine=np.clip(cosine,-1,1)
    distance=np.linalg.norm((abox[:,:2]+abox[:,2:])[:,None]/2-(bbox[:,:2]+bbox[:,2:])[None,:]/2,axis=-1)
    same=a[:,133:].argmax(-1)[:,None]==b[:,133:].argmax(-1)[None,:]
    allowed=same&((iou>=.1)|((cosine>=.85)&(distance<=.3)))
    cost=.6*(1-iou)+.4*(1-cosine);allowed&=cost<=.75
    left,right=linear_sum_assignment(np.where(allowed,cost,1e6))
    for x,y in zip(left,right):
        if allowed[x,y]:matches[ci[x]]=pi[y]
    return matches


def pair_features(current,previous,current_valid,previous_valid,current_geometry,previous_geometry,gap):
    result=np.zeros((8,12),np.float32)
    if gap<=0 or gap>16:return result
    matching=associate(current,previous,current_valid,previous_valid)
    for slot,old in enumerate(matching):
        if old<0:continue
        a=current[slot];b=previous[old]
        cosine=float(np.dot(a[:128],b[:128])/max(float(np.linalg.norm(a[:128])*np.linalg.norm(b[:128])),1e-8))
        depth_valid=current_geometry[slot,-1]>0 and previous_geometry[old,-1]>0
        delta=current_geometry[slot,:3]-previous_geometry[old,:3] if depth_valid else np.zeros(3)
        result[slot]=np.r_[1.,gap/16.,a[128:132]-b[128:132],np.clip(cosine,-1,1),b[132],delta,float(depth_valid)]
    return result


def add_candidate_history(data):
    result=np.zeros((len(data['rows']),8,12),np.float32)
    for ep in sorted({row['episode_id'] for row in data['rows']}):
        indices=sorted([i for i,row in enumerate(data['rows']) if row['episode_id']==ep],key=lambda i:data['rows'][i]['step'])
        for left,right in zip(indices,indices[1:]):
            gap=data['rows'][right]['step']-data['rows'][left]['step']
            if gap<=0:raise ValueError('Duplicate or nonincreasing observation steps')
            result[right]=pair_features(data['candidates'][right],data['candidates'][left],data['valid'][right],data['valid'][left],
                data['geometry_candidates'][right],data['geometry_candidates'][left],gap)
    return dict(data,candidate_history=result)


def tracking_view(data,view):
    from data import feature_view
    c,x=feature_view(data,rich=view=='rich_geometry',geometry=True)
    # Replace original pooled candidates with a pool of the enriched candidates.
    x=x[:,:-c.shape[-1]];c=np.concatenate([c,data['candidate_history']],-1)
    m=data['valid'][:,:,None]
    return c.astype(np.float32),np.c_[x,(c*m).sum(1)/m.sum(1).clip(1)].astype(np.float32)
