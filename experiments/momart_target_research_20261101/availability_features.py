"""Public availability evidence and mass-preserving identity/visibility factorization."""
import numpy as np


def public_availability_features(candidates,valid,geometry,context):
    classes=candidates[:,:,133:].argmax(-1);stats=[]
    for cls in range(6):
        mask=valid&(classes==cls);depth_valid=mask&(geometry[:,:,-1]>0)
        stats.extend([mask.mean(-1),np.where(mask,candidates[:,:,132],0).max(-1),
            np.minimum(np.where(depth_valid,geometry[:,:,0],1.5).min(-1),1.5)/1.5])
    return np.c_[context,np.stack(stats,-1)].astype(np.float32)


def replace_availability(probability,availability,mix=1.,offset=0.):
    if mix==0 and offset==0:return probability.copy()
    mass=probability[:,:8].sum(-1);q=np.asarray(availability).reshape(-1)
    def logit(value):
        value=np.clip(value,1e-6,1-1e-6);return np.log(value)-np.log1p(-value)
    z=(1-mix)*logit(mass)+mix*logit(q)+offset;new_mass=1/(1+np.exp(-np.clip(z,-30,30)))
    result=np.zeros_like(probability);nonempty=mass>0
    result[nonempty,:8]=probability[nonempty,:8]/mass[nonempty,None]*new_mass[nonempty,None]
    result[:,8]=1-result[:,:8].sum(-1)
    result=np.clip(result,0,None);result/=result.sum(-1,keepdims=True)
    return result
