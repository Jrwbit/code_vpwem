"""Ground an externally supplied class query; never infer or silently change it.

The caller must obtain the query causally for deployment. Ground-truth queries
are permitted only in explicitly labelled offline diagnostic upper bounds.
"""
import numpy as np


def ground_query(probability, classes, valid, query, *, keep_null=True):
    probability=np.asarray(probability)
    classes=np.asarray(classes);valid=np.asarray(valid,dtype=bool);query=np.asarray(query)
    if probability.ndim!=2 or probability.shape[1]!=9 or classes.shape!=probability[:,:8].shape or valid.shape!=classes.shape or query.shape!=(len(probability),):
        raise ValueError('Expected probability[N,9], classes/valid[N,8], query[N]')
    if not np.isfinite(probability).all() or (probability<0).any() or not np.allclose(probability.sum(-1),1,atol=1e-5):
        raise ValueError('Probability must be finite, nonnegative and normalized')
    if not np.isin(query,[-1,0,1,2,3,4,5]).all():raise ValueError('Unknown class query; use -1 for no query')
    allowed=valid&(classes==query[:,None])&(query[:,None]>=0)
    result=np.zeros_like(probability)
    result[:,:8]=np.where(allowed,probability[:,:8],0)
    if keep_null:result[:,8]=probability[:,8]
    total=result.sum(-1)
    empty=total<=0
    result[empty,8]=1
    return result/result.sum(-1,keepdims=True)
