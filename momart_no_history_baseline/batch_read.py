"""Coalesce raw frame reads while preserving sequence order and tail padding."""
import numpy as np


def read_sequence_batch(dataset, indices):
    indices=np.asarray(indices,dtype=np.int64)
    if indices.ndim!=1:raise ValueError('Expected one-dimensional sample indices')
    if not len(indices):return []
    windows=dataset.sampler.indices[indices]
    if np.any(windows[:,2]!=0):raise ValueError('No-history batch reads require no leading padding')
    start=windows[:,0,None];last=windows[:,1,None]-1;arrays={}
    for key in (*dataset.obs_keys,'action'):
        length=dataset.seq_length if key=='action' else dataset.obs_steps
        frames=np.minimum(start+np.arange(length)[None,:],last)
        source=dataset.replay_buffer[key]
        values=source.read_frames(frames.reshape(-1))
        arrays[key]=values.reshape(len(indices),length,*source.shape[1:])
    return [{key:value[j] for key,value in arrays.items()} for j in range(len(indices))]
