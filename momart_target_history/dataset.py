"""Same target-painted current observations; only main RGB is read for history."""
import numpy as np
from momart_target_no_history.dataset import TargetNoHistoryDataset
from momart_no_history_baseline.batch_read import read_sequence_batch

VARIANTS=('box','recent8','stride20_8')


def history_indices(query,start,variant):
    if variant=='recent8':frames=np.arange(max(start,query-8),query,dtype=np.int64)
    elif variant=='stride20_8':
        last=(query-start-1)//20
        frames=start+20*np.arange(max(0,last-7),last+1,dtype=np.int64) if last>=0 else np.empty(0,np.int64)
    else:raise ValueError('Unknown history variant')
    indices=np.full(8,-1,np.int64);indices[8-len(frames):]=frames
    return indices


class HistoryTargetDataset(TargetNoHistoryDataset):
    def __init__(self,*args,variant,**kwargs):
        super().__init__(*args,mode='box',**kwargs)
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant
        self.detector_metadata.update(history_variant=variant,history_length=0 if variant=='box' else 8,
            history_rule='strictly before newest observation; stride20 uses episode steps0,20,40,...')
        self._ends=np.asarray(self.replay_buffer.episode_ends)
        self._starts=np.r_[0,self._ends[:-1]]

    def get_dataset_sampler(self):return None

    def __getitem__(self,index):return self.__getitems__([index])[0]

    def __getitems__(self,indices):
        if not len(indices):return []
        samples=[self._target_sample(sample,index) for sample,index in zip(read_sequence_batch(self,indices),indices)]
        for sample,index in zip(samples,indices):sample['sample_index']=np.int64(index)
        if self.variant=='box':return samples
        histories=[];queries=[]
        for index in indices:
            begin,end,_,_=self.sampler.indices[index];query=min(begin+self.obs_steps-1,end-1)
            episode=int(np.searchsorted(self._ends,query,side='right'));start=int(self._starts[episode])
            histories.append(history_indices(query,start,self.variant));queries.append(query)
        frames=np.stack(histories);real=frames>=0
        rgb=np.zeros((*frames.shape,*self.replay_buffer['rgb'].shape[1:]),np.uint8)
        if real.any():rgb[real]=self.replay_buffer['rgb'].read_frames(frames[real])
        boxes=np.zeros((*frames.shape,4),np.float32);valid=np.zeros(frames.shape,bool);ages=np.zeros(frames.shape,np.float32)
        boxes[real]=self.boxes[frames[real]];valid[real]=self.valid[frames[real]]
        ages[real]=(np.asarray(queries)[:,None]-frames)[real]
        for j,sample in enumerate(samples):
            sample['obs'].update(_history_rgb=np.moveaxis(rgb[j],-1,1),_history_boxes=boxes[j],_history_valid=valid[j],_history_ages=ages[j])
        return samples
