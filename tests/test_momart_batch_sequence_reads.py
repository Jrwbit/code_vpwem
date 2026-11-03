import numpy as np
import torch
import robomimic.utils.obs_utils as ObsUtils
from cleandiffuser.dataset.dataset_utils import SequenceSampler
from momart_no_history_baseline.cache import TrainingArray
from momart_no_history_baseline.dataset import ZarrNoHistorySequenceDataset
from momart_boxed_no_history.dataset import BoxedNoHistoryDataset


class Replay(dict):
    episode_ends=np.array([1,6])


def make_dataset(cls):
    ObsUtils.initialize_obs_modality_mapping_from_dict(dict(rgb=['rgb'],depth=['depth'],low_dim=['proprio']))
    replay=Replay();rng=np.random.default_rng(100);mapping=np.array([0,2,3,5,6,9])
    for key,shape,dtype in [('rgb',(8,9,3),np.uint8),('depth',(8,9,1),np.float32),('proprio',(10,),np.float32),('action',(10,),np.float32)]:
        array=rng.integers(0,255,(10,*shape)).astype(dtype);replay[key]=TrainingArray(array,mapping)
    d=object.__new__(cls);d.replay_buffer=replay;d.obs_keys=('rgb','depth','proprio');d.obs_steps=2;d.seq_length=9
    d.sampler=SequenceSampler(replay,9,pad_after=8,keys=[*d.obs_keys,'action'],key_first_k={k:2 for k in d.obs_keys})
    if cls is BoxedNoHistoryDataset:
        d.boxes=np.tile([.1,.2,.8,.9],(6,1));d.valid=np.array([True,False,True,True,False,True])
    return d


def test_batched_reads_preserve_pixels_actions_order_duplicates_and_short_episode_padding():
    for cls in [ZarrNoHistorySequenceDataset,BoxedNoHistoryDataset]:
        d=make_dataset(cls);indices=[5,0,4,2,5,1,3];expected=[d[i] for i in indices];actual=d.__getitems__(indices)
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            for key in a['obs']:
                np.testing.assert_array_equal(a['obs'][key],b['obs'][key]);assert a['obs'][key].dtype==b['obs'][key].dtype
            np.testing.assert_array_equal(a['actions'],b['actions'])
        assert d.__getitems__([])==[]
        batch=next(iter(torch.utils.data.DataLoader(d,batch_sampler=[indices],num_workers=0)))
        np.testing.assert_array_equal(batch['actions'],np.stack([s['actions'] for s in expected]))


def test_training_array_merges_contiguous_hdf5_reads_and_preserves_noncontiguous_mapping():
    class CountingArray:
        def __init__(self):self.value=np.arange(60).reshape(10,6);self.shape=self.value.shape;self.dtype=self.value.dtype;self.calls=[]
        def __len__(self):return len(self.value)
        def __getitem__(self,key):self.calls.append(key);return self.value[key]
    array=CountingArray();mapped=TrainingArray(array,np.array([0,1,4,5,8,9]))
    ids=np.array([3,2,3,0,1]);actual=mapped.read_frames(ids)
    np.testing.assert_array_equal(actual,array.value[[5,4,5,0,1]])
    assert len(array.calls)==2


def test_object_wrapper_retains_batch_loading_and_exact_anchor_snapshots():
    from momart_object_memory.dataset import ObjectDataset
    from momart_object_memory.state import ObjectBank,ObjectObservation
    bank=ObjectBank(2,3,4);snapshots=[]
    for step in range(6):
        bank.update(step,step*.05,[ObjectObservation(0,np.array([.1,.1,.3,.3]),.9,np.full(4,step,np.float32))])
        snapshots.append(bank.snapshot('a3'))
    for mode in ['a1','a2','a3','a3_latest_control']:
        wrapped=object.__new__(ObjectDataset);wrapped.base=make_dataset(BoxedNoHistoryDataset);wrapped.mode=mode
        wrapped.root={key:np.stack([s[key] for s in snapshots]) for key in snapshots[0]}
        ids=[5,0,4,2,5];expected=[wrapped[i] for i in ids];actual=wrapped.__getitems__(ids)
        for a,b in zip(actual,expected):
            for key in a['objects']:np.testing.assert_array_equal(a['objects'][key],b['objects'][key])
            for key in a['obs']:np.testing.assert_array_equal(a['obs'][key],b['obs'][key])
            np.testing.assert_array_equal(a['actions'],b['actions'])
