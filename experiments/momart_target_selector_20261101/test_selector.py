import numpy as np
import torch
from proxy import motion_intervals,target_at,target_candidate
from selector import TargetSelector


def test_motion_requires_extent_and_groups_short_pauses():
    assert motion_intervals(np.zeros(20),.001,.01)==[]
    assert motion_intervals([0,0,.006,.012,.018,.018],.001,.01)==[(2,4)]


def test_targets_do_not_force_unknown_or_simultaneous_intent():
    a=dict(instance_id='a',onset=20,end=30)
    b=dict(instance_id='b',onset=25,end=40)
    assert target_at([a],0,lookahead=16) is None
    assert target_at([a],4,lookahead=16)==a
    assert target_at([a,b],26) is None
    assert target_at([a],31) is None


def test_instance_matching_is_class_and_location_specific():
    truth=[dict(instance_id='a',class_id=5,box=[0,0,10,10])]
    target=dict(instance_id='a')
    objects=[dict(class_id=5,box=[20,20,30,30]),dict(class_id=4,box=[0,0,10,10]),
             dict(class_id=5,box=[0,0,10,10])]
    assert target_candidate(objects,truth,target)==2
    assert target_candidate(objects[:2],truth,target)==-1
    assert target_candidate(objects,truth,None) is None


def test_pointer_masks_invalid_slots_and_is_permutation_equivariant():
    torch.manual_seed(100)
    model=TargetSelector(5,7).eval();context=torch.randn(2,7);objects=torch.randn(2,8,5)
    valid=torch.tensor([[1,0,1,0,0,0,0,0],[0]*8],dtype=torch.bool)
    first=model(objects,context,valid)
    assert first[1].argmax()==8 and torch.isneginf(first[0,1])
    perm=torch.tensor([3,1,0,7,5,6,2,4])
    second=model(objects[:,perm],context,valid[:,perm])
    torch.testing.assert_close(second[:,:8],first[:,:8][:,perm])
    torch.testing.assert_close(second[:,8],first[:,8])


def test_lock_never_returns_old_box_when_target_missing_and_can_reset():
    from selector import TargetLock
    lock=TargetLock();obj=dict(class_id=0,box=[0,0,10,10])
    assert lock.choose([obj],[.8,.2],0)==0
    assert lock.choose([],[1.],8) is None
    assert lock.choose([obj],[.1,.9],16) is None
    lock.reset()
    assert lock.choose([obj],[.9,.1],0)==0
