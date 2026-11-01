import numpy as np
import torch
from subject_labels import subject_label
from export_extra import depth_summary
from models import Pointer,RecurrentPointer,weighted_loss


def record(**kwargs):
    value=dict(old_target=dict(instance_id='rack',class_id=3,onset=20),step=15,
       evidence=dict(bowl_distance=.01,grasped=1),label=dict(subject='bowl'),held_now=False,held_before=False,held_soon=False)
    value.update(kwargs);return value


def test_holding_subject_is_bowl_and_release_is_not_negative_intent():
    label=subject_label(record(held_now=True))
    assert label['target']['instance_id']=='bowl' and label['target']['class_id']==0
    assert subject_label(record(held_before=True)) is None
    assert subject_label(record(held_soon=True))['target']['class_id']==0
    assert subject_label(record())['target']['class_id']==3


def test_invalid_depth_is_explicit_and_soft_prior_can_be_overridden():
    np.testing.assert_array_equal(depth_summary(np.zeros((3,3))),np.zeros(4))
    torch.manual_seed(7)
    net=Pointer(4,5,width=16,soft_role=.5).eval()
    logits,_=net(torch.randn(1,2,4),torch.randn(1,5),torch.tensor([[True,True]]),torch.tensor([[0,3]]))
    assert torch.isfinite(logits).all()


def test_loss_smoothing_does_not_use_invalid_slots_or_unknown_labels():
    logits=torch.tensor([[1.,float('-inf'),0.],[0.,float('-inf'),1.]],requires_grad=True)
    # The selectors use eight slots; directly exercise a full shape with seven masked slots.
    logits=torch.cat([logits[:,:1],logits[:,1:2].expand(-1,7),logits[:,-1:]],-1)
    valid=torch.tensor([[True]+[False]*7]*2)
    loss=weighted_loss(logits,torch.tensor([0,-100]),torch.tensor([1.,1.]),valid,.1)
    assert torch.isfinite(loss);loss.backward()


def test_recurrent_predictions_are_prefix_causal_and_episode_local():
    torch.manual_seed(10);net=RecurrentPointer(4,5,16).eval()
    c=torch.randn(2,4,8,4);x=torch.randn(2,4,5);valid=torch.ones(2,4,8,dtype=torch.bool);gap=torch.ones(2,4)
    a=net(c,x,valid,gap);b=net(c[:,:2],x[:,:2],valid[:,:2],gap[:,:2])
    torch.testing.assert_close(a[:,:2],b)
    single=net(c[:1],x[:1],valid[:1],gap[:1]);torch.testing.assert_close(a[:1],single)


def test_tree_public_features_do_not_depend_on_private_targets():
    from benchmark import candidate_rows
    data=dict(rows=[{}],valid=np.array([[True]+[False]*7]),y=np.array([0]),
        positive=np.array([[True]+[False]*7]),weights=np.array([1.]))
    view=dict(raw=np.ones((1,8,4),np.float32),context=np.ones((1,3),np.float32))
    first=candidate_rows(data,view)
    data['y'][0]=8;data['positive'][:]=False;data['weights'][:]=.2
    second=candidate_rows(data,view)
    np.testing.assert_array_equal(first[0],second[0]);assert first[3]==second[3]


def test_subject_memory_is_prefix_causal_and_uses_public_grasp_signal():
    from benchmark import subject_memory
    context=np.zeros((3,280),np.float32);context[:,258]=1
    candidates=np.ones((3,8,139),np.float32)
    rows=[dict(episode_id='e',step=t,objects=[dict(class_id=0,box=[0,0,10,10]),dict(class_id=3,box=[20,20,30,30])]) for t in (0,8,16)]
    data=dict(context=context,candidates=candidates,rows=rows)
    prob=np.zeros((3,9),np.float32);prob[:,0]=[.8,.25,.2];prob[:,1]=[.1,.65,.7];prob[:,8]=.1
    full=subject_memory(prob,data,2.)
    prefix={k:v[:2] for k,v in data.items()};partial=subject_memory(prob[:2],prefix,2.)
    np.testing.assert_allclose(full[:2],partial)
    assert full[1].argmax()==0
