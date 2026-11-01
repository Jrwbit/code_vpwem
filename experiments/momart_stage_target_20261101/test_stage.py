import numpy as np
import torch
from phase_logic import holding_proxy,stage_label
from model import gated_logits


def evidence(grasped=1,distance=.01,relative=(0,0,0)):
    return dict(grasped=grasped,bowl_distance=distance,bowl_relative=relative)


def test_gripper_closed_alone_does_not_establish_holding_bowl():
    assert holding_proxy(evidence(),evidence())
    assert not holding_proxy(evidence(distance=.1),evidence())
    assert not holding_proxy(evidence(relative=(.1,0,0)),evidence())
    assert not holding_proxy(evidence(grasped=-1),evidence())


def test_roles_keep_subject_separate_from_unavailable_recipient():
    rack=dict(instance_id='rack',class_id=3)
    label=stage_label(rack,True,False,False,True,'bowl')
    assert label['role']==4 and label['subject']=='bowl' and label['target'] is None
    label=stage_label(rack,False,False,True,True,'bowl')
    assert label['role']==0 and label['target']['instance_id']=='bowl'
    label=stage_label(rack,False,False,False,False,'bowl')
    assert label['role']==2 and label['target']==rack
    assert stage_label(None,False,False,False,False,'bowl') is None


def test_gate_uses_predicted_role_and_never_emits_wrong_class_or_invalid_slot():
    pointer=torch.tensor([[2.,10.,20.,0.],[2.,10.,20.,0.]])
    classes=torch.tensor([[0,3,0],[0,3,0]])
    valid=torch.tensor([[True,True,False],[True,True,False]])
    role=torch.tensor([[10.,0.,0.,0.,0.],[0.,0.,0.,0.,10.]])
    logits=gated_logits(pointer,role,classes,valid)
    assert logits[0].argmax()==0
    assert logits[1].argmax()==3
    assert torch.isneginf(logits[0,1]) and torch.isneginf(logits[0,2])


def test_rejection_cannot_hide_full_recall_or_role_errors():
    from run import measure
    data=dict(roles=np.array([0,0,4]),positive=np.array([[1,0],[1,0],[0,0]],bool),
              visible=np.array([1,1,0],bool),rows=[dict(objects=[dict(class_id=0),dict(class_id=3)])]*3)
    before=measure(np.array([0,1,1]),[0,1,2],data)
    rejected=measure(np.array([8,8,8]),[0,1,2],data)
    assert before['wrong_role_boxes']==2 and before['non_bowl_box_on_acquire']==1
    assert rejected['selection_precision'] is None and rejected['visible_recall']==0
    assert rejected['supported_coverage']==0
