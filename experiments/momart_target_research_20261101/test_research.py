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


def test_robot_camera_projection_axes_and_observed_depth_distance():
    from robot_geometry import project_uv,box_geometry
    uv,valid=project_uv([1,0,0],90)
    assert valid;np.testing.assert_allclose(uv,[.5,.5])
    left,_=project_uv([1,1,0],90);assert left[0]<.5
    assert not project_uv([-1,0,0],90)[1]
    feature=box_geometry([0,0,1,1],np.array([[1/3]],np.float32),[1,0,0],90,3)
    assert feature.shape==(16,) and feature[0]<1e-6 and feature[-1]==1
    missing=box_geometry([0,0,1,1],np.zeros((1,1)),[1,0,0],90,3)
    assert np.isfinite(missing).all() and missing[-1]==0


def test_box_quality_transfers_removed_mass_to_null_without_switching():
    from refine import multiply_quality
    p=np.zeros((2,9),np.float32);p[:,0]=.7;p[:,1]=.2;p[:,8]=.1
    q=np.ones((2,8),np.float32);q[0,0]=0;q[1,1]=.25
    out=multiply_quality(p,q)
    np.testing.assert_allclose(out.sum(-1),1)
    assert out[0].argmax()==8 and out[0,1]==p[0,1]
    np.testing.assert_allclose(out[1,8],.25)


def test_calibration_does_not_pass_by_rejecting_an_entire_role():
    from refine import calibrate_v2
    n=20;rows=[dict(episode_id='episode',objects=[dict(class_id=0)],label=dict(target=dict(class_id=0))) for _ in range(n)]
    data=dict(rows=rows,y=np.zeros(n,int),roles=np.r_[np.zeros(16,int),np.full(4,2)],visible=np.ones(n,bool),positive=np.zeros((n,8),bool))
    data['positive'][:,0]=True
    p=np.zeros((n,9));p[:16,0]=.99;p[:16,8]=.01;p[16:,0]=.4;p[16:,8]=.6
    c=calibrate_v2(p,np.arange(n),data,.6)
    assert c['overall_precision90_feasible'] and not c['joint_goal_met']
    assert not c['critical_roles_covered']


def test_soft_role_filter_is_causal_and_ignores_private_labels():
    from temporal import soft_role_filter
    n=4;candidate=np.zeros((n,8,139),np.float32);candidate[:,0,133]=1;candidate[:,1,136]=1
    data=dict(rows=[dict(episode_id='e',step=8*i,task='task') for i in range(n)],tasks=['task'],
        valid=np.array([[True,True]+[False]*6]*n),candidates=candidate)
    p=np.zeros((n,9));p[:,0]=.45;p[:,1]=.45;p[:,8]=.1
    emissions=np.full((n,5),.025);emissions[:,2]=.9
    rates=np.full((5,5),.01);np.fill_diagonal(rates,-.04);priors=np.full((1,5),.2)
    full=soft_role_filter(p,emissions,data,rates,priors)
    prefix=dict(data,rows=data['rows'][:2],valid=data['valid'][:2],candidates=candidate[:2])
    short=soft_role_filter(p[:2],emissions[:2],prefix,rates,priors)
    np.testing.assert_allclose(full[:2],short)
    data['roles']=np.zeros(n);data['y']=np.full(n,8)
    np.testing.assert_allclose(full,soft_role_filter(p,emissions,data,rates,priors))
    assert full[0].argmax()==1 and np.allclose(full.sum(-1),1)


def test_mask_geometry_rejects_background_inside_loose_box():
    from mask_surface import surface_features
    depth=np.full((3,3),1/3,np.float32);prob=np.zeros((3,3),np.float32)
    # Center background has depth equal to EEF; foreground mask is away from center.
    prob[0,0]=1
    value=surface_features([0,0,1,1],depth,prob,[1,0,0],[.5,.5],90,3)
    assert value.shape==(23,) and value[0]>.1 and value[-1]==1
    assert value[-2]==0  # no object mask at projected EEF


def test_role_recall_floor_prevents_aggregate_gain_hiding_stage_regression():
    from benchmark_dense import calibrate_v3
    n=40;roles=np.r_[np.zeros(20,int),np.full(20,2)]
    rows=[dict(episode_id='e',objects=[dict(class_id=0 if role==0 else 3)],label=dict(target=dict(class_id=0 if role==0 else 3))) for role in roles]
    data=dict(rows=rows,y=np.zeros(n,int),roles=roles,visible=np.ones(n,bool),positive=np.zeros((n,8),bool));data['positive'][:,0]=True
    p=np.zeros((n,9));p[:30,0]=1;p[30:,8]=1
    baseline=dict(metrics=dict(recall=.6),roles={'0':dict(visible=20,recall=.2),'2':dict(visible=20,recall=1.)})
    cal=calibrate_v3(p,np.arange(n),data,baseline)
    assert cal['metrics']['precision']==1 and cal['metrics']['recall']>.6
    assert cal['critical_roles_covered'] and not cal['role_recall_retained'] and not cal['joint_goal_met']


def test_mask_feature_views_pool_only_valid_candidates():
    from mask_benchmark import mask_view
    data=dict(candidates=np.zeros((1,8,139),np.float32),context=np.zeros((1,280),np.float32),
        geometry_candidates=np.zeros((1,8,16),np.float32),geometry_context=np.zeros((1,14),np.float32),
        mask_candidates=np.ones((1,8,23),np.float32),valid=np.array([[True]+[False]*7]))
    data['mask_candidates'][:,1:]=100
    candidate,context=mask_view(data,'mask_small')
    assert candidate.shape==(1,8,34) and context.shape==(1,72)
    np.testing.assert_array_equal(context[0,-23:],np.ones(23))


def test_sequence_diagnostics_separate_chatter_from_gold_switches_and_nulls():
    from sequence_metrics import sequence_metrics
    rows=[]
    for i,target in enumerate(['bowl','bowl','rack','rack']):
        rows.append(dict(episode_id='e',step=i*8,objects=[dict(class_id=0),dict(class_id=3)],label=dict(target=dict(instance_id=target,class_id=0 if target=='bowl' else 3))))
    positive=np.zeros((4,8),bool);positive[:2,0]=True;positive[2:,1]=True
    data=dict(rows=rows,positive=positive)
    m=sequence_metrics(np.array([0,1,1,8]),data,range(4))
    assert m['stable_class_switches']==1 and m['gold_subject_changes']==1 and m['correct_box_at_change']==1
    assert m['stable_correct_to_null']==1 and m['stable_both_selected']==1


def test_public_runtime_returns_one_box_or_null_and_rejects_stale_inputs(tmp_path):
    import json
    import pytest
    from runtime import FrameFeatures,SingleBoxSelector
    previous=tmp_path/'momart_target_selector_20261101';previous.mkdir();(previous/'features.json').write_text(json.dumps(dict(tasks=['t0','t1','t2','t3'])))
    net=Pointer(139,419,width=16,kind='linear').eval()
    for value in net.parameters():value.data.zero_()
    net.null[-1].bias.data.fill_(-4.)
    torch.save(dict(spec=dict(kind='linear',view='base',width=16),state=net.state_dict(),normalizer=dict(cm=np.zeros(139,np.float32),cs=np.ones(139,np.float32),xm=np.zeros(419,np.float32),xs=np.ones(419,np.float32))),tmp_path/'model.pt')
    selection=tmp_path/'selected.json';selection.write_text(json.dumps(dict(result=dict(component=dict(kind='parent',name='model',screen=str(tmp_path)),calibration=dict(threshold=0.,margin=0.)))))
    runtime=SingleBoxSelector(selection,tmp_path);c=np.zeros((8,139),np.float32);c[0,128:132]=[.1,.2,.3,.4];c[0,133]=1
    x=np.zeros(280,np.float32);x[-4]=1;valid=np.array([True]+[False]*7);frame=FrameFeatures(c,x,valid,'t0')
    result=runtime.select(frame,'first',0);assert result['valid'] and result['slot']==0
    np.testing.assert_array_equal(result['box'],c[0,128:132]);result['probability'][0]=123
    assert runtime.select(frame,'first',0)['probability'][0]<1
    with pytest.raises(ValueError,match='increase'):runtime.select(frame,'first',-1)
    changed=FrameFeatures(c,x.copy(),valid,'t0');changed.context[0]=1
    with pytest.raises(ValueError,match='different public inputs'):runtime.select(changed,'first',0)
    empty=FrameFeatures(c,x,np.zeros(8,bool),'t0');missing=runtime.select(empty,'first',8)
    assert not missing['valid'] and missing['box'] is None and missing['slot']==8
    assert runtime.select(frame,'next_episode',0)['slot']==0
    assert not({'label','positive','future','actions'}&set(FrameFeatures.__dataclass_fields__))
