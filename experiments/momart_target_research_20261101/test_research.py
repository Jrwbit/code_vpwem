import numpy as np
import pytest
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
    selection=tmp_path/'selected.json';selection.write_text(json.dumps(dict(name='test_model',result=dict(component=dict(kind='parent',name='model',screen=str(tmp_path)),calibration=dict(threshold=0.,margin=0.,class_thresholds={'0':0.})))))
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
    from export_runtime_bundle import export_bundle,sha
    export_bundle(selection,tmp_path,tmp_path/'bundle')
    (tmp_path/'bundle').rename(tmp_path/'moved_bundle');bundle=tmp_path/'moved_bundle'
    assert str(tmp_path) not in (bundle/'selected.json').read_text()
    relocated=SingleBoxSelector(bundle/'selected.json',bundle).select(frame,'new_episode',0)
    np.testing.assert_array_equal(relocated['probability'],runtime.select(frame,'next_episode',0)['probability'])
    manifest=json.loads((bundle/'manifest.json').read_text())
    assert manifest['status']=='research_candidate_not_promoted'
    assert json.loads((bundle/'selected.json').read_text())['result']['calibration']['class_thresholds']=={'0':0.}
    assert all(sha(bundle/name)==digest for name,digest in manifest['files'].items())


def test_priority_gate_does_not_call_second_confirmation_a_complete_success():
    from expanded import priority_gate
    baseline=dict(metrics=dict(wrong_acquire=5,wrong_carry=8,acquire_samples=86),
        sequence=dict(stable_class_switch_rate=24/430,gold_subject_changes=8,stale_previous_class_at_change=1,correct_box_at_change=4))
    aggregate_winner=dict(wrong_acquire=8,wrong_carry=1)
    sequence=dict(stable_class_switch_rate=13/434,stale_previous_class_at_change=3,correct_box_at_change=3)
    gate=priority_gate(aggregate_winner,sequence,baseline)
    assert not gate['priority_errors_improved'] and not gate['switching_retained']
    assert gate['acquisition_error_limit']==4 and gate['transition_gate_enabled']


def test_raw_spatial_frontend_rejects_future_history_and_unavailable_visuals():
    import pytest
    from spatial_frontend import SpatialFrontEnd,RobotObservation
    from runtime import FrameFeatures,view
    current=RobotObservation(np.zeros((90,120,3),np.uint8),np.zeros((90,120)),np.zeros(10),0.,'e',8,'t')
    past=RobotObservation(None,None,np.zeros(10),0.,'e',0,'t');SpatialFrontEnd.validate_pair(current,past)
    past.step=9
    with pytest.raises(ValueError,match='eight sensor steps'):SpatialFrontEnd.validate_pair(current,past)
    frame=FrameFeatures(np.zeros((8,139),np.float32),np.zeros(280,np.float32),np.zeros(8,bool),'t',geometry_context=np.zeros(14),geometry_candidates=np.zeros((8,16)),visual_features_available=False)
    assert view(frame,'spatial')[0].shape==(8,27)
    with pytest.raises(ValueError,match='visual ROI'):view(frame,'base')


def test_availability_factorization_preserves_identity_and_never_fabricates_empty_candidates():
    from availability_features import replace_availability
    rng=np.random.default_rng(4);p=np.zeros((101,9),np.float32)
    p[:100,:8]=rng.dirichlet(np.ones(8),100).astype(np.float32);p[100,8]=1
    r=replace_availability(p,np.ones(101),0.,4.)
    assert (r>=0).all();np.testing.assert_allclose(r.sum(-1),1,atol=1e-6)
    np.testing.assert_array_equal(r[:100,:8].argmax(-1),p[:100,:8].argmax(-1))
    assert r[100,8]==1 and not r[100,:8].any()
    np.testing.assert_array_equal(replace_availability(p,np.zeros(101),0.,0.),p)


def test_third_confirmation_priority_improvement_does_not_hide_a_lost_switch():
    from confirm import evaluation_gate
    roles={'0':dict(visible=81,correct=78,recall=78/81),'4':dict(visible=189,correct=189,recall=1.)}
    base_roles={'0':dict(visible=81,correct=73,recall=73/81),'4':dict(visible=189,correct=173,recall=173/189)}
    base_m=dict(recall=357/430,wrong_acquire=7,wrong_carry=17,acquire_samples=81)
    primary=dict(accepted=dict(precision=388/410,recall=388/430,wrong_acquire=1,wrong_carry=0,acquire_samples=81),roles=roles,
        sequence=dict(stable_class_switch_rate=9/361,gold_subject_changes=5,correct_box_at_change=3,stale_previous_class_at_change=0))
    baseline=dict(raw=base_m,accepted=base_m,roles=base_roles,sequence=dict(stable_class_switch_rate=34/386,gold_subject_changes=5,correct_box_at_change=4,stale_previous_class_at_change=0))
    gate=evaluation_gate(primary,baseline,4)
    assert gate['priority_gate']['priority_errors_improved'] and not gate['priority_gate']['switching_retained']
    assert not gate['joint_goal_met']


def test_support_audit_separates_carrier_motion_from_bilateral_grasp_evidence():
    from audit_support_corpus import facts
    before=dict(supports={'rack':dict(position=[0,0,0],orientation=[0,0,0,1],bowl_relative=[0,0,.17],hand_links=[1,2])})
    current=dict(finger_distances={'r_gripper_finger_link':.0001,'l_gripper_finger_link':.031},
        supports={'rack':dict(position=[.05,0,0],orientation=[0,0,0,1],bowl_relative=[0,0,.17],hand_links=[1,2])})
    assert facts(current,before)['carrier_motion_with_unilateral_grip']
    current['finger_distances']['l_gripper_finger_link']=.0002
    assert facts(current,before)['bilateral_bowl_5mm'] and not facts(current,before)['carrier_motion_with_unilateral_grip']


def test_class_query_cannot_fall_back_to_rack_or_invalid_bowl():
    from query_grounding import ground_query
    p=np.zeros((4,9),np.float32);p[:,0]=.8;p[:,1]=.1;p[:,8]=.1
    classes=np.full((4,8),-1);classes[:,:2]=[3,0]
    valid=np.zeros((4,8),bool);valid[:,:2]=True;valid[1,1]=False
    query=np.array([0,0,-1,3])
    result=ground_query(p,classes,valid,query,keep_null=False)
    np.testing.assert_array_equal(result.argmax(-1),[1,8,8,0])
    assert result[0,0]==0 and result[1,0]==0
    np.testing.assert_allclose(result.sum(-1),1)


def test_class_query_preserves_instance_ranking_and_optional_null():
    from query_grounding import ground_query
    p=np.array([[.15,.05,.1,0,0,0,0,0,.7]])
    cls=np.array([[0,0,3,-1,-1,-1,-1,-1]])
    valid=cls>=0
    masked=ground_query(p,cls,valid,np.array([0]))
    assert masked.argmax(-1)[0]==8
    np.testing.assert_allclose(masked[0,0]/masked[0,1],3)
    assert ground_query(p,cls,valid,np.array([0]),keep_null=False).argmax(-1)[0]==0


def test_failure_decomposition_keeps_missing_proposals_separate_from_intent():
    from diagnose_intent import failure_bucket
    data=dict(positive=np.array([[False]*8,[True]+[False]*7,[True]+[False]*7]),
        rows=[dict(label=dict(target=dict(class_id=0)),objects=[dict(class_id=3),dict(class_id=0)]) for _ in range(3)])
    assert failure_bucket(0,[0,0,0],data)=='no_matching_instance_proposal'
    assert failure_bucket(1,[0,8,0],data)=='rejected_matching_proposal'
    assert failure_bucket(2,[0,0,0],data)=='correct'


def test_autonomous_query_history_is_prefix_causal_and_episode_local():
    from hard_query import query_beliefs,hard_query_distribution
    rows=[dict(episode_id=ep,step=t,task='task') for ep,t in [('a',0),('b',0),('a',8),('a',16)]]
    emissions=np.array([[.8,.05,.05,.05,.05],[.05,.05,.8,.05,.05],[.4,.1,.3,.1,.1],[.05,.05,.8,.05,.05]])
    rates=np.full((5,5),.02);np.fill_diagonal(rates,-.08)
    priors=np.full((1,5),.2)
    full=query_beliefs(emissions,rows,['task'],rates,priors,True)
    np.testing.assert_allclose(full[:3],query_beliefs(emissions[:3],rows[:3],['task'],rates,priors,True))
    np.testing.assert_allclose(full[[0,2,3]],query_beliefs(emissions[[0,2,3]],[rows[i] for i in [0,2,3]],['task'],rates,priors,True))
    p=np.zeros((4,9));p[:,0]=.8;p[:,1]=.1;p[:,8]=.1
    cls=np.full((4,8),-1);cls[:,:2]=[3,0];valid=cls>=0
    for mode in ['parent','product']:
        out=hard_query_distribution(p,full,cls,valid,mode)
        np.testing.assert_allclose(out.sum(-1),1)
        assert out[0,0]==0  # A bowl query cannot silently return the more likely rack.


def test_robot_history_does_not_use_current_or_future_commands():
    from robot_history import command_features,finger_features,history_at
    commands=np.arange(160,dtype=np.float32).reshape(16,10)
    q=np.arange(32,dtype=np.float32).reshape(16,2)
    t=8;finger,before=history_at(q,commands,t)
    commands[t:]=-999;q[t+1:]=-999
    changed_finger,changed=history_at(q,commands,t)
    np.testing.assert_array_equal(finger,changed_finger)
    np.testing.assert_array_equal(before,changed)
    np.testing.assert_array_equal(before[:10],commands[t-1])
    np.testing.assert_array_equal(command_features(np.zeros((0,10))),np.zeros(31))
    np.testing.assert_allclose(finger_features([.01,.02],[.01,.02],0),[.01,.02,0,0,.015,0])


def test_robot_history_join_uses_physical_frame_not_reused_corpus_id(tmp_path):
    import json
    from robot_history import attach_history,frame_key
    rows=[dict(source='/tmp/source.hdf5',demo='demo_1',step=t,id='reused') for t in [8,0]]
    keys=[frame_key(row) for row in rows[::-1]]
    path=tmp_path/'history.npz';np.savez(path,gripper=np.array([[1]*6,[2]*6]),actions=np.array([[3]*31,[4]*31]))
    path.with_suffix('.json').write_text(json.dumps(dict(protocol='causal_robot_history_v1',keys=keys,actions_strictly_before_observation=True)))
    result=attach_history(dict(rows=rows),path)
    np.testing.assert_array_equal(result['history_gripper'][:,0],[2,1])
    np.testing.assert_array_equal(result['history_actions'][:,0],[4,3])


def test_audit_confidence_changes_only_train_weights_and_never_labels():
    from supervision_confidence import downweight_training
    data=dict(weights=np.array([.95,.9,.8]),y=np.array([0,1,2]),positive=np.eye(3,dtype=bool))
    out=downweight_training(data,[0,1],np.array([True,False,True]),.25)
    np.testing.assert_allclose(out['weights'],[.2375,.9,.8])
    np.testing.assert_allclose(data['weights'],[.95,.9,.8])
    assert out['y'] is data['y'] and out['positive'] is data['positive']
    import pytest
    with pytest.raises(ValueError):downweight_training(data,[0],np.array([True,False,True]),0)


def test_proposal_association_is_one_to_one_and_class_constrained():
    from candidate_history import associate
    a=np.zeros((8,139),np.float32);a[0,0]=1;a[1,1]=1
    a[:2,128:132]=[[.1,.1,.3,.3],[.6,.6,.8,.8]];a[:2,136]=1
    valid=np.array([True,True]+[False]*6)
    b=a.copy();b[:2]=a[[1,0]]
    np.testing.assert_array_equal(associate(a,b,valid,valid)[:2],[1,0])
    b[1,136]=0;b[1,133]=1
    assert associate(a,b,valid,valid)[0]==-1


def test_candidate_history_is_prefix_causal_label_free_and_episode_local():
    from candidate_history import add_candidate_history
    c=np.zeros((4,8,139),np.float32);c[:,0,0]=1;c[:,0,128:132]=[.1,.1,.3,.3];c[:,0,133]=1
    c[:,0,132]=.9;c[2,0,128:132]+=.01
    valid=np.zeros((4,8),bool);valid[:,0]=True
    g=np.ones((4,8,16),np.float32)
    rows=[dict(episode_id=ep,step=t) for ep,t in [('a',0),('b',0),('a',8),('a',40)]]
    data=dict(rows=rows,candidates=c,valid=valid,geometry_candidates=g,y=np.array([0,8,0,0]))
    full=add_candidate_history(data)['candidate_history']
    prefix={k:v[:3] for k,v in data.items()}
    np.testing.assert_array_equal(full[:3],add_candidate_history(prefix)['candidate_history'])
    assert not full[0].any() and not full[1].any() and not full[3].any()
    assert full[2,0,0]==1
    data['y'][:]=8
    np.testing.assert_array_equal(full,add_candidate_history(data)['candidate_history'])


def test_policy_observations_use_only_two_recent_public_sensor_frames():
    from policy_features import prepare_observations
    obs={k:np.zeros((4,90,120,c),np.uint8 if c==3 else np.float32) for k,c in [('rgb',3),('rgb_wrist',3),('depth',1),('depth_wrist',1)]}
    obs.update(proprio=np.zeros((4,10)),scan=np.zeros((4,1,100)))
    obs['rgb'][1]=255;obs['rgb'][2]=128
    first=prepare_observations(obs,2)
    obs['rgb'][3]=0;obs['proprio'][3]=1000
    second=prepare_observations(obs,2)
    for key in first:np.testing.assert_array_equal(first[key],second[key])
    assert first['rgb'].shape==(2,3,90,120) and first['rgb'][0].mean()==1
    np.testing.assert_array_equal(prepare_observations(obs,0)['rgb'][0],prepare_observations(obs,0)['rgb'][1])


def test_plan_targets_are_label_only_and_end_padding_is_masked():
    from plan_labels import future_commands
    from plan_model import plan_loss,ActionPlanPointer
    commands=np.arange(30,dtype=np.float32).reshape(3,10)
    target,valid=future_commands(commands,2)
    np.testing.assert_array_equal(target[0],commands[2]);assert valid.sum()==1
    p=torch.zeros(1,8,10,requires_grad=True);y=torch.tensor(target[None]);mask=torch.tensor(valid[None])
    a=plan_loss(p,y,mask);y[:,1:]=1e6
    torch.testing.assert_close(a,plan_loss(p,y,mask))
    a.backward();assert p.grad[:,1:].abs().sum()==0
    net=ActionPlanPointer(4,5).eval()
    logits,plan=net(torch.zeros(2,8,4),torch.zeros(2,5),torch.tensor([[True]+[False]*7]*2))
    assert logits.shape==(2,9) and plan.shape==(2,8,10)
    assert torch.isneginf(logits[:,1:8]).all() and torch.isfinite(logits[:,[0,8]]).all()


def test_growth_sampling_reserves_confirmations_and_splits_only_episodes():
    from freeze_growth import select_growth
    pool=[f'demo_{i}' for i in range(10)];excluded={'task/demo_1','task/demo_8'}
    chosen=select_growth(pool,excluded,'task',4,2,np.random.default_rng(7))
    assert len(chosen)==len({demo for demo,_ in chosen})==6
    assert all('task/'+demo not in excluded for demo,_ in chosen)
    assert [split for _,split in chosen].count('train')==4
    assert [split for _,split in chosen].count('dev')==2
    assert chosen==select_growth(pool,excluded,'task',4,2,np.random.default_rng(7))


def test_growth_partitions_keep_old_train_and_new_development_disjoint():
    from growth_data import growth_partitions
    rows=[dict(tag='old',split=split,cohort=cohort) for split,cohort in [('train','dense'),('dev','dense'),('train','growth'),('dev','growth')]]
    data=dict(rows=rows,y=np.zeros(4,dtype=int))
    split=growth_partitions(data)
    np.testing.assert_array_equal(split['old_train'],[0]);np.testing.assert_array_equal(split['old_dev'],[1])
    np.testing.assert_array_equal(split['new_dev'],[3]);assert not set(split['train'])&set(split['dev'])


def test_growth_loader_excludes_examined_and_reserved_confirmation_episodes():
    import pytest
    from growth_data import require_growth_isolation
    old={'old_train','old_dev'};reserved={'examined3','examined4','reserved5'};exclusions=old|reserved
    require_growth_isolation(old,{'new_train','new_dev'},reserved,exclusions)
    for contaminated in ['old_dev','examined3','examined4','reserved5']:
        with pytest.raises(ValueError):require_growth_isolation(old,{contaminated},reserved,exclusions)
    with pytest.raises(ValueError):require_growth_isolation(old,{'new_train'},reserved,{'old_train'})


def test_growth_confirmation_freeze_rejects_overlap_or_already_examined_holdout(tmp_path):
    import json
    import pytest
    from types import SimpleNamespace
    from freeze_growth_confirmation import main
    selected=tmp_path/'selected.json';control=tmp_path/'control.json';audit=tmp_path/'audit.json'
    selected.write_text(json.dumps(dict(name='winner',result=dict(joint_growth_goal_met=True,subgroup_gates={'old':dict(joint_goal_met=True),'new':dict(joint_goal_met=True)}))))
    control.write_text(json.dumps(dict(name='control',result=dict(calibration=dict(threshold=0,margin=.2)))))
    audit.write_text(json.dumps(dict(episodes=dict(train=['seen']))))
    holdout=tmp_path/'holdout';holdout.mkdir()
    manifest=holdout/'manifest.json'
    manifest.write_text(json.dumps(dict(episodes=[dict(episode_id='seen',source_partition='nonvalid')])) )
    args=SimpleNamespace(selection=selected,control=control,training_audit=audit,holdout=holdout,output=tmp_path/'frozen')
    with pytest.raises(ValueError,match='overlaps'):main(args)
    manifest.write_text(json.dumps(dict(episodes=[dict(episode_id='new',source_partition='nonvalid')])) )
    examined=holdout/'annotations_all.json';examined.write_text('{}')
    with pytest.raises(ValueError,match='untouched'):main(args)
    assert not args.output.exists()
    examined.unlink();main(args)
    frozen=json.loads((args.output/'selected.json').read_text())
    assert frozen['control_calibration']==dict(threshold=0,margin=.2)
    assert frozen['holdout_evaluated'] is False


def test_transition_emphasis_is_train_only_episode_local_and_unknown_breaks_window():
    from transition_weights import training_weights
    labels=[dict(target=dict(instance_id=k)) if k else None for k in ['a','b',None,'b','c','c','other']]
    data=dict(rows=[dict(episode_id='train' if i<6 else 'dev',step=i*8,label=label) for i,label in enumerate(labels)],weights=np.ones(7),roles=np.zeros(7,int))
    weights,info=training_weights(data,[0,1,3,4,5],4,2)
    np.testing.assert_array_equal(weights,[2,8,1,2,8,8,1])
    np.testing.assert_array_equal(data['weights'],np.ones(7))
    assert info['transition_events']==2 and info['window_frames']==3
    import json
    json.dumps(info)
    unchanged,_=training_weights(data,[0,1,3,4,5])
    np.testing.assert_array_equal(unchanged,data['weights'])


def test_confirmation_sampling_excludes_retained_regressions_from_fitted_audit():
    from freeze_confirmation import excluded_episode_ids
    excluded=excluded_episode_ids([dict(episodes=[dict(episode_id='dense_train')],excluded_episodes=['earlier_examined'])],[dict(episodes=dict(train=['dense_train'],regression=['retained_old_test'],dev=['dense_dev']))])
    assert excluded=={'dense_train','retained_old_test','dense_dev','earlier_examined'}


def test_growth_screen_join_checks_episode_identity_not_just_equal_counts():
    import pytest
    from select_growth_final import require_same_development
    a=dict(episodes=dict(train=['a'],dev=['b']),counts=dict(train=10,dev=10))
    require_same_development(a,a)
    with pytest.raises(ValueError):require_same_development(dict(a,episodes=dict(train=['a'],dev=['other'])),a)


def test_incumbent_gate_compares_frozen_accepted_recall_instead_of_raw_recall():
    from confirm import incumbent_evaluation_gate
    role={'0':dict(visible=20,correct=19,recall=.95)}
    sequence=dict(stable_class_switch_rate=0,gold_subject_changes=0,correct_box_at_change=0,stale_previous_class_at_change=0)
    primary=dict(accepted=dict(precision=.95,recall=.95,wrong_acquire=0,wrong_carry=0,acquire_samples=20),roles=role,sequence=sequence)
    control=dict(accepted=dict(precision=.95,recall=.9,wrong_acquire=1,wrong_carry=0,acquire_samples=20),raw=dict(recall=.99),roles=role,sequence=sequence)
    assert incumbent_evaluation_gate(primary,control,4)['joint_goal_met']
    control['accepted']['recall']=.96;control['raw']['recall']=.5
    assert not incumbent_evaluation_gate(primary,control,4)['joint_goal_met']


def test_switch_diagnosis_does_not_bridge_unknown_or_excessive_gaps():
    from diagnose_intent import subject_changes
    subjects=['a',None,'b','a','b'];steps=[0,8,16,24,48]
    rows=[]
    for i,(target,step) in enumerate(zip(subjects,steps)):
        rows.append(dict(episode_id='e',id=str(i),demo='demo',step=step,label=dict(target=dict(instance_id=target,class_id=int(target=='b'))) if target else None,objects=[dict(class_id=0)]))
    data=dict(rows=rows,positive=np.ones((5,8),bool));changes=subject_changes(data,np.zeros(5,int),np.zeros(5,bool))
    assert len(changes)==1 and changes[0]['step']==24 and changes[0]['outcome']=='correct'


def test_class_rejection_uses_predicted_classes_and_never_turns_null_into_a_box():
    import pytest
    from calibration import apply_calibration
    p=np.zeros((3,9),np.float32);p[:2,0]=.55;p[:2,1]=.45;p[2,8]=.9;p[2,0]=.1
    classes=np.zeros((3,8),int);classes[1,0]=3
    cal=dict(threshold=.6,margin=0,class_thresholds={'0':.5})
    np.testing.assert_array_equal(apply_calibration(p,cal,classes),[0,8,8])
    np.testing.assert_array_equal(apply_calibration(p,dict(threshold=.6,margin=0)),[8,8,8])
    with pytest.raises(ValueError,match='public proposal'):apply_calibration(p,cal)
    np.testing.assert_array_equal(apply_calibration(p,dict(cal,margin=.2,class_margins={'0':0.}),classes),[0,8,8])


def test_review_sampling_includes_every_change_and_blinds_labels_with_matched_controls():
    from build_switch_review import select_cases,reviewer_entry
    rows=[]
    for i,target in enumerate(['a']*10+['b']*10+[None]+['c']*9):
        label=dict(target=dict(instance_id=target,class_id={'a':3,'b':0,'c':2}[target])) if target else None
        rows.append(dict(source='/tmp/source.hdf5',demo='demo',episode_id='task/demo',task='task',step=i*8,label=label))
    groups={'dev':(rows,np.flatnonzero([r['label'] is not None for r in rows]))}
    selected=select_cases(groups,7);again=select_cases(groups,7)
    assert [(x['row']['step'],x['kind']) for x in selected]==[(x['row']['step'],x['kind']) for x in again]
    assert len(selected)==2
    assert [x['row']['step'] for x in selected if x['kind']=='proxy_change']==[80]
    entry=reviewer_entry('R001',dict(rows[1]),[0,0,8,16,24])
    assert entry['views']==['main']
    assert entry['annotations'][0]['relative_steps']==-8
    assert all(x['status']=='pending' and x['target_class'] is None for x in entry['annotations'])
    assert not {'label','proxy_class','kind','cohort','predicted_class'}&set(entry)
    assert select_cases({'gap':([rows[0],dict(rows[10],step=40)],[0,1])})==[]


def test_hard_negative_ranking_masks_unknown_unavailable_and_other_correct_boxes():
    from models import hard_negative_ranking_loss
    logits=torch.tensor([[1.,9.,2.,99.,3.],[1.,2.,3.,4.,5.],[1.,2.,3.,4.,5.]],requires_grad=True)
    valid=torch.tensor([[True,True,True,False]]*3)
    positive=torch.tensor([[True,True,False,False],[False]*4,[False]*4])
    labels=torch.tensor([0,-100,4]);weights=torch.tensor([2.,100.,100.])
    loss=hard_negative_ranking_loss(logits,labels,positive,weights,valid)
    assert torch.allclose(loss,torch.nn.functional.softplus(torch.tensor(2.)))
    loss.backward()
    assert logits.grad[0,0]<0 and logits.grad[0,2]>0
    assert torch.count_nonzero(logits.grad)==2
    empty=hard_negative_ranking_loss(logits,torch.tensor([-100,-100,4]),positive,weights,valid)
    assert torch.isfinite(empty) and empty==0
    all_correct=hard_negative_ranking_loss(logits[:1],labels[:1],valid[:1],weights[:1],valid[:1])
    assert torch.isfinite(all_correct) and all_correct==0


def test_training_consistency_does_not_bridge_unknown_unavailable_split_or_identity():
    from training_consistency import stable_training_predecessors
    rows=[dict(episode_id='a',step=i*8,label=dict(target=dict(instance_id=k)) if k else None) for i,k in enumerate(['a','a',None,'a','a','b','b','b','b'])]
    positive=np.ones((9,8),bool);positive[6]=False
    data=dict(rows=rows,positive=positive)
    previous=stable_training_predecessors(data,[0,1,2,3,4,5,6,8])
    np.testing.assert_array_equal(previous,[-1,0,-1,-1,3,-1,-1,-1,-1])
    rows[4]['step']=100
    assert stable_training_predecessors(data,[3,4])[4]==-1


def test_class_consistency_is_slot_invariant_and_includes_null_with_finite_gradients():
    from training_consistency import class_consistency_loss
    a=torch.tensor([[2.,1.,float('-inf'),0.]],requires_grad=True)
    b=torch.tensor([[1.,2.,float('-inf'),0.]],requires_grad=True)
    ca=torch.tensor([[0,3,5]]);cb=torch.tensor([[3,0,2]])
    loss=class_consistency_loss(a,b,ca,cb,torch.ones(1))
    assert torch.allclose(loss,torch.tensor(0.),atol=1e-7)
    shifted=torch.tensor([[1.,2.,float('-inf'),4.]],requires_grad=True)
    changed=class_consistency_loss(a,shifted,ca,cb,torch.ones(1))
    assert 0<float(changed.detach())<.7
    changed.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(shifted.grad).all()
    assert torch.isfinite(class_consistency_loss(a[:0],b[:0],ca[:0],cb[:0],torch.ones(0)))


def test_epoch_snapshot_callback_preserves_training_and_captures_real_weights():
    from benchmark_growth import train_model
    rng=np.random.default_rng(73);n=8
    raw=rng.normal(size=(n,8,3)).astype(np.float32);context=rng.normal(size=(n,4)).astype(np.float32)
    valid=np.zeros((n,8),bool);valid[:,:2]=True
    rows=[dict(episode_id=str(i//2),objects=[dict(class_id=0),dict(class_id=3)],label=dict(target=dict(class_id=0))) for i in range(n)]
    positive=np.zeros((n,8),bool);positive[:,0]=True
    data=dict(rows=rows,valid=valid,y=np.zeros(n,np.int64),positive=positive,weights=np.ones(n,np.float32),visible=np.ones(n,bool),roles=np.zeros(n,np.int64))
    train=np.arange(4);dev=np.arange(4,8)
    plain,reference,history=train_model(data,(raw,context),train,dev,17,'cpu')
    saved={}
    def snapshot(epoch,net,probability,normalizer):
        saved[epoch]=(dict(state={k:v.detach().clone() for k,v in net.state_dict().items()},normalizer=normalizer),probability.copy())
    replay,checkpoint,observed=train_model(data,(raw,context),train,dev,17,'cpu',epoch_callback=snapshot)
    assert checkpoint['best_epoch']==reference['best_epoch']
    assert observed['history']==history['history']
    for key in reference['state']:assert torch.equal(reference['state'][key],checkpoint['state'][key])
    np.testing.assert_array_equal(plain,replay)
    selected,probability=saved[checkpoint['best_epoch']]
    for key in selected['state']:assert torch.equal(selected['state'][key],checkpoint['state'][key])
    np.testing.assert_array_equal(probability,replay)
    assert len(saved)==len(history['history'])


def test_current_checkpoint_retention_rejects_priority_and_temporal_regressions():
    from checkpoint_selection import current_retention_gate,checkpoint_merit
    import copy
    current=dict(accepted=dict(correct=100,wrong_acquire=2,wrong_carry=0),sequence=dict(stable_class_switch_rate=.01,correct_box_at_change=3,stale_previous_class_at_change=0),roles={'0':dict(visible=20,recall=.9)})
    assert current_retention_gate(current,current)['joint_goal_met']
    for section,key,value in [('accepted','correct',99),('accepted','wrong_acquire',3),('accepted','wrong_carry',1),('sequence','stable_class_switch_rate',.02),('sequence','correct_box_at_change',2),('sequence','stale_previous_class_at_change',1)]:
        candidate=copy.deepcopy(current);candidate[section][key]=value
        assert not current_retention_gate(candidate,current)['joint_goal_met']
    candidate=copy.deepcopy(current);candidate['roles']['0']['recall']=.84
    assert not current_retention_gate(candidate,current)['joint_goal_met']
    def point(eligible,recall,f1,acquire,rate):
        return dict(joint_growth_goal_met=eligible,calibration=dict(metrics=dict(recall=recall,f1=f1,wrong_acquire=acquire,wrong_carry=0,precision=.95),sequence=dict(stable_class_switch_rate=rate,correct_box_at_change=3)))
    good=point(True,.9,.91,2,.01);high_f1_failure=point(False,.99,.99,0,0.)
    assert checkpoint_merit(good)>checkpoint_merit(high_f1_failure)
    assert checkpoint_merit(point(True,.91,.9,2,.01))>checkpoint_merit(good)
    assert checkpoint_merit(point(True,.9,.9,1,.01))>checkpoint_merit(good)
    assert checkpoint_merit(point(True,.9,.9,2,.005))>checkpoint_merit(good)


def test_retention_freeze_binds_complete_comparator_and_all_dev_groups():
    import pytest
    from freeze_growth_confirmation import validate_retention_reference,component_weights
    import copy
    value=dict(current_retention={k:dict(joint_goal_met=True) for k in ['dev','old_dev','new_dev']})
    parent=dict(kind='parent',screen='/tmp/models',name='selector')
    registered=dict(component=dict(kind='availability',parent=parent),calibration=dict(threshold=.6,margin=0.,metrics={'correct':10}))
    retention=dict(result=dict(component=registered['component'],calibration=dict(threshold=.6,margin=0.,metrics={'correct':11})))
    validate_retention_reference(value,registered,retention)
    changed=copy.deepcopy(retention);changed['result']['calibration']['threshold']=.5
    with pytest.raises(ValueError,match='differs'):validate_retention_reference(value,registered,changed)
    bad=copy.deepcopy(value);del bad['current_retention']['old_dev']
    with pytest.raises(ValueError,match='all registered'):validate_retention_reference(bad,registered,retention)
    assert [str(p) for p in component_weights(dict(kind='blend',members=[registered['component']]))]==['/tmp/models/selector.pt']


def test_paired_error_changes_separate_same_class_instance_errors_from_wrong_class():
    from compare_confirmation import paired_error_sets
    rows=[dict(objects=[dict(class_id=0),dict(class_id=cls)]) for cls in [3,3,0,0]]
    positive=np.zeros((4,8),bool);positive[:,0]=True
    data=dict(rows=rows,positive=positive,roles=np.array([0,0,4,4]))
    changes=paired_error_sets(data,np.array([0,1,1,0]),np.array([1,0,8,1]),np.arange(4))
    assert changes==dict(correct_gained=[0,3],correct_lost=[1],acquire_added=[1],acquire_fixed=[0],carry_added=[],carry_fixed=[])


def test_raw_confirmation_sampling_does_not_inherit_processing_claims(tmp_path):
    import argparse,json,h5py
    from freeze_confirmation import main
    source=tmp_path/'source.hdf5'
    with h5py.File(source,'w') as f:
        f.create_dataset('mask/valid',data=np.array([b'demo_1']))
        for demo in ['demo_0','demo_1']:
            ep=f.create_group('data/'+demo);ep.create_dataset('actions',data=np.zeros((9,1)))
            ep.create_dataset('obs/rgb',data=np.zeros((9,2,3,3),np.uint8))
    prior=tmp_path/'prior';prior.mkdir()
    old=dict(episodes=[dict(task='task',source=str(source),demo='demo_0',episode_id='task/demo_0')],image_hw=[2,3],classes=['bowl'],
        excluded_episodes=['task/earlier'],alignment_filter=dict(candidates=999,rejected=['old_id']),annotation_version='old_processed',source_manifest_sha256='stale',selection_restriction='old train/dev')
    (prior/'manifest.json').write_text(json.dumps(old));out=tmp_path/'new'
    main(argparse.Namespace(exclude=[prior],exclude_audit=[],output=out,id_prefix='new',source_pool='valid-first',episodes_per_task=1))
    actual=json.loads((out/'manifest.json').read_text())
    assert len(actual['records'])==2 and actual['episodes'][0]['demo']=='demo_1'
    assert actual['image_hw']==[2,3] and actual['classes']==['bowl'] and actual['stride']==8
    assert not {'alignment_filter','annotation_version','source_manifest_sha256','selection_restriction'}&actual.keys()
    assert 'task/earlier' in actual['excluded_episodes'] and len(actual['sampling_exclusion_manifest_sha256'])==1
    assert not (out/'annotations_all.json').exists()


def test_fixed_clock_observations_do_not_depend_on_labels_or_query_schedule():
    from main_history import observation_schedule
    rows=[dict(episode_id='a',source='/tmp/a.hdf5',demo='a',task='task',step=t,label=None) for t in [0,5,16]]
    old=observation_schedule(rows,{'a':25})
    assert [r['step'] for r in old if r['is_grid']]==[0,8,16,24]
    assert [r['step'] for r in old if not r['is_grid']]==[5]
    rows[1]['label']={'target':'different supervised subject'}
    assert observation_schedule(rows,{'a':25})==old
    removed=observation_schedule([rows[0],rows[2]],{'a':25})
    assert [r for r in old if r['is_grid']]==removed


@pytest.mark.parametrize("skip_current",[False,True])
def test_recurrent_forward_is_causal_and_matches_incremental_state_with_reset_control(skip_current):
    from models import RecurrentPointer
    torch.manual_seed(193);c=torch.randn(2,5,8,3);x=torch.randn(2,5,4);mask=torch.ones(2,5,8,dtype=torch.bool);gap=torch.full((2,5),.69)
    for reset in [False,True]:
        net=RecurrentPointer(3,4,16,reset_each_frame=reset,skip_current=skip_current).eval()
        with torch.inference_mode():
            expected=net(c,x,mask,gap);hidden=None;steps=[]
            for t in range(5):
                logits,hidden=net.step(c[:,t],x[:,t],mask[:,t],gap[:,t],hidden);steps.append(logits)
            torch.testing.assert_close(torch.stack(steps,1),expected,atol=1e-6,rtol=1e-6)
            changed=x.clone();changed[:,3:]+=10
            torch.testing.assert_close(net(c,changed,mask,gap)[:,:3],expected[:,:3],atol=1e-6,rtol=1e-6)
            if reset:assert hidden is None
    torch.manual_seed(81);a=RecurrentPointer(3,4,16,False,skip_current)
    torch.manual_seed(81);b=RecurrentPointer(3,4,16,True,skip_current)
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())


@pytest.mark.parametrize("skip_current",[False,True])
def test_recurrent_fit_retains_unknown_observations_and_real_checkpoint_policies(skip_current):
    from recurrent_selection import fit_recurrent
    from data import metrics
    rng=np.random.default_rng(311);n=8
    raw=rng.normal(size=(n,8,3)).astype(np.float32);context=rng.normal(size=(n,4)).astype(np.float32)
    rows=[dict(episode_id='train' if i<4 else 'dev',step=[0,8,24,32][i%4],objects=[dict(class_id=0),dict(class_id=3)],label=dict(target=dict(class_id=0))) for i in range(n)]
    rows[1]['label']=None;y=np.zeros(n,np.int64);y[1]=-100
    positive=np.zeros((n,8),bool);positive[:,0]=True;valid=np.zeros((n,8),bool);valid[:,:2]=True
    data=dict(rows=rows,y=y,positive=positive,valid=valid,roles=np.zeros(n,np.int64),visible=np.ones(n,bool),weights=(y>=0).astype(np.float32))
    splits=dict(train=np.array([0,2,3]),dev=np.arange(4,8))
    def assess(p):
        m=metrics(p.argmax(-1),splits['dev'],data)
        return dict(joint_goal_met=False,metrics=m,sequence=dict(stable_class_switch_rate=0.,correct_box_at_change=0))
    history_rows=[dict(episode_id=ep,step=t,is_grid=True) for ep in ['train','dev'] for t in [0,8,16,24,32]]
    history_context=rng.normal(size=(10,4)).astype(np.float32);history_context[[0,1,3,4,5,6,8,9]]=context
    history=dict(rows=history_rows,context=history_context)
    probability,states,info=fit_recurrent(data,(raw,context),splits,9,'cpu','gru',assess,history,skip_current=skip_current)
    assert info['unknown_training_queries']==1 and info['background_training_observations']==1 and info['train_episodes']==1
    assert set(states)=={'raw_f1','fixed_gate'} and set(probability)==set(states)
    for p in probability.values():assert p.shape==(n,9) and np.isfinite(p).all() and np.allclose(p.sum(-1),1)
    from checkpoint_selection import checkpoint_merit
    value=dict(joint_growth_goal_met=False,calibration=dict(metrics=dict(recall=0.,f1=0.,wrong_acquire=0,wrong_carry=0,precision=None),sequence=dict(stable_class_switch_rate=None,correct_box_at_change=0)))
    assert checkpoint_merit(value)[-1]==0.


@pytest.mark.parametrize("skip_current",[False,True])
def test_fixed_clock_query_logits_match_streaming_and_ignore_temporary_queries(skip_current):
    from recurrent_selection import grid_batch,query_logits
    from models import RecurrentPointer
    torch.manual_seed(189)
    history_rows=[dict(episode_id=ep,step=t,is_grid=True) for ep,t in [('a',0),('a',8),('a',16),('b',0),('b',8)]]
    h=torch.randn(5,4);rows=[dict(episode_id=ep,step=t) for ep,t in [('a',0),('a',5),('a',8),('a',16),('b',8)]]
    c=torch.randn(5,8,3);x=torch.randn(5,4);x[[0,2,3,4]]=h[[0,1,2,4]];mask=torch.ones(5,8,dtype=torch.bool)
    for reset in [False,True]:
        net=RecurrentPointer(3,4,16,reset,skip_current).eval();batch=grid_batch(rows,c,x,mask,h,history_rows)
        with torch.inference_mode():
            expected=query_logits(net,batch);stream=[]
            for q,row in enumerate(rows):
                state=None;actual=None
                for j,frame in enumerate(history_rows):
                    if frame['episode_id']!=row['episode_id'] or frame['step']>row['step']//8*8:continue
                    gap=torch.tensor([0. if frame['step']==0 else np.log(2)],dtype=torch.float32)
                    actual,state=net.step(c[q:q+1],h[j:j+1],mask[q:q+1],gap,state)
                if row['step']%8:
                    gap=torch.tensor([np.log1p((row['step']%8)/8)],dtype=torch.float32)
                    actual,_=net.step(c[q:q+1],x[q:q+1],mask[q:q+1],gap,state)
                stream.append(actual[0])
            torch.testing.assert_close(torch.stack(stream),expected,atol=1e-6,rtol=1e-6)
            from verify_grid_replay import stream_probability
            clock_rows=[history_rows[0],dict(episode_id='a',step=5,is_grid=False),*history_rows[1:]]
            clock_context=torch.cat([h[:1],x[1:2],h[1:]]).numpy()
            actual=stream_probability(net,dict(rows=rows),dict(rows=clock_rows,context=clock_context),c,x,mask,dict(xm=np.zeros(4,np.float32),xs=np.ones(4,np.float32)))
            np.testing.assert_allclose(actual,expected.softmax(-1).numpy(),atol=1e-6,rtol=1e-6)
            extra=grid_batch(rows+[dict(episode_id='a',step=3)],torch.cat([c,c[:1]]),torch.cat([x,x[:1]+10]),torch.cat([mask,mask[:1]]),h,history_rows)
            torch.testing.assert_close(query_logits(net,extra)[:5],expected,atol=1e-6,rtol=1e-6)
            future=h.clone();future[2]+=10
            other=grid_batch(rows,c,x,mask,future,history_rows)
            torch.testing.assert_close(query_logits(net,other)[[0,1,2,4]],expected[[0,1,2,4]],atol=1e-6,rtol=1e-6)


def test_confirmation_history_binds_encoder_clock_tasks_and_manifest():
    from history_contract import public_history_contract,validate_history_contract
    meta=dict(protocol='fixed_clock_main_context_v1',grid_stride=8,context_dim=294,tasks=['a','b'],feature_checkpoint_sha256='encoder',urdf_sha256='robot',tf32=False,corpus_manifest_sha256='cohort')
    frozen=public_history_contract(meta);validate_history_contract(meta,frozen,'cohort')
    for key,value in [('feature_checkpoint_sha256','other'),('tasks',['b','a']),('grid_stride',4),('tf32',True),('corpus_manifest_sha256','other')]:
        with pytest.raises(ValueError):validate_history_contract(dict(meta,**{key:value}),frozen,'cohort')


def test_contact_priority_keeps_shared_collision_identity_ambiguity():
    from types import SimpleNamespace
    from audit_contact_priority import collision_groups,contact_facts
    cabinet=SimpleNamespace(pybullet_uuid=9,link_ids=[4,4,4]);bowl=SimpleNamespace(pybullet_uuid=31)
    parts=[dict(inst=cabinet,indices=[i],identity=name,cls=cls) for i,name,cls in [(0,'drawer',4),(1,'left_handle',5),(2,'right_handle',5)]]
    parts.append(dict(inst=bowl,indices=None,identity='bowl',cls=0))
    groups=collision_groups(parts);assert len(groups)==2
    shared=next(g for g in groups if g['body']==9);assert shared['resolved_instance'] is None and shared['identities']==['drawer','left_handle','right_handle']
    values=[dict(g,min_hand_distance_m=.001 if g['body']==9 else .15,translation_m=.01,rotation_rad=0.) for g in groups]
    facts=contact_facts(values,'bowl');assert facts['competing_contact'] and facts['competing_moving_contact'] and not facts['target_touching']
    assert facts['contact_groups'][0]['resolved_instance'] is None
    values[1]['min_hand_distance_m']=.001
    assert not contact_facts(values,'bowl')['competing_contact']


def test_contact_audit_plan_excludes_unknown_nonacquire_and_other_partitions():
    from audit_contact_priority import audit_plan
    rows=[dict(id=str(i),source='/tmp/source',demo='demo',episode_id='ep',step=i*8,label=dict(target=dict(instance_id='bowl',class_id=0,onset=32),reason='proxy')) for i in range(5)]
    data=dict(rows=rows,y=np.array([0,0,0,-100,0]),roles=np.array([0,2,0,0,0]))
    plan=audit_plan(data,dict(train=[0,1,3],dev=[2]));assert [r['id'] for r in plan]==['0','2']
    assert plan[0]['future_motion_steps']==32 and plan[1]['partition']=='dev'


def test_far_future_confidence_changes_training_weights_only_and_requires_complete_audit():
    from supervision_timing import far_future_mask
    from supervision_confidence import downweight_training
    rows=[dict(id=str(i),source='/tmp/source',demo='demo',step=i*8,label=dict(target=dict(instance_id='bowl',class_id=0,onset=40))) for i in range(3)]
    data=dict(rows=rows,y=np.zeros(3,np.int64),roles=np.zeros(3,np.int64),weights=np.ones(3,np.float32))
    audit=dict(records=[dict(rows[i],target='bowl',future_motion_steps=40-i*8,facts=dict(target_far=i!=1)) for i in range(3)])
    mask=far_future_mask(data,audit,np.arange(3));assert mask.tolist()==[True,False,True]
    actual=downweight_training(data,np.array([0,1]),mask,.5)
    np.testing.assert_array_equal(actual['weights'],[.5,1.,1.]);np.testing.assert_array_equal(data['weights'],np.ones(3));assert actual['y'] is data['y']
    with pytest.raises(ValueError):far_future_mask(data,dict(records=audit['records'][:2]),np.arange(3))
    audit['records'][0]['future_motion_steps']=1
    with pytest.raises(ValueError):far_future_mask(data,audit,np.arange(3))


def test_border_audit_keeps_alternate_positives_and_exhaustive_denominators():
    from audit_border_visibility import border_flags,summarize
    objects=[dict(box=[40,0,60,2],class_id=0),dict(box=[40,20,60,40],class_id=0)]
    assert border_flags(objects,np.array([True,False]),[90,120],4)
    assert not border_flags(objects,np.array([True,True]),[90,120],4)
    assert not border_flags(objects,np.array([False,False]),[90,120],4)
    d=dict(rows=[dict(objects=objects,id=str(i),episode_id='ep',step=i) for i in range(3)],positive=np.array([[True,False],[True,True],[False,False]]),roles=np.zeros(3,int),visible=np.ones(3,bool))
    result=summarize(d,np.arange(3),np.array([0,1,8]),4)
    assert [result['buckets'][k]['known'] for k in ['edge_sliver','other_available','no_matching_proposal']]==[1,1,1]
    assert sum(v['correct'] for v in result['buckets'].values())==2
    import json
    json.dumps(result)


def test_border_emphasis_leaves_labels_inputs_and_dev_weights_unchanged():
    from border_training import border_training_mask,emphasize_training
    objects=[dict(box=[40,0,60,2],class_id=0),dict(box=[40,20,60,40],class_id=0)]
    data=dict(rows=[dict(objects=objects) for _ in range(4)],y=np.array([0,0,0,-100]),positive=np.array([[True,False],[True,True],[True,False],[True,False]]),roles=np.array([0,4,3,0]),weights=np.ones(4,np.float32),context=np.zeros((4,3)))
    mask=border_training_mask(data,np.arange(4));assert mask.tolist()==[True,False,True,False]
    assert border_training_mask(data,np.arange(4),scope='bowl').tolist()==[True,False,False,False]
    fitted=emphasize_training(data,np.array([0,1,3]),mask,4)
    np.testing.assert_array_equal(fitted['weights'],[4,1,1,1]);np.testing.assert_array_equal(data['weights'],[1,1,1,1])
    assert fitted['y'] is data['y'] and fitted['context'] is data['context']
    assert np.array_equal(emphasize_training(data,np.arange(4),mask,1)['weights'],data['weights'])
    with pytest.raises(ValueError):emphasize_training(data,np.arange(4),mask,.5)


def test_sensor_crop_aligns_pixels_masks_and_camera_geometry():
    from sensor_crop import apply_crop,pixel_affine,mask_box,choose_crop,transform_geometry_context
    from robot_geometry import box_geometry
    crop=[12,9,96,72];mask=np.zeros((90,120),bool);mask[20:40,30:50]=True
    transformed=apply_crop(mask,crop,True);assert mask_box(transformed)==[22,14,47,39]
    affine=pixel_affine(crop);point=np.array([.3,.4]);np.testing.assert_allclose(point*affine[:2]+affine[2:],[(.3*120-12)/96,(.4*90-9)/72],atol=1e-7)
    # 80% crop maps central sensor pixels exactly to central rays; all plane points remain on the same surface.
    depth=np.ones((90,120),np.float32);ee=np.array([1.,0,0]);box=np.array([0,0,1,1])
    identity=box_geometry(box,depth,ee,60,1,pixel_affine=np.array([1,1,0,0]))
    np.testing.assert_array_equal(identity,box_geometry(box,depth,ee,60,1))
    original=box_geometry(np.array([.1,.1,.9,.9]),depth,ee,60,1)
    augmented=box_geometry(box,apply_crop(depth,crop,True),ee,60,1,pixel_affine=affine)
    assert abs(original[0]-augmented[0])<.01 and abs(original[2]-augmented[2])<.01
    g=np.arange(14,dtype=np.float32);new=transform_geometry_context(g,crop)
    np.testing.assert_array_equal(new[:3],g[:3]);np.testing.assert_allclose(new[3:5],g[3:5]*affine[:2]+affine[2:]);np.testing.assert_allclose(new[12:],g[12:]*affine[:2])
    for bad in [[-1,0,96,72],[0,0,96,71],[0,0,121,90]]:
        with pytest.raises(ValueError):pixel_affine(bad)
    near=np.zeros((90,120),bool);near[2:18,40:65]=True
    chosen,targeted=choose_crop(near,np.random.default_rng(7));assert targeted
    b=mask_box(apply_crop(near,chosen,True));assert 1<=min(b[2]-b[0],b[3]-b[1])<=8


def test_cropped_labels_use_transformed_visible_mask_including_target_absence():
    from sensor_crop import apply_crop,mask_box,pixel_affine
    from proxy import target_candidate
    mask=np.zeros((90,120),bool);mask[2:18,40:65]=True
    visible=apply_crop(mask,[0,14,96,72],True);box=mask_box(visible)
    truth=[dict(instance_id='bowl',class_id=0,box=box)];proposal=[dict(class_id=0,box=box)]
    assert target_candidate(proposal,truth,dict(instance_id='bowl',class_id=0))==0
    removed=apply_crop(mask,[0,18,96,72],True);assert mask_box(removed) is None
    assert target_candidate(proposal,[],dict(instance_id='bowl',class_id=0))==-1
    # The same crop must map current/past principal points by the same affine.
    from sensor_crop import transform_geometry_context
    g=np.zeros(14,np.float32);g[3:5]=[.5,.6];g[12:14]=[.1,-.05]
    a=pixel_affine([0,14,96,72]);new=transform_geometry_context(g,[0,14,96,72])
    np.testing.assert_allclose(new[3:5]-new[12:14],(g[3:5]-g[12:14])*a[:2]+a[2:],atol=1e-7)


def test_augmentation_loss_uses_original_normalization_and_exact_training_partition():
    from augmentation_loss import prepare_augmentation
    data=dict(y=np.array([0,0,0,0]));train=np.array([0,2]);raw=np.full((2,8,3),5,np.float32);context=np.full((2,4),6,np.float32)
    aug=dict(base_index=np.array([2,0]),candidates=raw,context=context,valid=np.ones((2,8),bool),y=np.array([8,1]))
    norm=dict(cm=np.ones(3),cs=np.ones(3)*2,xm=np.ones(4)*2,xs=np.ones(4)*4)
    value=prepare_augmentation(data,aug,train,norm,'cpu');assert value['lookup'].tolist()==[1,-1,0,-1]
    np.testing.assert_array_equal(value['c'].numpy(),np.ones((2,8,3))*2);np.testing.assert_array_equal(value['x'].numpy(),np.ones((2,4)))
    assert value['labels'].tolist()==[8,1]
    with pytest.raises(ValueError):prepare_augmentation(data,dict(aug,base_index=np.array([0,1])),train,norm,'cpu')
    invalid=aug['valid'].copy();invalid[1,1]=False
    with pytest.raises(ValueError):prepare_augmentation(data,dict(aug,valid=invalid),train,norm,'cpu')
    with pytest.raises(ValueError):prepare_augmentation(dict(y=np.array([-100,0,0,0])),aug,train,norm,'cpu')


def test_crop_stage_audit_cli_imports_without_prior_data_loader_imports():
    import subprocess,sys
    from pathlib import Path
    script=Path(__file__).with_name('audit_crop_detection.py')
    completed=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    assert '--archive' in completed.stdout


def test_detector_crop_targets_keep_all_visible_instances_without_intent_labels():
    from perception_data import detector_target
    masks=np.zeros((2,90,120),bool);masks[0,2:18,40:65]=True;masks[1,30:60,50:100]=True
    objects=[dict(class_id=0),dict(class_id=3)]
    original=detector_target(objects,masks);assert original['labels'].tolist()==[1,4]
    cropped=detector_target(objects,masks,[0,18,96,72]);assert cropped['labels'].tolist()==[4] and cropped['masks'].shape==(1,90,120)
    empty=detector_target([],np.empty((0,90,120),bool));assert empty['boxes'].shape==(0,4) and empty['masks'].shape==(0,90,120)
    assert (cropped['boxes'][:,2:]>cropped['boxes'][:,:2]).all()


def test_detector_metrics_match_each_instance_once_and_count_false_classes():
    from detector_metrics import instance_counts,aggregate_counts
    truth=[dict(class_id=0,box=[0,0,5,5]),dict(class_id=3,box=[10,10,20,20])]
    detections=[dict(class_id=0,box=[0,0,5,5],score=.9),dict(class_id=0,box=[0,0,5,5],score=.8),dict(class_id=5,box=[10,10,20,20],score=.7)]
    counts=instance_counts(detections,truth);assert counts['0']==dict(tp=1,fp=1,fn=0) and counts['3']['fn']==1 and counts['5']['fp']==1
    repeated_truth=[truth[0],truth[0]]
    one_to_one,matched=instance_counts(detections[:1],repeated_truth,return_matches=True);assert len(matched)==1 and one_to_one['0']['fn']==1
    summary=aggregate_counts([counts]);assert summary['micro']['tp']==1 and summary['micro']['fp']==2 and summary['micro']['fn']==1


def test_detector_gate_requires_real_gain_without_extra_false_boxes_or_lost_classes():
    import copy
    from train_perception_adaptation import detector_gate
    baseline=dict(bounded8=dict(micro=dict(fp=0),classes={str(c):dict(tp=30,fn=0,fp=0,recall=1.) for c in range(6)}),
        weak_targets=dict(available=100,acquire_available=20),edge_small=dict(matched=10))
    candidate=copy.deepcopy(baseline);candidate['edge_small']['matched']=11
    assert detector_gate(candidate,baseline)['eligible']
    for section,key,value in [('weak_targets','available',99),('weak_targets','acquire_available',19),('edge_small','matched',10)]:
        changed=copy.deepcopy(candidate);changed[section][key]=value;assert not detector_gate(changed,baseline)['eligible']
    changed=copy.deepcopy(candidate);changed['bounded8']['micro']['fp']=1;assert not detector_gate(changed,baseline)['eligible']
    changed=copy.deepcopy(candidate);changed['bounded8']['classes']['0']['fp']=1;assert not detector_gate(changed,baseline)['eligible']
    changed=copy.deepcopy(candidate);changed['bounded8']['classes']['3']['recall']=.9;assert not detector_gate(changed,baseline)['eligible']


def test_detector_interpolation_preserves_initializer_endpoints_and_frozen_backbone():
    import torch
    from detector_interpolation import interpolate_detector
    initial={'backbone.weight':torch.tensor([1.,2.]),'head.weight':torch.tensor([2.,4.]),'count':torch.tensor(3)}
    adapted={k:v.clone() for k,v in initial.items()};adapted['head.weight']+=4
    for alpha,endpoint in [(0,initial),(1,adapted)]:
        result=interpolate_detector(initial,adapted,alpha)
        assert all(torch.equal(result[k],v) and result[k].data_ptr()!=v.data_ptr() for k,v in endpoint.items())
    result=interpolate_detector(initial,adapted,.25)
    assert torch.equal(result['head.weight'],torch.tensor([3.,5.]))
    assert torch.equal(result['backbone.weight'],initial['backbone.weight'])
    for alpha in [-.1,1.1,float('nan')]:
        with pytest.raises(ValueError):interpolate_detector(initial,adapted,alpha)
    for key in ['backbone.weight','count']:
        changed={k:v.clone() for k,v in adapted.items()};changed[key]+=1
        with pytest.raises(ValueError):interpolate_detector(initial,changed,.25)


def test_positive_set_loss_matches_singleton_ce_and_keeps_unknown_and_missing_semantics():
    import torch
    from positive_set_loss import positive_set_loss
    logits=torch.tensor([[2.,1.,*([float('-inf')]*6),0.],[1.,3.,*([float('-inf')]*6),2.],[0.,1.,*([float('-inf')]*6),2.]],requires_grad=True)
    valid=torch.tensor([[True,True,*([False]*6)]]*3);positive=torch.tensor([[True,*([False]*7)],[False]*8,[False]*8])
    labels=torch.tensor([0,-100,8]);weights=torch.tensor([1.,5.,2.])
    expected=weighted_loss(logits,labels,weights,valid)
    for mode in ['uniform','marginal']:
        actual=positive_set_loss(logits,labels,positive,weights,valid,mode)
        torch.testing.assert_close(actual,expected)
        grad=torch.autograd.grad(actual,logits,retain_graph=True)[0]
        assert torch.isfinite(grad).all() and (grad[1]==0).all() and (grad[:,2]==0).all()


def test_positive_set_loss_rewards_either_matching_box_without_accepting_rivals():
    import torch
    from positive_set_loss import positive_set_loss
    logits=torch.tensor([[0.,3.,-1.,-2.]],requires_grad=True)
    positive=torch.tensor([[True,True,False]]);valid=torch.ones_like(positive);weights=torch.ones(1)
    value=positive_set_loss(logits,torch.tensor([0]),positive,weights,valid)
    probability=logits.softmax(-1)
    torch.testing.assert_close(value,-probability[0,:2].sum().log())
    grad=torch.autograd.grad(value,logits)[0]
    assert (grad[0,:2]<0).all() and (grad[0,2:]>0).all()
    for labels in [torch.tensor([0]),torch.tensor([1])]:
        torch.testing.assert_close(positive_set_loss(logits,labels,positive,weights,valid),value)
    with pytest.raises(ValueError):positive_set_loss(logits,torch.tensor([3]),positive,weights,valid)
    invalid=valid.clone();invalid[0,1]=False
    with pytest.raises(ValueError):positive_set_loss(logits,torch.tensor([0]),positive,weights,invalid)


def test_box_matching_does_not_hide_cross_instance_ambiguity():
    from audit_positive_sets import matching_instances
    truth=[dict(class_id=0,instance_id='bowl_a',box=[0,0,10,10]),dict(class_id=0,instance_id='bowl_b',box=[2,0,12,10]),
        dict(class_id=3,instance_id='rack',box=[0,0,10,10])]
    assert matching_instances(dict(class_id=0,box=[1,0,11,10]),truth)==['bowl_a','bowl_b']
    assert matching_instances(dict(class_id=3,box=[1,0,11,10]),truth)==['rack']


def test_bowl_class_auxiliary_reduces_other_class_mass_without_changing_nonbowl_or_missing_targets():
    import torch
    from bowl_class_loss import bowl_class_loss
    logits=torch.tensor([[0.,1.,2.,-1.]]*5,requires_grad=True)
    labels=torch.tensor([0,1,3,-100,0]);roles=torch.tensor([0,2,4,0,4])
    classes=torch.tensor([[0,0,3]]*5);valid=torch.ones((5,3),dtype=torch.bool);weights=torch.ones(5)
    value=bowl_class_loss(logits,labels,roles,classes,weights,valid)
    expected=-logits.softmax(-1)[[0,4],:2].sum(-1).log().mean()
    torch.testing.assert_close(value,expected)
    grad=torch.autograd.grad(value,logits)[0]
    assert (grad[[0,4],:2]<0).all() and (grad[[0,4],2:]>0).all()
    assert (grad[1:4]==0).all()
    empty=bowl_class_loss(logits,torch.full((5,),3),roles,classes,weights,valid)
    assert empty.item()==0
    bad=classes.clone();bad[0]=3
    with pytest.raises(ValueError):bowl_class_loss(logits,labels,roles,bad,weights,valid)
