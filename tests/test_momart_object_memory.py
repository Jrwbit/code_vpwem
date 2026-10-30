import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from momart_object_memory.state import ObjectBank, ObjectObservation, ObservationStream, query_schedule
from momart_object_memory.cache import select_mode
from momart_object_memory.model import ObjectConditionedPTP, ObjectMemoryEncoder, PackedObjectCondition
from momart_object_memory.train import comparison_contract
from momart_object_memory.online import OnlineObjects


def observation(value=1., cls=0, box=(.1,.1,.3,.3), d=4):
    return ObjectObservation(cls,np.array(box,np.float32),.9,np.full(d,value,np.float32))


def bank():
    return ObjectBank(slots=3,history=3,feature_dim=4)


def test_modes_and_absent_box_semantics():
    b=bank();b.update(0,0,[observation(1)]);b.update(2,.1,[observation(2)]);b.update(4,.2,[])
    assert not b.snapshot('a1')['valid'].any()
    a2=b.snapshot('a2');a3=b.snapshot('a3')
    assert a2['valid'][0] and not a2['visible'][0] and not a2['boxes'].any()
    np.testing.assert_allclose(a2['ages'][0,-1],.1)
    assert a2['event_valid'].sum()==1 and a3['event_valid'].sum()==2
    np.testing.assert_array_equal(a3['features'][0,-2:,0],[1,2])
    for mode in ('a1','a2','a3','a3_latest_control'):
        expected=b.snapshot(mode);actual=select_mode(b.snapshot('a3'),mode)
        for k in expected:np.testing.assert_array_equal(expected[k],actual[k])


def test_instances_keep_slots_across_reordered_detections():
    b=bank();left=observation(1);right=observation(2,box=(.6,.1,.8,.3))
    b.update(0,0,[right,left]);b.update(1,.05,[left,right])
    s=b.snapshot('a3')
    np.testing.assert_array_equal(s['features'][:2,-1,0],[1,2])
    b.update(2,.1,[]);b.update(3,.15,[right,left])
    assert len(b.tracks)==2
    b.reset();assert not b.snapshot('a3')['valid'].any()


def test_unique_sensor_steps_and_capacity_failures():
    b=bank();stream=ObservationStream(b,lambda image:[observation(float(image))])
    stream.observe([1,1],[0,0],[0,0]);assert b.snapshot('a3')['event_valid'].sum()==1
    # Equal pixels at a real new sensor time must still produce a new event.
    stream.observe([1,1],[7,8],[.35,.4]);assert b.snapshot('a3')['event_valid'].sum()==3
    with pytest.raises(ValueError):b.update(8,.4,[])
    with pytest.raises(ValueError):stream.observe([1],[9,10],[.45])
    full=ObjectBank(1,3,4);full.update(0,0,[observation()])
    with pytest.raises(ValueError,match='capacity'):full.update(1,.05,[observation(cls=1)])


def test_prefix_replay_cadence_and_future_independence():
    assert query_schedule(16,8)==[[0,0],[7,8],[15,16]]
    assert query_schedule(9,8)==[[0,0],[0,1],[8,9]]
    frames=np.arange(25)
    def replay(frames):
        stream=ObservationStream(bank(),lambda x:[observation(float(x))])
        for steps in query_schedule(16,8):stream.observe(frames[steps],steps,np.array(steps)*.05)
        return stream.bank.snapshot('a3')
    old=replay(frames);frames[17:]=-100
    for k,v in old.items():np.testing.assert_array_equal(v,replay(frames)[k])
    np.testing.assert_array_equal(old['features'][0,:,0],[8,15,16])


def payload(b=1,n=3,d=16):
    # Model reads a width-d scene, then n*d token values, then n validity values.
    scene=torch.randn(b,2,d)
    token=torch.randn(b,n,d)
    valid=torch.tensor([[1.,0.,1.]]).expand(b,-1)
    pack=torch.cat([token.flatten(1),valid],-1)
    return torch.cat([scene,pack[:,None].expand(-1,2,-1)],-1)


def network():
    torch.manual_seed(31)
    return ObjectConditionedPTP(2,16,4,2,3,d_model=16,nhead=4,num_layers=2,
                               p_drop_emb=0.,p_drop_attn=0.).eval()


def test_padding_blocks_invalid_tokens_and_ptp_blocks_latest_bank():
    model=network();c=payload();x=torch.randn(1,4,2);t=torch.tensor([3])
    first=model(x,t,c)
    invalid=c.clone();invalid[:,:,16+16:16+32]+=1000
    torch.testing.assert_close(first,model(x,t,invalid),atol=1e-6,rtol=1e-6)
    changed=c.clone();changed[:,:,16:16+48]+=100
    other=model(x,t,changed)
    torch.testing.assert_close(first[:,0],other[:,0],atol=1e-6,rtol=1e-6)
    assert not torch.allclose(first[:,1:],other[:,1:])
    empty=c.clone();empty[:,:,-3:]=0
    assert torch.isfinite(model(x,t,empty)).all()


def test_invalid_history_cannot_affect_temporal_encoder():
    encoder=ObjectMemoryEncoder(4,2,width=16).eval()
    b=bank();b.update(0,0,[observation()]);data={k:torch.as_tensor(v)[None] for k,v in b.snapshot('a2').items()}
    first,_=encoder(data);data['features'][:,:,:-1]=1000
    other,_=encoder(data)
    torch.testing.assert_close(first,other)


def test_contract_only_allows_mode_and_output_directory_differences():
    cfg=OmegaConf.load(Path(__file__).parents[1]/'momart_object_memory/config.yaml')
    other=copy.deepcopy(cfg);other.object_memory.mode='a3';other.work_dir='/tmp/another-arm'
    assert comparison_contract(cfg)==comparison_contract(other)
    other.action_steps=4
    assert comparison_contract(cfg)!=comparison_contract(other)


def test_online_streams_isolated_and_reset_first_frame_once():
    cfg=OmegaConf.create(dict(action_steps=8,object_memory=dict(slots=3,history=3,
        feature_dim=4,match_iou=.3,period_seconds=.05,mode='a3')))
    calls=[]
    def perceive(image):
        calls.append(int(image[0,0,0]));return [observation(float(image[0,0,0]))]
    online=OnlineObjects(cfg,perceive,2)
    rgb=np.zeros((2,2,3,90,120),np.float32);rgb[1]=1
    result=online({'rgb':rgb})
    assert calls==[0,255]
    np.testing.assert_array_equal(result['_object_event_valid'][:,0].sum(-1),[1,1])
    np.testing.assert_array_equal(result['_object_features'][:,0,-1,0],[0,255])
    online({'rgb':rgb});assert len(calls)==6
    online.reset_all();result=online({'rgb':rgb})
    assert len(calls)==8 and result['_object_event_valid'].sum()==2


def test_all_three_arms_have_identical_parameters_and_initialization():
    from momart_object_memory.model import build_agent
    import robomimic.utils.obs_utils as ObsUtils
    from momart_boxed_no_history.train import initialize_modalities
    cfg=OmegaConf.load(Path(__file__).parents[1]/'momart_object_memory/config.yaml')
    cfg.device='cpu';cfg.diffusion_num_layers=1;cfg.sample_steps=2
    initialize_modalities(cfg)
    class ActionNorm:
        range=torch.ones(10);min=-torch.ones(10)
    saved=None
    for mode in ('a1','a2','a3'):
        cfg.object_memory.mode=mode;torch.manual_seed(100)
        agent=build_agent(cfg,{'action':ActionNorm()})
        state=agent.model.state_dict()
        if saved is None:saved={k:v.clone() for k,v in state.items()}
        else:
            assert state.keys()==saved.keys()
            for k in state:torch.testing.assert_close(state[k],saved[k],atol=0,rtol=0)
        # Baseline observation positional rows must be copied to their new locations.
        assert state['diffusion.cond_pos_emb'].shape==(1,11,256)


def test_offline_snapshots_match_online_prefix_for_all_phases(tmp_path):
    import zarr
    from momart_object_memory.cache import write_cache
    root=zarr.open_group(str(tmp_path/'source.zarr'),mode='w')
    root.attrs.update(vpwem_cache_complete=True,vpwem_cache_kind='momart',demo_names=['demo_0','demo_1'])
    root.create_dataset('meta/episode_ends',data=np.array([19,24]))
    images=np.broadcast_to(np.arange(24,dtype=np.uint8)[:,None,None,None],(24,90,120,3)).copy()
    root.create_dataset('data/rgb',data=images)
    root.create_dataset('data/action',data=np.zeros((24,10),np.float32))
    cfg=OmegaConf.create(dict(obs_steps=2,action_steps=8,object_memory=dict(slots=3,history=3,
        feature_dim=4,match_iou=.3,period_seconds=.05)))
    class Perception:
        metadata={'test_only':True}
        def __call__(self,image):
            v=int(image[0,0,0]);return [] if v%4==3 else [observation(v)]
    write_cache(cfg,tmp_path/'source.zarr',tmp_path/'objects.zarr',Perception())
    cached=zarr.open_group(str(tmp_path/'objects.zarr'),mode='r')
    assert cached.attrs['complete'] is True
    for start,end in ((0,19),(19,24)):
        for t in range(end-start):
            stream=ObservationStream(bank(),Perception())
            for steps in query_schedule(t,8):
                stream.observe(images[np.array(steps)+start],steps,np.array(steps)*.05)
            for k,v in stream.bank.snapshot('a3').items():
                np.testing.assert_array_equal(v,cached[k][start+t])
    with pytest.raises(FileExistsError):write_cache(cfg,tmp_path/'source.zarr',tmp_path/'objects.zarr',Perception())


def test_full_agent_backward_ema_and_denoising_do_not_mutate_bank():
    from momart_object_memory.model import build_agent
    from momart_boxed_no_history.train import initialize_modalities
    cfg=OmegaConf.load(Path(__file__).parents[1]/'momart_object_memory/config.yaml')
    cfg.device='cpu';cfg.diffusion_num_layers=1;cfg.sample_steps=3
    cfg.short_cond_dropout=0.;cfg.long_cond_dropout=0.
    initialize_modalities(cfg)
    class ActionNorm:
        range=torch.ones(10);min=-torch.ones(10)
    agent=build_agent(cfg,{'action':ActionNorm()})
    obs={k:torch.zeros(2,2,*list(v.shape)) for k,v in cfg.shape_meta.obs.items()}
    obs['_task_id']=torch.tensor([0,1])
    b=ObjectBank(8,3,128);b.update(0,0,[observation(d=128)])
    snap=b.snapshot('a3');before=copy.deepcopy(snap)
    obs.update({'_object_'+k:torch.as_tensor(v)[None].expand(2,*v.shape).clone() for k,v in snap.items()})
    initial=agent.model['condition'].objects.event[0].weight.detach().clone()
    loss=agent.update(torch.randn(2,9,10).clamp(-1,1),obs)
    assert np.isfinite(loss['loss'])
    assert not torch.equal(initial,agent.model['condition'].objects.event[0].weight)
    agent.model.eval();agent.model_ema.eval();agent.reset_cache()
    out,_=agent.sample(prior=torch.zeros(2,9,10),n_samples=2,sample_steps=3,
                       condition_cfg=obs,w_cfg=1.,use_ema=True)
    assert out.shape==(2,9,10) and torch.isfinite(out).all()
    for k,v in before.items():np.testing.assert_array_equal(v,b.snapshot('a3')[k])


def test_dataset_preserves_actions_other_modalities_and_episode_padding(tmp_path):
    import hashlib
    import zarr
    import robomimic.utils.obs_utils as ObsUtils
    from momart_bbox_memory.detector import MODEL_ID,REVISION
    from momart_boxed_no_history.dataset import BoxedNoHistoryDataset
    from momart_object_memory.cache import write_cache
    from momart_object_memory.dataset import ObjectDataset
    from momart_object_memory.train import split_batch
    ObsUtils.initialize_obs_modality_mapping_from_dict({'rgb':['rgb'],'low_dim':['proprio']})
    source=tmp_path/'sample.zarr';root=zarr.open_group(str(source),mode='w')
    root.attrs.update(vpwem_cache_complete=True,vpwem_cache_kind='momart',demo_names=['demo_0','demo_1'])
    ends=np.array([4,7]);root.create_dataset('meta/episode_ends',data=ends)
    images=np.broadcast_to(np.arange(7,dtype=np.uint8)[:,None,None,None],(7,90,120,3)).copy()
    root.create_dataset('data/rgb',data=images)
    root.create_dataset('data/action',data=np.arange(70,dtype=np.float32).reshape(7,10))
    root.create_dataset('data/proprio',data=np.arange(70,dtype=np.float32).reshape(7,10)+100)
    boxes=zarr.open_group(str(tmp_path/'boxes.zarr'),mode='w')
    boxes.attrs.update(complete=True,completed_frames=7,signature=dict(source=str(source.resolve()),
        shape=list(images.shape),episode_ends_sha256=hashlib.sha256(ends.tobytes()).hexdigest(),
        detector=dict(model_id=MODEL_ID,revision=REVISION,prompt='a red bowl.')))
    boxes.create_dataset('boxes',data=np.zeros((7,4),np.float32));boxes.create_dataset('valid',data=np.zeros(7,bool))
    cfg=OmegaConf.create(dict(data_root=str(tmp_path),device='cpu',obs_steps=2,action_steps=8,
        object_memory=dict(cache_root=str(tmp_path),mode='a3',slots=3,history=3,feature_dim=4,
                           match_iou=.3,period_seconds=.05,classes=['bowl'])))
    class Perception:
        metadata={'classes':['bowl'],'feature_dim':4}
        def __call__(self,image):return [observation(float(image[0,0,0]))]
    write_cache(cfg,source,tmp_path/'sample.objects.zarr',Perception())
    base=BoxedNoHistoryDataset(source,{'rgb':{},'proprio':{}},9,2,20,tmp_path/'boxes.zarr')
    class Identity:
        def normalize(self,x):return x
    base.normalizer={'obs':{'rgb':Identity(),'proprio':Identity()},'action':Identity()}
    wrapped=ObjectDataset(base,cfg,'sample')
    for index in (0,2,3,4,6):
        old=base[index];new=wrapped[index]
        np.testing.assert_array_equal(old['actions'],new['actions'])
        np.testing.assert_array_equal(old['obs']['proprio'],new['obs']['proprio'])
        if index in (3,6):np.testing.assert_array_equal(new['obs']['rgb'][0],new['obs']['rgb'][1])
    batch=torch.utils.data.default_collate([wrapped[0],wrapped[3]])
    condition,actions=split_batch(cfg,wrapped,2,batch)
    torch.testing.assert_close(actions,batch['actions'])
    assert condition['_object_features'].shape==(2,3,3,4)
    assert condition['_object_classes'].dtype==torch.int64
    assert condition['_object_valid'].dtype==torch.bool
    cfg.action_steps=4
    with pytest.raises(ValueError,match='incompatible'):ObjectDataset(base,cfg,'sample')


def test_roi_uses_original_pixels_instead_of_padded_grid_extent():
    from momart_object_memory.perception import raw_roi_features
    # A feature ramp in y has a known mean at the ROI midpoint. 90px height is
    # padded to a 12-row stride-8 grid, so scaling boxes by 12 would be wrong.
    grid=torch.arange(12,dtype=torch.float32)[None,None,:,None].expand(1,1,12,15)
    output=raw_roi_features(grid,torch.tensor([[.2,.2,.6,.8]]))
    torch.testing.assert_close(output,torch.tensor([[5.125]]))
