import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf
from momart_no_history_baseline.dataset import ZarrNoHistorySequenceDataset
from momart_target_no_history.inputs import object_inputs, replace_target_inputs
from momart_target_no_history.dataset import TargetNoHistoryDataset, source_signature, STANDARD
from momart_target_no_history.configure import replacement_configs
from test_momart_batch_sequence_reads import make_dataset


def metadata():
    return dict(annotation_standard=STANDARD,bundle_manifest_sha256='a'*64,
        sensor_protocol_sha256='b'*64,supported_sources=['test'])


def fixture_dataset(mode):
    base = make_dataset(ZarrNoHistorySequenceDataset)
    d = object.__new__(TargetNoHistoryDataset)
    d.__dict__.update(base.__dict__)
    d.boxes = np.tile([.1,.2,.8,.9],(6,1)).astype(np.float32)
    d.valid = np.array([True,False,True,True,False,True])
    d.mode = mode
    return base,d


@pytest.mark.parametrize('mode',['box','box_coordinates_roi'])
def test_cached_and_online_inputs_match_preserving_every_other_observation_and_action(mode):
    base,d = fixture_dataset(mode)
    ids = [5,0,4,2,5,1]
    for i,sample in zip(ids,d.__getitems__(ids)):
        original=base[i]
        start,end,_,_=d.sampler.indices[i]
        frames=np.minimum(np.arange(start,start+2),end-1)
        online=replace_target_inputs({k:v[None] for k,v in original['obs'].items()},
            d.boxes[frames][None],d.valid[frames][None],mode)
        for k,v in sample['obs'].items():
            np.testing.assert_array_equal(v,online[k][0])
        for k in ('depth','proprio'):
            np.testing.assert_array_equal(sample['obs'][k],original['obs'][k])
        np.testing.assert_array_equal(sample['actions'],original['actions'])
        for k,v in d[i]['obs'].items():
            np.testing.assert_array_equal(v,sample['obs'][k])
        if mode == 'box_coordinates_roi':
            _,roi,_=object_inputs(original['obs']['rgb'][-1],d.boxes[frames[-1]],d.valid[frames[-1]])
            np.testing.assert_array_equal(sample['obs']['_roi'],roi)
    batch=next(iter(torch.utils.data.DataLoader(d,batch_sampler=[ids],num_workers=0)))
    assert batch['obs']['rgb'].shape[:2] == (len(ids),2)


def test_wrist_observations_unchanged_missing_latest_masks_objects_and_no_input_mutation():
    rgb=np.random.default_rng(5).random((2,2,3,90,120),dtype=np.float32)
    boxes=np.tile([.1,.2,.7,.8],(2,2,1))
    valid=np.array([[True,False],[False,True]])
    wrist=rgb.copy();depth=np.ones((2,2,1,90,120),np.float32)
    result=replace_target_inputs(dict(rgb=rgb,rgb_wrist=wrist,depth_wrist=depth),boxes,valid,'box_coordinates_roi')
    np.testing.assert_array_equal(rgb,wrist)
    assert result['rgb_wrist'] is wrist and result['depth_wrist'] is depth
    assert result['_bbox_valid'][0] == 0 and not result['_roi'][0].any()
    np.testing.assert_array_equal(result['rgb'][0,1],rgb[0,1])
    assert not np.array_equal(result['rgb'][0,0],rgb[0,0])
    with pytest.raises(ValueError,match='Every observation'):
        replace_target_inputs(dict(rgb=rgb),boxes[:,:1],valid,'box')
    boxes[1,1,2]=0
    with pytest.raises(ValueError,match='normalized'):
        replace_target_inputs(dict(rgb=rgb),boxes,valid,'box')


def test_cache_rejects_old_incomplete_wrong_selection_and_wrong_mapping(tmp_path,monkeypatch):
    _,d=fixture_dataset('box')
    source=tmp_path/'test.zarr'
    def fake_init(self,*args,**kwargs): self.__dict__.update(d.__dict__)
    monkeypatch.setattr(ZarrNoHistorySequenceDataset,'__init__',fake_init)
    # The fake sequence dataset only needs real metadata for signature checks.
    d.source_frame_map=np.arange(6);d._demo_id_to_start_indices={'one':0,'two':1}
    cache=zarr.open_group(str(tmp_path/'cache.zarr'),mode='w')
    cache.array('boxes',d.boxes);cache.array('valid',d.valid)
    meta=metadata()
    cache.attrs.update(dict(complete=True,completed_frames=6,target_metadata=meta,signature=source_signature(d,source)))
    def load():return TargetNoHistoryDataset(source,d.obs_keys,9,2,20,tmp_path/'cache.zarr','box',meta)
    assert len(load()) == 6
    cache.attrs['complete']=False
    with pytest.raises(ValueError,match='mismatched'):load()
    cache.attrs['complete']=True
    cache.attrs['target_metadata']=dict(meta,bundle_manifest_sha256='c'*64)
    with pytest.raises(ValueError,match='mismatched'):load()
    cache.attrs['target_metadata']=meta
    cache.attrs['signature']=dict(source_signature(d,source),source_frame_map_sha256='x')
    with pytest.raises(ValueError,match='mismatched'):load()


def test_replacement_configs_keep_old_modalities_all_ten_sources_and_scientific_settings():
    root=Path(__file__).resolve().parents[1]
    base=OmegaConf.load(root/'momart_boxed_no_history/config_bs256_150k.yaml')
    configs=replacement_configs(base,metadata(),'/new/boxes','/new/runs')
    allowed={'history_method','work_dir','bbox_root','target_mode','target_metadata'}
    for cfg in configs.values():
        for key in base:
            if key not in allowed:
                assert OmegaConf.to_container(OmegaConf.create({key:cfg[key]}),resolve=False) == OmegaConf.to_container(OmegaConf.create({key:base[key]}),resolve=False)
        assert len(cfg.task_datasets) == 10
        assert 'rgb_wrist' in cfg.shape_meta.obs and 'depth_wrist' in cfg.shape_meta.obs
    from momart_target_no_history.train import build_suite
    with pytest.raises(ValueError,match='does not cover'):build_suite(configs['box'])


def test_original_current_object_transformer_masks_missing_and_earliest_action():
    from momart_target_no_history.legacy_tokens import CurrentObjectTransformer
    args=SimpleNamespace(action_dim=10,horizon=9,obs_steps=2,diffusion_num_layers=1,p_drop_emb=0.,p_drop_attn=0.)
    torch.manual_seed(3)
    net=CurrentObjectTransformer(args).eval()
    x=torch.randn(2,9,10);t=torch.ones(2);c=torch.randn(2,2,769);c[...,-1]=1
    changed=c.clone();changed[...,256:768]+=3
    with torch.no_grad():
        before=net(x,t,c);after=net(x,t,changed)
    torch.testing.assert_close(before[:,0],after[:,0])
    assert (before[:,1:]-after[:,1:]).abs().max() > 1e-6
    c[...,-1]=0;changed[...,-1]=0
    with torch.no_grad():torch.testing.assert_close(net(x,t,c),net(x,t,changed))


def test_both_full_multimodal_policy_structures_forward_and_backward():
    from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer
    from momart_no_history_baseline.model import build_agent as build_box
    from momart_target_no_history.legacy_tokens import build_agent as build_tokens
    root=Path(__file__).resolve().parents[1]
    cfg=OmegaConf.load(root/'momart_boxed_no_history/config_bs256_150k.yaml')
    cfg.device='cpu'
    normalizer={'action':MinMaxNormalizer(np.stack([-np.ones(10),np.ones(10)]),device='cpu')}
    original_threads=torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for mode,builder in [('box',build_box),('box_coordinates_roi',build_tokens)]:
            agent=builder(cfg,normalizer)
            inputs={k:torch.rand(1,2,*v.shape) for k,v in cfg.shape_meta.obs.items()}
            if mode != 'box':
                inputs.update(_bbox=torch.tensor([[.1,.2,.7,.8]]),_roi=torch.rand(1,3,64,64),_bbox_valid=torch.ones(1))
            inputs['_task_id']=torch.zeros(1,dtype=torch.long)
            condition=agent.model['condition'](inputs)
            assert condition.shape == (1,2,256 if mode == 'box' else 769)
            action=agent.model['diffusion'](torch.randn(1,9,10),torch.ones(1),condition)
            assert action.shape == (1,9,10) and torch.isfinite(action).all()
            action.square().mean().backward()
            encoder=agent.model['condition']
            if mode != 'box':encoder=encoder.encoder
            for camera in ('rgb','rgb_wrist','depth','depth_wrist'):
                assert any(p.grad is not None for p in encoder.encoder.key_model_map[camera].parameters())
            del agent
    finally:
        torch.set_num_threads(original_threads)
