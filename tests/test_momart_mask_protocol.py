"""Mask protocol regressions without downloading weights or running GDINO."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from momart_object_memory.perception import mask_perception_contract, validate_perception_metadata, file_sha256
from momart_object_memory.mask_detector import select_boxes


def config(tmp_path):
    cfg=OmegaConf.load(Path(__file__).parents[1]/'momart_object_memory/config.yaml').object_memory
    for name in ('detector','feature'):
        path=tmp_path/(name+'.pt');path.write_bytes(name.encode())
        cfg[name+'_checkpoint']=str(path);cfg[name+'_checkpoint_sha256']=file_sha256(path)
    return cfg


def test_mask_signature_rejects_old_detector_and_changed_parameters(tmp_path):
    cfg=config(tmp_path);metadata=mask_perception_contract(cfg)
    validate_perception_metadata(cfg,metadata)
    for changed in ({'version':'gdino_raw_layer2_roi_v1'},dict(metadata,thresholds=[.1]*6),
                    dict(metadata,feature_checkpoint_sha256='old'),dict(metadata,max_detections=16)):
        with pytest.raises(ValueError,match='protocol'):validate_perception_metadata(cfg,changed)
    cfg.thresholds[0]=.1
    with pytest.raises(ValueError,match='protocol'):validate_perception_metadata(cfg,metadata)
    cfg.detector_backend='gdino'
    with pytest.raises(ValueError,match='GDINO'):validate_perception_metadata(cfg,metadata)


@pytest.mark.parametrize('change',[{'slots':16},{'max_detections':16},{'thresholds':[.5]},
    {'thresholds':[float('nan')]*6},{'classes':['wrong']*6},{'feature_dim':256}])
def test_invalid_mask_protocol_fails_before_model_loading(tmp_path,change):
    cfg=config(tmp_path)
    for k,v in change.items():cfg[k]=v
    with pytest.raises(ValueError):mask_perception_contract(cfg)


def test_weights_must_match_frozen_checksum(tmp_path):
    cfg=config(tmp_path);Path(cfg.detector_checkpoint).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):mask_perception_contract(cfg)


def test_mask_threshold_class_mapping_clipping_and_nms_leave_rgb_untouched():
    rgb=np.zeros((90,120,3),np.uint8);before=rgb.copy()
    result=dict(boxes=torch.tensor([[-1.,10,30,40],[0.,10,30,40],[50.,10,70,40],[70.,10,90,40]]),
        labels=torch.tensor([6,6,5,3]),scores=torch.tensor([.9,.8,.7,.4]))
    selected=select_boxes(rgb,result,[.05,.6,.5,.9,.8,.35])
    assert len(selected)==1 and selected[0][0]==5
    np.testing.assert_allclose(selected[0][1],[0,10/90,30/120,40/90])
    np.testing.assert_array_equal(rgb,before)


def test_cache_writer_and_online_adapter_reject_legacy_metadata(tmp_path):
    from types import SimpleNamespace
    from momart_object_memory.cache import write_cache
    from momart_object_memory.online import OnlineObjects
    cfg=OmegaConf.create(dict(obs_steps=2,action_steps=8,object_memory=config(tmp_path)))
    old=SimpleNamespace(metadata={'version':'gdino_raw_layer2_roi_v1'})
    with pytest.raises(ValueError,match='Mask protocol'):
        write_cache(cfg,tmp_path/'absent.zarr',tmp_path/'objects.zarr',old)
    with pytest.raises(ValueError,match='Mask protocol'):
        OnlineObjects(cfg,old,1,None)


def test_evaluation_rejects_old_or_changed_stream_protocol(tmp_path):
    from momart_object_memory.evaluate import validate_experiment_protocol
    from momart_object_memory.main_visual import main_visual_contract
    from momart_object_memory.cache import stream_contract
    from momart_object_memory.train import comparison_contract
    cfg=OmegaConf.load(Path(__file__).parents[1]/'momart_object_memory/config.yaml')
    bundle=tmp_path/'bundle.json';bundle.write_text('{}')
    cfg.main_visual.precision_bundle=str(bundle);cfg.main_visual.precision_bundle_sha256=file_sha256(bundle)
    protocol=dict(protocol_version='independent_main_visual_v2',main_visual=main_visual_contract(cfg),
        comparison_contract=comparison_contract(cfg),mode=cfg.object_memory.mode,stream=stream_contract(cfg))
    validate_experiment_protocol(cfg,protocol)
    old=copy.deepcopy(protocol);old.pop('stream')
    with pytest.raises(ValueError,match='protocol'):validate_experiment_protocol(cfg,old)
    changed=copy.deepcopy(protocol);changed['stream']['slots']=16
    with pytest.raises(ValueError,match='protocol'):validate_experiment_protocol(cfg,changed)
