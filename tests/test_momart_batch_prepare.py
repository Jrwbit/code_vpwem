"""Batch tails and committed-frame resumption must preserve frame order."""
from types import SimpleNamespace
import numpy as np
import pytest
from omegaconf import OmegaConf
from momart_object_memory import prepare_train as module


@pytest.mark.parametrize('batch_size', [16,64,128,256])
def test_batched_prepare_resumes_committed_frames_and_keeps_tail(tmp_path, monkeypatch, batch_size):
    cfg = OmegaConf.create(dict(data_root=str(tmp_path), object_memory=dict(
        cache_root=str(tmp_path/'cache'), feature_dim=128, inference_batch_size=batch_size)))
    images = np.zeros((35,90,120,3), np.uint8)
    images[:,0,0,0] = np.arange(35)
    monkeypatch.setattr(module, 'validate_perception_metadata', lambda *args: None)
    monkeypatch.setattr(module.zarr, 'open_group', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'training_replay', lambda group: ({'rgb':images}, [], {}))
    monkeypatch.setattr(module, 'replay_signature', lambda *args: {'frames':35})
    batches = []
    def batch(perceive, rgb):
        batches.append(rgb[:,0,0,0].tolist())
        return [[] for _ in rgb]
    monkeypatch.setattr(module, 'perceive_batch', batch)
    def write(*args, observation_reader, **kwargs):
        assert all(observation_reader(i) == [] for i in range(35))
    monkeypatch.setattr(module, 'write_cache', write)
    p = SimpleNamespace(metadata={})
    module.prepare_dataset(cfg, 'example', p, tmp_path/'status.json')
    assert batches == [list(range(start,min(start+batch_size,35))) for start in range(0,35,batch_size)]
    batches.clear()
    module.prepare_dataset(cfg, 'example', p, tmp_path/'status.json')
    assert batches == []
