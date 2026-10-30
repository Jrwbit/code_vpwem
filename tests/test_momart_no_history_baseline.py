"""Check the extracted baseline keeps the old policy interface and skips history I/O."""

import numpy as np
import torch

from momart_no_history_baseline.dataset import (
    NoHistorySequenceDataset,
    ZarrNoHistorySequenceDataset,
)
from momart_no_history_baseline.model import TaskConditionedObsEncoder
from cleandiffuser.nn_condition.momart_history_condition import (
    TaskConditionedObsEncoder as OriginalTaskConditionedObsEncoder,
)


def test_no_history_reads_only_two_recent_observations(monkeypatch):
    dataset = object.__new__(NoHistorySequenceDataset)
    dataset.obs_steps = 2
    dataset._hdf5_file = None
    calls = []

    def get_sequence(demo, index, keys, num_frames_to_stack, seq_length):
        calls.append((demo, index, keys, seq_length))
        return {"obs/proprio": np.zeros((2, 10), dtype=np.float32)}, None

    monkeypatch.setattr(dataset, "get_sequence_from_demo", get_sequence)
    monkeypatch.setattr(
        "momart_no_history_baseline.dataset.ObsUtils.process_obs_dict",
        lambda obs: obs,
    )
    sample = dataset.get_obs_sequence_from_demo(
        "demo_0", 4, ("proprio",), seq_length=9)

    assert calls == [("demo_0", 4, ("obs/proprio",), 2)]
    assert tuple(sample["proprio"].shape) == (2, 10)
    assert not any(key.endswith("_mem") for key in sample)


def test_extracted_task_encoder_matches_existing_checkpoint_layout():
    class IdentityEncoder(torch.nn.Module):
        def forward(self, obs, mask=None):
            return obs["features"]

    torch.manual_seed(100)
    extracted = TaskConditionedObsEncoder(IdentityEncoder(), 5)
    torch.manual_seed(100)
    original = OriginalTaskConditionedObsEncoder(IdentityEncoder(), 5)
    features = torch.randn(3, 2, 256)
    obs = {"features": features, "_task_id": torch.tensor([0, 2, 4])}

    assert extracted.state_dict().keys() == original.state_dict().keys()
    torch.testing.assert_close(extracted(obs), original(obs))


def test_zarr_no_history_dataset_is_available():
    assert ZarrNoHistorySequenceDataset.__name__ == "ZarrNoHistorySequenceDataset"
