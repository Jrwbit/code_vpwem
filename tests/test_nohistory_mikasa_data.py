import numpy as np

from cleandiffuser.dataset.dataset_utils import LongMemSequenceSampler
from cleandiffuser.dataset.mikasa_dataset_longmem import LongMemMikasaDataset


class ArrayReplay:
    def __init__(self):
        self.episode_ends = np.array([12, 24])
        self.data = {
            "rgb": np.arange(24 * 4 * 4 * 3, dtype=np.uint16)
            .reshape(24, 4, 4, 3).astype(np.uint8),
            "joints": np.arange(24 * 3, dtype=np.float32).reshape(24, 3),
            "action": np.arange(24 * 2, dtype=np.float32).reshape(24, 2),
        }

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key):
        return self.data[key]


def test_nohistory_sampler_keeps_current_window_and_omits_history():
    replay = ArrayReplay()
    kwargs = dict(
        replay_buffer=replay,
        sequence_length=9,
        pad_before=1,
        pad_after=7,
        key_first_k={"rgb": 2, "joints": 2},
        maximum_memory_length=1,
        history_sampling_seed=100,
    )
    regular = LongMemSequenceSampler(**kwargs)
    nohistory = LongMemSequenceSampler(**kwargs, include_history=False)

    for idx in (0, 1, 7, 12, len(regular) - 1):
        expected = regular.sample_sequence(idx)
        actual = nohistory.sample_sequence(idx)
        for key in ("rgb", "joints", "action"):
            np.testing.assert_array_equal(actual[key], expected[key])
        assert not any(key.endswith("_mem") for key in actual)
        np.testing.assert_array_equal(actual["ep_step"], [0.0])
        assert actual["actual_num_frames"] == 1


def test_nohistory_dataset_returns_no_memory_images():
    replay = ArrayReplay()
    dataset = object.__new__(LongMemMikasaDataset)
    dataset.sampler = LongMemSequenceSampler(
        replay_buffer=replay,
        sequence_length=9,
        pad_before=1,
        pad_after=7,
        key_first_k={"rgb": 2, "joints": 2},
        include_history=False,
    )
    dataset.rgb_keys = ["rgb"]
    dataset.lowdim_keys = ["joints"]
    dataset.n_obs_steps = 2
    dataset.include_history = False
    dataset.ncm_manifest = None

    sample = dataset[0]
    assert set(sample["obs"]) == {"rgb", "joints", "ep_step", "actual_num_frames"}
    assert sample["obs"]["rgb"].shape == (2, 3, 4, 4)
    assert sample["action"].shape == (9, 2)
