"""MoMaRT samples containing only the observations used by No-History."""

import json

import numpy as np
import torch
import zarr

import robomimic.utils.obs_utils as ObsUtils

from cleandiffuser.dataset.dataset_utils import SequenceSampler
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.utils.robomimic_dataset_utils import LongMemSequenceDataset
from cleandiffuser.utils.robomimic_dataset_utils import SortedByLengthSampler


class NoHistorySequenceDataset(LongMemSequenceDataset):
    """Keep the original action horizon and sampler, but read only To observations.

    The parent dataset's get_item calls this method for observations. Its
    standard implementation also reads and processes up to 16 history frames.
    No-History never consumes those frames, so this override avoids the HDF5
    reads, collation, and GPU transfers without changing action targets.
    """

    def __init__(self, *args, obs_steps, **kwargs):
        self.obs_steps = int(obs_steps)
        if self.obs_steps < 1 or self.obs_steps > int(kwargs["seq_length"]):
            raise ValueError("obs_steps must be between 1 and seq_length")
        super().__init__(*args, **kwargs)

    def get_obs_sequence_from_demo(
        self, demo_id, index_in_demo, keys, num_frames_to_stack=0,
        seq_length=1, prefix="obs",
    ):
        obs, _ = self.get_sequence_from_demo(
            demo_id, index_in_demo,
            keys=tuple(f"{prefix}/{key}" for key in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=min(seq_length, self.obs_steps),
        )
        return ObsUtils.process_obs_dict({
            key: obs[f"{prefix}/{key}"] for key in keys
        })


class ZarrNoHistorySequenceDataset(torch.utils.data.Dataset):
    """Read the two current observations from a persistent flat Zarr cache."""

    def __init__(self, zarr_path, obs_keys, seq_length, obs_steps,
                 subsample_ratio):
        root = zarr.open_group(str(zarr_path), mode="r")
        from .cache import training_replay
        self.replay_buffer, demo_names, self.source_frame_map = training_replay(root)
        self.obs_keys = tuple(obs_keys)
        self.obs_steps = int(obs_steps)
        self.seq_length = int(seq_length)
        keys = list(self.obs_keys) + ["action"]
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.seq_length,
            pad_before=0,
            pad_after=self.seq_length - 1,
            keys=keys,
            key_first_k={key: self.obs_steps for key in self.obs_keys},
        )
        self.subsample_ratio = int(subsample_ratio)
        self.pad_frame_stack = True
        self.n_frame_stack = 1
        ends = self.replay_buffer.episode_ends[:]
        self._index_to_demo_id = {}
        self._demo_id_to_start_indices = {}
        self._demo_id_to_demo_length = {}
        start = 0
        for demo, end in zip(demo_names, ends):
            end = int(end)
            self._demo_id_to_start_indices[demo] = start
            self._demo_id_to_demo_length[demo] = end - start
            for index in range(start, end):
                self._index_to_demo_id[index] = demo
            start = end
        self.normalizer = None

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        observations = ObsUtils.process_obs_dict({
            key: sample[key][:self.obs_steps] for key in self.obs_keys
        })
        return {
            "obs": observations,
            "actions": sample["action"].astype(np.float32),
        }

    def get_dataset_sampler(self):
        return SortedByLengthSampler(self)
