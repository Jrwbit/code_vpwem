from pathlib import Path
from typing import Dict

import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.dataset_utils import (
    EmptyNormalizer,
    ImageNormalizer,
    LongMemSequenceSampler,
    MinMaxNormalizer,
    SequenceSampler,
    SortedByLengthSampler,
    dict_apply,
)
from cleandiffuser.dataset.replay_buffer import ReplayBuffer


def resolve_replay_buffer_path(dataset_path: str) -> Path:
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    if path.is_dir() and path.suffix == ".zarr":
        return path
    candidate = path / "replay_buffer.zarr"
    if candidate.is_dir():
        return candidate
    if (path / "data").is_dir() and (path / "meta").is_dir():
        return path
    raise FileNotFoundError(
        f"Cannot find replay buffer zarr from dataset_path={dataset_path}. "
        "Expected '<path>.zarr', '<path>/replay_buffer.zarr', or a zarr root "
        "containing data/ and meta/."
    )


class RealZarrImageDataset(BaseDataset):
    """Image imitation dataset backed by a Diffusion Policy-style replay zarr.

    The dataset is intentionally task-agnostic: any observation key declared in
    shape_meta is read from zarr/data. This supports two-camera and three-camera
    real-robot tasks through config only.
    """

    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        action_key="action",
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.action_key = action_key

        zarr_path = resolve_replay_buffer_path(dataset_dir)
        obs_shape_meta = shape_meta["obs"]
        rgb_keys = []
        lowdim_keys = []
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type '{obs_type}' for key '{key}'")

        keys = [action_key] + rgb_keys + lowdim_keys
        self.replay_buffer = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
        missing = [key for key in keys if key not in self.replay_buffer]
        if missing:
            raise KeyError(f"Missing keys in replay buffer {zarr_path}: {missing}")

        action_shape = tuple(shape_meta["action"]["shape"])
        actual_action_shape = tuple(self.replay_buffer[action_key].shape[1:])
        if actual_action_shape != action_shape:
            raise ValueError(
                f"Action shape mismatch: config={action_shape}, zarr={actual_action_shape}"
            )

        for key in rgb_keys + lowdim_keys:
            expected_shape = tuple(obs_shape_meta[key]["shape"])
            actual_shape = tuple(self.replay_buffer[key].shape[1:])
            if key in rgb_keys:
                # Config uses torch image layout [C,H,W], zarr stores [H,W,C].
                expected_shape = (expected_shape[1], expected_shape[2], expected_shape[0])
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Observation '{key}' shape mismatch: "
                    f"config={expected_shape}, zarr={actual_shape}"
                )

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=keys,
            key_first_k=key_first_k,
        )

        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        normalizer = {"obs": {}}
        for key in self.lowdim_keys:
            if key.endswith("emb"):
                normalizer["obs"][key] = EmptyNormalizer()
            else:
                normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer()
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer[self.action_key][:])
        return normalizer

    def __str__(self) -> str:
        return (
            f"Keys: {self.replay_buffer.keys()} "
            f"Steps: {self.replay_buffer.n_steps} "
            f"Episodes: {self.replay_buffer.n_episodes}"
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            obs = np.moveaxis(sample[key][t_slice], -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs)

        for key in self.lowdim_keys:
            obs = sample[key][t_slice].astype(np.float32)
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs)

        action = sample[self.action_key].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        return {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
        }


class LongMemRealZarrImageDataset(BaseDataset):
    """Long-memory variant for replay-buffer zarr datasets with precomputed embeddings."""

    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        action_key="action",
        dataset_memory_type="random_from_full",
        subsample_ratio=10,
        maximum_memory_length=16,
        device="cpu",
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.action_key = action_key
        self.device = device

        zarr_path = resolve_replay_buffer_path(dataset_dir)
        obs_shape_meta = shape_meta["obs"]
        rgb_keys = []
        lowdim_keys = []
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type '{obs_type}' for key '{key}'")

        keys = [action_key] + rgb_keys + lowdim_keys
        self.replay_buffer = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
        missing = [key for key in keys if key not in self.replay_buffer]
        if missing:
            raise KeyError(f"Missing keys in replay buffer {zarr_path}: {missing}")

        action_shape = tuple(shape_meta["action"]["shape"])
        actual_action_shape = tuple(self.replay_buffer[action_key].shape[1:])
        if actual_action_shape != action_shape:
            raise ValueError(
                f"Action shape mismatch: config={action_shape}, zarr={actual_action_shape}"
            )

        for key in rgb_keys + lowdim_keys:
            expected_shape = tuple(obs_shape_meta[key]["shape"])
            actual_shape = tuple(self.replay_buffer[key].shape[1:])
            if key in rgb_keys:
                expected_shape = (expected_shape[1], expected_shape[2], expected_shape[0])
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Observation '{key}' shape mismatch: "
                    f"config={expected_shape}, zarr={actual_shape}"
                )

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = LongMemSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=keys,
            key_first_k=key_first_k,
            memory_type=dataset_memory_type,
            subsample_ratio=subsample_ratio,
            maximum_memory_length=maximum_memory_length,
        )

        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.normalizer = self.get_normalizer(device=device)

    def get_normalizer(self, device=None):
        normalizer = {"obs": {}}
        for key in self.lowdim_keys:
            if key.endswith("emb"):
                normalizer["obs"][key] = EmptyNormalizer()
            else:
                normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:], device=device)
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer(device=device)
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer[self.action_key][:], device=device)
        return normalizer

    def __str__(self) -> str:
        return (
            f"Keys: {self.replay_buffer.keys()} "
            f"Steps: {self.replay_buffer.n_steps} "
            f"Episodes: {self.replay_buffer.n_episodes}"
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(sample[key][t_slice], -1, 1).astype(np.float32) / 255.0
            obs_dict[f"{key}_mem"] = np.moveaxis(sample[f"{key}_mem"], -1, 1).astype(np.float32) / 255.0

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][t_slice].astype(np.float32)
            obs_dict[f"{key}_mem"] = sample[f"{key}_mem"].astype(np.float32)

        obs_dict["ep_step"] = sample["ep_step"].astype(np.int32)
        obs_dict["actual_num_frames"] = sample.get(
            "actual_num_frames", len(sample["ep_step"])
        )

        action = sample[self.action_key].astype(np.float32)
        return {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
        }

    def get_dataset_sampler(self):
        return SortedByLengthSampler(self)
