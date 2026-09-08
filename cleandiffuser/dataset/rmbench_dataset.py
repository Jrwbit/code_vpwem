import io
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
import zarr
from PIL import Image

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.dataset_utils import (
    EmptyNormalizer,
    ImageNormalizer,
    LongMemSequenceSampler,
    MinMaxNormalizer,
    SequenceSampler,
    dict_apply,
)
from cleandiffuser.dataset.replay_buffer import ReplayBuffer


def _decode_rgb_frame(frame):
    if isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 3:
        return frame
    if isinstance(frame, np.ndarray):
        frame = frame.tobytes()
    if isinstance(frame, np.bytes_):
        frame = bytes(frame)
    with Image.open(io.BytesIO(frame)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_hdf5_path(file, path):
    node = file
    for part in path.split("/"):
        node = node[part]
    return node


def _episode_files(dataset_dir):
    root = Path(dataset_dir).expanduser()
    data_dir = root / "data"
    if data_dir.is_dir():
        files = sorted(data_dir.glob("episode*.hdf5"))
    else:
        files = sorted(root.glob("episode*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No RMBench episode*.hdf5 files found under {root}")
    return files


def _load_rgb_dataset(dataset):
    return np.stack([_decode_rgb_frame(dataset[i]) for i in range(dataset.shape[0])], axis=0)


def _is_replay_zarr_path(path):
    path = Path(path).expanduser()
    return path.is_dir() and (
        path.suffix == ".zarr" or ((path / "data").is_dir() and (path / "meta").is_dir())
    )


def _validate_and_collect_episode_metadata(dataset_dir, action_source, shift_action=False):
    episode_files = _episode_files(dataset_dir)
    episode_lengths = []
    episode_ends = []
    total = 0
    for episode_file in episode_files:
        with h5py.File(episode_file, "r") as f:
            length = _read_hdf5_path(f, action_source).shape[0]
        if shift_action:
            if length < 2:
                raise ValueError(f"{episode_file} has length={length}; need at least 2 frames for shifted actions")
            length -= 1
        total += length
        episode_lengths.append(length)
        episode_ends.append(total)
    return episode_files, episode_lengths, episode_ends, total


def _write_rmbench_replay_group(
    root,
    shape_meta,
    dataset_dir,
    action_source="joint_action/vector",
    chunk_length=None,
    compressor=None,
    shift_action=False,
):
    rgb_keys = []
    lowdim_keys = []
    obs_meta = shape_meta["obs"]
    for key, attr in obs_meta.items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            rgb_keys.append(key)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)

    episode_files, _, episode_ends, total = _validate_and_collect_episode_metadata(
        dataset_dir, action_source, shift_action=shift_action
    )
    if chunk_length is None:
        chunk_length = total
    chunk_length = max(1, min(int(chunk_length), int(total)))

    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)
    meta_group.array("episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True)

    action_shape = tuple(shape_meta["action"]["shape"])
    data_group.zeros(
        name="action",
        shape=(total,) + action_shape,
        chunks=(chunk_length,) + action_shape,
        compressor=compressor,
        dtype=np.float32,
        overwrite=True,
    )
    for key in lowdim_keys:
        shape = tuple(obs_meta[key]["shape"])
        data_group.zeros(
            name=key,
            shape=(total,) + shape,
            chunks=(chunk_length,) + shape,
            compressor=compressor,
            dtype=np.float32,
            overwrite=True,
        )
    for key in rgb_keys:
        c, h, w = tuple(obs_meta[key]["shape"])
        shape = (h, w, c)
        data_group.zeros(
            name=key,
            shape=(total,) + shape,
            chunks=(chunk_length,) + shape,
            compressor=compressor,
            dtype=np.uint8,
            overwrite=True,
        )

    offset = 0
    for episode_file in episode_files:
        with h5py.File(episode_file, "r") as f:
            action_full = _read_hdf5_path(f, action_source)[:].astype(np.float32)
            action = action_full[1:] if shift_action else action_full
            length = action.shape[0]
            data_group["action"][offset : offset + length] = action
            for key in lowdim_keys:
                source = obs_meta[key].get("source", key)
                value = _read_hdf5_path(f, source)[:].astype(np.float32)
                if shift_action:
                    value = value[:-1]
                if value.shape[0] != length:
                    raise ValueError(f"{episode_file} key {key} length {value.shape[0]} != action length {length}")
                data_group[key][offset : offset + length] = value
            for key in rgb_keys:
                source = obs_meta[key].get("source", key)
                value = _load_rgb_dataset(_read_hdf5_path(f, source))
                if shift_action:
                    value = value[:-1]
                if value.shape[0] != length:
                    raise ValueError(f"{episode_file} key {key} length {value.shape[0]} != action length {length}")
                data_group[key][offset : offset + length] = value

        offset += length


def convert_rmbench_to_replay_zarr(
    dataset_dir,
    output_path,
    shape_meta,
    action_source="joint_action/vector",
    chunk_length=100,
    overwrite=True,
    compressor=None,
    shift_action=False,
):
    output_path = Path(output_path).expanduser()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output zarr already exists: {output_path}")
        shutil.rmtree(output_path)
    if compressor is None:
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    root = zarr.group(str(output_path))
    _write_rmbench_replay_group(
        root=root,
        shape_meta=shape_meta,
        dataset_dir=dataset_dir,
        action_source=action_source,
        chunk_length=chunk_length,
        compressor=compressor,
        shift_action=shift_action,
    )
    return output_path


def _convert_rmbench_to_replay(store, shape_meta, dataset_dir, action_source="joint_action/vector", shift_action=False):
    root = zarr.group(store)
    _write_rmbench_replay_group(
        root=root,
        shape_meta=shape_meta,
        dataset_dir=dataset_dir,
        action_source=action_source,
        chunk_length=None,
        compressor=None,
        shift_action=shift_action,
    )

    return ReplayBuffer(root)


class RMBenchImageDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        action_source="joint_action/vector",
        shift_action=False,
        device="cpu",
    ):
        super().__init__()
        if _is_replay_zarr_path(dataset_dir):
            self.replay_buffer = ReplayBuffer.create_from_path(str(Path(dataset_dir).expanduser()), mode="r")
        else:
            self.replay_buffer = _convert_rmbench_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_dir=dataset_dir,
                action_source=action_source,
                shift_action=shift_action,
            )

        self.shape_meta = shape_meta
        self.rgb_keys = []
        self.lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k,
        )
        self.n_obs_steps = n_obs_steps
        self.normalizer = self.get_normalizer(device=device)

    def get_normalizer(self, device=None):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            if key.endswith("emb"):
                normalizer["obs"][key] = EmptyNormalizer()
            else:
                normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:], device=device)
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer(device=device)
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:], device=device)
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __str__(self):
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)
        obs_dict = {}

        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(sample[key][t_slice], -1, 1).astype(np.float32) / 255.0

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][t_slice].astype(np.float32)

        return {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(sample["action"].astype(np.float32)),
        }


class LongMemRMBenchImageDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        dataset_memory_type="random_from_full",
        subsample_ratio=10,
        maximum_memory_length=16,
        load_lowdim_to_memory=False,
        action_source="joint_action/vector",
        shift_action=False,
        device="cpu",
    ):
        super().__init__()
        if _is_replay_zarr_path(dataset_dir):
            self.replay_buffer = ReplayBuffer.create_from_path(str(Path(dataset_dir).expanduser()), mode="r")
        else:
            self.replay_buffer = _convert_rmbench_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_dir=dataset_dir,
                action_source=action_source,
                shift_action=shift_action,
            )

        self.shape_meta = shape_meta
        self.rgb_keys = []
        self.lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)

        if load_lowdim_to_memory:
            if self.rgb_keys:
                raise ValueError("load_lowdim_to_memory=True only supports low-dim observations")
            keys = ["action"] + self.lowdim_keys
            self.replay_buffer = ReplayBuffer({
                "meta": {"episode_ends": self.replay_buffer.episode_ends[:]},
                "data": {key: self.replay_buffer[key][:] for key in keys},
            })
            print(f"Loaded low-dim replay keys into memory: {keys}", flush=True)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = LongMemSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k,
            memory_type=dataset_memory_type,
            subsample_ratio=subsample_ratio,
            maximum_memory_length=maximum_memory_length,
        )
        self.n_obs_steps = n_obs_steps
        self.normalizer = self.get_normalizer(device=device)

    def get_normalizer(self, device=None):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            if key.endswith("emb"):
                normalizer["obs"][key] = EmptyNormalizer()
            else:
                normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:], device=device)
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer(device=device)
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:], device=device)
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def get_dataset_sampler(self):
        from cleandiffuser.dataset.dataset_utils import SortedByLengthSampler

        return SortedByLengthSampler(self)

    def __str__(self):
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
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
        obs_dict["actual_num_frames"] = np.asarray(sample.get("actual_num_frames", len(sample["ep_step"])), dtype=np.int32)

        return {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(sample["action"].astype(np.float32)),
        }
