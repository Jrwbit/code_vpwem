"""MoMaRT samples containing only the observations used by No-History."""

import json

import numpy as np
import torch
import zarr

import robomimic.utils.obs_utils as ObsUtils

from cleandiffuser.dataset.dataset_utils import SequenceSampler
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.utils.robomimic_dataset_utils import SortedByLengthSampler


class BoxedNoHistoryDataset(torch.utils.data.Dataset):
    """Read the two current observations from a persistent flat Zarr cache."""

    def __init__(self, zarr_path, obs_keys, seq_length, obs_steps,
                 subsample_ratio, bbox_path):
        root = zarr.open_group(str(zarr_path), mode="r")
        from momart_no_history_baseline.cache import training_replay
        self.replay_buffer, demo_names, self.source_frame_map = training_replay(root)
        import hashlib
        from pathlib import Path
        from momart_bbox_memory.detector import MODEL_ID, REVISION
        self.bbox = zarr.open_group(str(bbox_path), mode="r")
        signature = self.bbox.attrs.get("signature", {})
        expected = dict(source=str(Path(zarr_path).resolve()),
            shape=list(self.replay_buffer["rgb"].shape),
            episode_ends_sha256=hashlib.sha256(
                np.asarray(self.replay_buffer.episode_ends).tobytes()).hexdigest())
        if (self.bbox.attrs.get("complete") is not True or
            int(self.bbox.attrs.get("completed_frames", -1)) != len(self.replay_buffer["rgb"]) or
            any(signature.get(k) != v for k,v in expected.items())):
            raise ValueError("Incomplete or mismatched bbox cache")
        detector = signature.get("detector", {})
        standard = self.bbox.attrs.get("annotation_standard", "")
        if standard != "simulator_visible_mask_v1":
            if (detector.get("model_id") != MODEL_ID or detector.get("revision") != REVISION
                or detector.get("prompt") != "a red bowl."):
                raise ValueError("Unexpected bbox detector")
        if "demo_names" in self.bbox.attrs and list(self.bbox.attrs["demo_names"]) != demo_names:
            raise ValueError("BBox training demo identities differ")
        self.detector_metadata = dict(detector)
        if standard == "quality_gate_v1":
            self.detector_metadata["quality_gate"] = dict(
                version=standard, threshold=float(self.bbox.attrs["quality_threshold"]))
        self.boxes = np.asarray(self.bbox["boxes"])
        self.valid = np.asarray(self.bbox["valid"])
        if self.boxes.shape != (len(self.replay_buffer["rgb"]), 4) or self.valid.shape != (len(self.boxes),):
            raise ValueError("Incorrect bbox array shapes")
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
        from .painting import draw_bbox
        buffer_start, buffer_end, _, _ = self.sampler.indices[index]
        frame_indices = np.minimum(
            np.arange(buffer_start, buffer_start+self.obs_steps), buffer_end-1)
        boxed = sample["rgb"][:self.obs_steps].copy()
        for offset, frame in enumerate(frame_indices):
            boxed[offset] = draw_bbox(boxed[offset], self.boxes[int(frame)],
                                     bool(self.valid[int(frame)]))
        sample["rgb"] = boxed
        observations = ObsUtils.process_obs_dict({
            key: sample[key][:self.obs_steps] for key in self.obs_keys
        })
        return {
            "obs": observations,
            "actions": sample["action"].astype(np.float32),
        }

    def get_dataset_sampler(self):
        return SortedByLengthSampler(self)
