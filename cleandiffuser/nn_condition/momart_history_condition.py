"""MOMART encoders for task-conditioned history ablations."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_nn_condition import BaseNNCondition
from .multi_image_condition import MultiImageObsConditionMomart


class TaskConditionedObsEncoder(BaseNNCondition):
    """Add a learned task embedding to every encoded observation token."""

    task_key = "_task_id"

    def __init__(self, encoder, num_tasks, emb_dim=256):
        super().__init__()
        self.encoder = encoder
        self.task_embedding = nn.Embedding(int(num_tasks), int(emb_dim))
        nn.init.normal_(self.task_embedding.weight, std=0.02)

    def _pop_task(self, obs):
        obs = dict(obs)
        if self.task_key not in obs:
            raise KeyError(f"missing {self.task_key!r} in task-conditioned input")
        task_id = obs.pop(self.task_key).long()
        if task_id.ndim > 1:
            task_id = task_id.reshape(task_id.shape[0], -1)[:, 0]
        return obs, task_id

    def _condition(self, features, task_id):
        task = self.task_embedding(task_id).to(dtype=features.dtype)
        while task.ndim < features.ndim:
            task = task.unsqueeze(1)
        return features + task

    def forward(self, obs, mask=None):
        obs, task_id = self._pop_task(obs)
        return self._condition(self.encoder(obs, mask), task_id)


class TaskConditionedEmbeddingEncoder(BaseNNCondition):
    """Add a shared task embedding to precomputed MOMART observation tokens.

    The fair-history embedding suite passes dictionaries so the same learned
    task embedding is applied to both the current ``dp_emb`` tokens and the
    historical ``dp_emb_mem`` tokens.  Direct-history variants optionally
    normalize the history to a fixed frame count and pool it to a fixed token
    budget, matching the raw-observation fair-history protocol.
    """

    task_key = "_task_id"
    embedding_key = "dp_emb"

    def __init__(self, num_tasks, emb_dim=256, history_frames=None,
                 history_output_tokens=0, history_subsample_ratio=1):
        super().__init__()
        self.task_embedding = nn.Embedding(int(num_tasks), int(emb_dim))
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        self.history_frames = (
            None if history_frames is None else int(history_frames))
        self.history_output_tokens = int(history_output_tokens or 0)
        self.history_subsample_ratio = int(history_subsample_ratio)

    def _features_and_task(self, obs):
        obs = dict(obs)
        if self.task_key not in obs:
            raise KeyError(f"missing {self.task_key!r} in task-conditioned input")
        if self.embedding_key not in obs:
            raise KeyError(
                f"missing {self.embedding_key!r} in embedding input")
        features = obs[self.embedding_key]
        task_id = obs[self.task_key].long()
        if task_id.ndim > 1:
            task_id = task_id.reshape(task_id.shape[0], -1)[:, 0]
        task = self.task_embedding(task_id).to(dtype=features.dtype)
        while task.ndim < features.ndim:
            task = task.unsqueeze(1)
        return features + task

    def forward(self, obs, mask=None):
        return self._features_and_task(obs)

    def _fixed_history(self, features):
        if self.history_frames is None:
            return features
        if features.shape[1] > self.history_frames:
            features = torch.cat([
                features[:, :1],
                features[:, -(self.history_frames - 1):],
            ], dim=1)
        elif features.shape[1] < self.history_frames:
            if features.shape[1] == 0:
                features = features.new_zeros(
                    features.shape[0], self.history_frames,
                    features.shape[-1])
            else:
                pad = features[:, :1].expand(
                    -1, self.history_frames - features.shape[1], -1)
                features = torch.cat([pad, features], dim=1)
        return features

    def encode_spatial_history(self, obs):
        return self.pack_history(self.encode_spatial_frames(obs))

    def encode_spatial_frames(self, obs):
        """Expose embedding frames to the online direct-history path."""
        return self._features_and_task(obs)

    def pack_history(self, features):
        """Apply the same fixed-budget packing used during training."""
        features = self._fixed_history(features)
        if self.history_output_tokens:
            features = F.adaptive_avg_pool1d(
                features.transpose(1, 2), self.history_output_tokens
            ).transpose(1, 2)
        return features


class TaskConditionedDirectObsEncoder(TaskConditionedObsEncoder):
    """Task-conditioned wrapper that preserves the Direct spatial API."""

    @property
    def history_frames(self):
        return self.encoder.history_frames

    @property
    def history_subsample_ratio(self):
        return self.encoder.history_subsample_ratio

    @property
    def history_token_count(self):
        return self.encoder.history_token_count

    def encode_spatial_frames(self, obs):
        obs, task_id = self._pop_task(obs)
        return self._condition(
            self.encoder.encode_spatial_frames(obs), task_id)

    def pack_history(self, spatial_frames):
        return self.encoder.pack_history(spatial_frames)

    def encode_spatial_history(self, obs):
        return self.pack_history(self.encode_spatial_frames(obs))


class DirectSpatialHistoryConditionMomart(MultiImageObsConditionMomart):
    """Keep 4x4 spatial history from MOMART RGB and depth cameras.

    The current-observation path remains a global embedding over every MOMART
    modality. Historical RGB/depth frames retain a spatial grid; proprio and
    scan history are intentionally excluded from the Direct token sequence.
    """

    def __init__(self, *args, pool_size=4, history_frames=16,
                 history_output_tokens=0, history_subsample_ratio=20,
                 **kwargs):
        # MultiImageObsConditionMomart probes output_shape during its own
        # constructor. Until the spatial trunks below exist, use its original
        # global forward for that probe.
        self._direct_ready = False
        super().__init__(*args, **kwargs)
        self.pool_size = int(pool_size)
        self.history_frames = int(history_frames)
        self.history_output_tokens = int(history_output_tokens or 0)
        self.history_subsample_ratio = int(history_subsample_ratio)
        self.spatial_dim = self.mlp[-1].out_features
        self.spatial_keys = tuple(self.rgb_keys + self.depth_keys)

        projections = nn.ModuleDict()
        for key in self.spatial_keys:
            model = self.key_model_map[key]
            children = list(model.children())
            if len(children) < 2:
                raise ValueError(f"visual model for {key!r} has no spatial trunk")
            trunk = nn.Sequential(*children[:-2])
            self.key_model_map[key] = trunk
            with torch.no_grad():
                example = torch.zeros(1, *self.key_shape_map[key])
                example = self.key_transform_map[key](example)
                channels = int(trunk(example).shape[1])
            projections[key] = nn.Linear(channels, self.spatial_dim)
        self.spatial_projections = projections

        per_view = self.pool_size * self.pool_size
        self.spatial_pos_emb = nn.Parameter(
            torch.zeros(1, 1, per_view, self.spatial_dim))
        self.view_emb = nn.Parameter(torch.zeros(
            1, 1, len(self.spatial_keys), 1, self.spatial_dim))
        nn.init.normal_(self.spatial_pos_emb, std=0.02)
        nn.init.normal_(self.view_emb, std=0.02)
        self._direct_ready = True

    @property
    def tokens_per_frame(self):
        return len(self.spatial_keys) * self.pool_size * self.pool_size

    @property
    def history_token_count(self):
        return (self.history_output_tokens or
                self.history_frames * self.tokens_per_frame)

    def multi_image_forward(self, input_obs_dict):
        if not self._direct_ready:
            return super().multi_image_forward(input_obs_dict)
        obs = dict(input_obs_dict)
        if self.use_seq:
            obs = {key: value.flatten(end_dim=1)
                   for key, value in obs.items()}
        features = []
        for key in self.spatial_keys:
            image = self.key_transform_map[key](obs[key])
            fmap = self.key_model_map[key](image)
            features.append(F.adaptive_avg_pool2d(fmap, 1).flatten(1))
        for key in self.low_dim_keys:
            features.append(obs[key])
        for key in self.scan_keys:
            features.append(self.key_model_map[key](obs[key]))
        return torch.cat(features, dim=-1)

    def encode_spatial_frames(self, obs_dict):
        batch_size, seq_len = self.get_batch_size(obs_dict)
        view_tokens = []
        for view_index, key in enumerate(self.spatial_keys):
            image = obs_dict[key].flatten(end_dim=1)
            image = self.key_transform_map[key](image)
            fmap = self.key_model_map[key](image)
            pooled = F.adaptive_avg_pool2d(
                fmap, (self.pool_size, self.pool_size))
            tokens = pooled.flatten(2).transpose(1, 2)
            tokens = self.spatial_projections[key](tokens)
            tokens = tokens.reshape(
                batch_size, seq_len, -1, self.spatial_dim)
            tokens = tokens + self.spatial_pos_emb
            tokens = tokens + self.view_emb[:, :, view_index]
            view_tokens.append(tokens)
        return torch.cat(view_tokens, dim=2)

    def pack_history(self, spatial_frames):
        if spatial_frames.ndim != 4:
            raise ValueError("spatial history must have shape [B,T,N,D]")
        if spatial_frames.shape[1] > self.history_frames:
            frames = torch.cat([
                spatial_frames[:, :1],
                spatial_frames[:, -(self.history_frames - 1):],
            ], dim=1)
        else:
            frames = spatial_frames
        if frames.shape[1] < self.history_frames:
            if frames.shape[1] == 0:
                frames = spatial_frames.new_zeros(
                    spatial_frames.shape[0], self.history_frames,
                    self.tokens_per_frame, self.spatial_dim)
            else:
                pad = frames[:, :1].expand(
                    -1, self.history_frames - frames.shape[1], -1, -1)
                frames = torch.cat([pad, frames], dim=1)
        tokens = frames.flatten(1, 2)
        if self.history_output_tokens:
            tokens = F.adaptive_avg_pool1d(
                tokens.transpose(1, 2), self.history_output_tokens
            ).transpose(1, 2)
        return tokens

    def encode_spatial_history(self, obs_dict):
        return self.pack_history(
            self.encode_spatial_frames(copy.copy(obs_dict)))
