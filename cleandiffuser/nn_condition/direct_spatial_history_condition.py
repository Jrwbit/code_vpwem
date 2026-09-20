"""Shared-backbone direct spatial-history encoder for controlled ablations."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_image_condition import MultiImageObsConditionRobomimic


class DirectSpatialHistoryCondition(MultiImageObsConditionRobomimic):
    """Encode current observations globally and selected history spatially.

    The current ``[t-1,t]`` path is identical to the ordinary MIKASA DP
    encoder. Historical RGB frames use the same ResNet weights, before global
    pooling, and are pooled to ``pool_size x pool_size`` per camera. Setting
    ``history_output_tokens`` applies deterministic (non-learned) pooling to a
    fixed number of tokens; zero keeps every spatial token.
    """

    def __init__(self, *args, pool_size=4, history_frames=16,
                 history_output_tokens=0, history_subsample_ratio=5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_size = int(pool_size)
        self.history_frames = int(history_frames)
        self.history_output_tokens = int(history_output_tokens or 0)
        self.history_subsample_ratio = int(history_subsample_ratio)
        self.spatial_dim = self.mlp[-1].out_features

        trunks = nn.ModuleDict()
        projections = nn.ModuleDict()
        for key in self.rgb_keys:
            model = self.key_model_map[key]
            children = list(model.children())
            if len(children) < 2:
                raise ValueError(f"RGB model for {key!r} has no spatial trunk")
            trunk = nn.Sequential(*children[:-2])
            trunks[key] = trunk
            with torch.no_grad():
                example = torch.zeros(1, *self.key_shape_map[key])
                example = self.key_transform_map[key](example)
                channels = int(trunk(example).shape[1])
            projections[key] = nn.Linear(channels, self.spatial_dim)
        self.key_model_map = trunks
        self.spatial_projections = projections

        per_camera = self.pool_size * self.pool_size
        self.spatial_pos_emb = nn.Parameter(
            torch.zeros(1, 1, per_camera, self.spatial_dim))
        self.camera_emb = nn.Parameter(torch.zeros(
            1, 1, len(self.rgb_keys), 1, self.spatial_dim))
        nn.init.normal_(self.spatial_pos_emb, std=0.02)
        nn.init.normal_(self.camera_emb, std=0.02)

    @property
    def tokens_per_frame(self):
        return len(self.rgb_keys) * self.pool_size * self.pool_size

    @property
    def history_token_count(self):
        return (self.history_output_tokens or
                self.history_frames * self.tokens_per_frame)

    @torch.no_grad()
    def output_shape(self):
        example = {}
        for key, attr in self.shape_meta['obs'].items():
            prefix = (1, 1) if self.use_seq else (1,)
            example[key] = torch.zeros(
                prefix + tuple(attr['shape']), dtype=self.dtype,
                device=self.device)
        feature = MultiImageObsConditionRobomimic.multi_image_forward(
            self, example)
        return feature.shape[-1]

    def multi_image_forward(self, obs_dict):
        features = []
        if self.use_seq:
            for key in list(obs_dict):
                obs_dict[key] = obs_dict[key].flatten(end_dim=1)
        for key in self.rgb_keys:
            image = self.key_transform_map[key](obs_dict[key])
            fmap = self.key_model_map[key](image)
            features.append(F.adaptive_avg_pool2d(fmap, 1).flatten(1))
        for key in self.low_dim_keys:
            features.append(obs_dict[key])
        return torch.cat(features, dim=-1)

    def encode_spatial_frames(self, obs_dict):
        batch_size, seq_len = self.get_batch_size(obs_dict)
        camera_tokens = []
        for camera_index, key in enumerate(self.rgb_keys):
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
            tokens = tokens + self.camera_emb[:, :, camera_index]
            camera_tokens.append(tokens)
        return torch.cat(camera_tokens, dim=2)

    def pack_history(self, spatial_frames):
        if spatial_frames.ndim != 4:
            raise ValueError("spatial history must have shape [B,T,N,D]")
        if spatial_frames.shape[1] > self.history_frames:
            if self.history_frames == 1:
                frames = spatial_frames[:, :1]
            else:
                frames = torch.cat([
                    spatial_frames[:, :1],
                    spatial_frames[:, -(self.history_frames - 1):]
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
