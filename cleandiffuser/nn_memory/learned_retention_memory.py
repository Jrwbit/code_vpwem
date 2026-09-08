from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory
from cleandiffuser.nn_memory.qformer_memory import QFormer
from cleandiffuser.utils.utils import SinusoidalEmbedding


class RetentionHead(nn.Module):
    """Predict hindsight action influence from an observation-history slot."""

    def __init__(self, emb_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim + 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def retention_features(memory: torch.Tensor, ep_step: torch.Tensor) -> torch.Tensor:
    """Causal slot features shared by offline teacher fitting and online use."""
    previous = torch.cat([memory[:, :1], memory[:, :-1]], dim=1)
    delta = memory - previous
    cosine_delta = 1.0 - F.cosine_similarity(memory, previous, dim=-1)
    latest = ep_step.max(dim=1, keepdim=True).values
    scale = latest.clamp_min(1).to(memory.dtype)
    age = (latest - ep_step).to(memory.dtype) / scale
    return torch.cat(
        [memory, delta, cosine_delta.unsqueeze(-1), age.unsqueeze(-1)], dim=-1)


class LearnedRetentionMemory(BaseNNMemory):
    """Fixed event bank ranked by a learned future-action influence predictor.

    The score head is fitted offline from leave-one-out action-denoising loss
    labels. The online policy only observes causal embeddings and slot ages.
    """

    def __init__(
            self,
            mem_compress_length: int = 2,
            emb_dim: int = 256,
            num_layers: int = 2,
            num_heads: int = 8,
            bank_size: int = 8,
            subsample_ratio: int = 20,
            pos_emb_scale: float = 1.0,
            retention_head_path: str = "",
    ):
        super().__init__()
        if bank_size < 1:
            raise ValueError("bank_size must be positive")
        self.mem_compress_length = mem_compress_length
        self.emb_dim = emb_dim
        self.bank_size = bank_size
        self.subsample_ratio = max(1, int(subsample_ratio))
        self.pos_emb_scale = float(pos_emb_scale)
        self.qformer = QFormer(
            input_dim=emb_dim,
            d_model=emb_dim,
            n_queries=mem_compress_length,
            n_layers=num_layers,
            nhead=num_heads,
            cache_compress="fifo",
            cache_max_length=bank_size,
        )
        self.retention_head = RetentionHead(emb_dim)
        if retention_head_path:
            state = torch.load(
                Path(retention_head_path).expanduser(),
                map_location="cpu", weights_only=False)
            self.retention_head.load_state_dict(
                state.get("retention_head", state), strict=True)
        self.mem_pos_emb = SinusoidalEmbedding(emb_dim)
        self.reset()

    def _select(
            self, memory: torch.Tensor, ep_step: torch.Tensor,
            valid_lengths: Optional[torch.Tensor] = None):
        batch, steps, _ = memory.shape
        scores = self.retention_head(retention_features(memory, ep_step))
        if valid_lengths is not None:
            positions = torch.arange(steps, device=memory.device)[None]
            valid = positions >= (steps - valid_lengths[:, None])
            scores = scores.masked_fill(~valid, -torch.inf)
        # Preserve the earliest valid observation as an episode anchor.
        first = (
            steps - valid_lengths if valid_lengths is not None
            else torch.zeros(batch, device=memory.device, dtype=torch.long)
        )
        anchor_bonus = torch.full_like(scores, -torch.inf)
        anchor_bonus.scatter_(1, first[:, None], torch.inf)
        scores = torch.maximum(scores, anchor_bonus)
        count = min(self.bank_size, steps)
        indices = scores.topk(count, dim=1).indices
        selected_steps = ep_step.gather(1, indices)
        order = selected_steps.argsort(dim=1)
        indices = indices.gather(1, order)
        selected_steps = selected_steps.gather(1, order)
        selected = memory.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
        return selected, selected_steps

    def _compress(self, bank: torch.Tensor, steps: torch.Tensor):
        positioned = bank + self.pos_emb_scale * self.mem_pos_emb(steps)
        return self.qformer(positioned)

    def forward(
            self, memory: torch.Tensor, ep_step: torch.Tensor,
            mask: torch.Tensor = None):
        valid_lengths = None
        if mask is not None and mask.ndim == 1:
            valid_lengths = mask.to(device=memory.device, dtype=torch.long)
        bank, steps = self._select(memory, ep_step, valid_lengths)
        return self._compress(bank, steps)

    def reset(self):
        self.step = 0
        self.bank = None
        self.bank_steps = None
        self.last_observation = None
        self.cp_tokens = None

    @torch.no_grad()
    def _append_and_prune(self, observation: torch.Tensor):
        batch = observation.shape[0]
        step = torch.full(
            (batch, 1), self.step, device=observation.device,
            dtype=torch.long)
        obs = observation[:, None]
        if self.bank is None:
            self.bank, self.bank_steps = obs, step
        else:
            self.bank = torch.cat([self.bank, obs], dim=1)
            self.bank_steps = torch.cat([self.bank_steps, step], dim=1)
        if self.bank.shape[1] > self.bank_size:
            self.bank, self.bank_steps = self._select(
                self.bank, self.bank_steps)

    def inference(self, new_memory: torch.Tensor):
        wrote = False
        for time_idx in range(new_memory.shape[1]):
            if self.step % self.subsample_ratio == 0:
                self._append_and_prune(new_memory[:, time_idx])
                wrote = True
            self.last_observation = new_memory[:, time_idx]
            self.step += 1
        if wrote or self.cp_tokens is None:
            self.cp_tokens = self._compress(self.bank, self.bank_steps)
        return self.cp_tokens
