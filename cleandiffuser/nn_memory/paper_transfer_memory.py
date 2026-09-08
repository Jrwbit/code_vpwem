"""Minimal, auditable transfers of recent robot-memory mechanisms to VPWEM.

These modules deliberately implement the smallest mechanism needed for a
controlled screening experiment.  They are *style* baselines, not verbatim
reproductions of the named papers.  Every module preserves VPWEM's fixed
``(batch, memory_tokens, embedding_dim)`` output contract.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory
from cleandiffuser.utils.utils import SinusoidalEmbedding


class _FixedTokenRecurrentMemory(BaseNNMemory):
    """Shared fixed-token recurrent encoder used by the screening variants."""

    def __init__(
            self, mem_compress_length: int = 4, emb_dim: int = 256,
            num_layers: int = 1, subsample_ratio: int = 1):
        super().__init__()
        if mem_compress_length < 1:
            raise ValueError("mem_compress_length must be positive")
        self.mem_compress_length = int(mem_compress_length)
        self.emb_dim = int(emb_dim)
        self.subsample_ratio = int(subsample_ratio)
        self.position = SinusoidalEmbedding(emb_dim)
        self.encoder = nn.GRU(
            emb_dim, emb_dim, num_layers=num_layers, batch_first=True)
        self.to_tokens = nn.Linear(
            emb_dim, self.mem_compress_length * emb_dim)
        self._aux_loss = torch.tensor(0.0)
        self.step = 0
        self.hidden: Optional[torch.Tensor] = None

    def _encode_inputs(
            self, memory: torch.Tensor, ep_step: Optional[torch.Tensor],
            **kwargs) -> torch.Tensor:
        del kwargs
        if ep_step is None:
            ep_step = torch.arange(
                memory.shape[1], device=memory.device).unsqueeze(0)
        return memory + self.position(ep_step.to(torch.long))

    def _make_tokens(self, state: torch.Tensor) -> torch.Tensor:
        return self.to_tokens(state).reshape(
            state.shape[0], self.mem_compress_length, self.emb_dim)

    def forward(
            self, memory: torch.Tensor, ep_step: Optional[torch.Tensor],
            **kwargs) -> torch.Tensor:
        encoded = self._encode_inputs(memory, ep_step, **kwargs)
        sequence, hidden = self.encoder(encoded)
        self._set_aux_loss(encoded, sequence)
        return self._make_tokens(hidden[-1])

    def _set_aux_loss(
            self, encoded: torch.Tensor, sequence: torch.Tensor) -> None:
        del encoded, sequence
        self._aux_loss = self.to_tokens.weight.sum() * 0.0

    def get_aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def inference(self, new_memory: torch.Tensor, **kwargs) -> torch.Tensor:
        selected = []
        for index in range(new_memory.shape[1]):
            if self.step % self.subsample_ratio == 0:
                selected.append(new_memory[:, index:index + 1])
            self.step += 1
        if selected:
            encoded = torch.cat(selected, dim=1)
            sequence, self.hidden = self.encoder(encoded, self.hidden)
            state = sequence[:, -1]
        elif self.hidden is not None:
            state = self.hidden[-1]
        else:
            state = new_memory.new_zeros(
                new_memory.shape[0], self.emb_dim)
        return self._make_tokens(state)

    def reset(self) -> None:
        self.step = 0
        self.hidden = None


class DSSPStyleMemory(_FixedTokenRecurrentMemory):
    """Recurrent full-history state with a next-feature predictive objective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.future_predictor = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.Mish(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def _set_aux_loss(
            self, encoded: torch.Tensor, sequence: torch.Tensor) -> None:
        if encoded.shape[1] < 2:
            self._aux_loss = sequence.sum() * 0.0
            return
        prediction = self.future_predictor(sequence[:, :-1])
        self._aux_loss = F.mse_loss(prediction, encoded[:, 1:].detach())


class AEMStyleMemory(_FixedTokenRecurrentMemory):
    """Action-conditioned recurrent memory with action-effect prediction.

    ``action_history`` must be temporally aligned with ``memory``.  Requiring
    it explicitly prevents an image-difference proxy from being mislabeled as
    the action-effect mechanism used by AEM.
    """

    def __init__(self, *args, action_dim: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_action_history = True
        self.action_dim = int(action_dim)
        self.action_encoder = nn.Linear(self.action_dim, self.emb_dim)
        self.effect_predictor = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.Mish(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def _encode_inputs(
            self, memory: torch.Tensor, ep_step: Optional[torch.Tensor],
            action_history: Optional[torch.Tensor] = None,
            **kwargs) -> torch.Tensor:
        del kwargs
        if action_history is None:
            raise ValueError(
                "AEMStyleMemory requires aligned action_history; using visual "
                "frame differences would not be an AEM-style experiment")
        if action_history.shape[:2] != memory.shape[:2]:
            raise ValueError(
                "action_history must match memory batch and time dimensions")
        encoded = super()._encode_inputs(memory, ep_step)
        return encoded + self.action_encoder(action_history)

    def _set_aux_loss(
            self, encoded: torch.Tensor, sequence: torch.Tensor) -> None:
        if encoded.shape[1] < 2:
            self._aux_loss = sequence.sum() * 0.0
            return
        predicted_effect = self.effect_predictor(sequence[:, :-1])
        observed_effect = (encoded[:, 1:] - encoded[:, :-1]).detach()
        self._aux_loss = F.mse_loss(predicted_effect, observed_effect)

    def inference(
            self, new_memory: torch.Tensor,
            action_history: Optional[torch.Tensor] = None,
            **kwargs) -> torch.Tensor:
        del kwargs
        if action_history is None:
            raise ValueError(
                "AEMStyleMemory online inference requires action_history")
        if action_history.shape[:2] != new_memory.shape[:2]:
            raise ValueError(
                "online action_history must align with new_memory")
        selected_memory = []
        selected_actions = []
        for index in range(new_memory.shape[1]):
            if self.step % self.subsample_ratio == 0:
                selected_memory.append(new_memory[:, index:index + 1])
                selected_actions.append(action_history[:, index:index + 1])
            self.step += 1
        if selected_memory:
            memory = torch.cat(selected_memory, dim=1)
            actions = torch.cat(selected_actions, dim=1)
            positions = torch.arange(
                memory.shape[1], device=memory.device).unsqueeze(0)
            encoded = self._encode_inputs(
                memory, positions, action_history=actions)
            sequence, self.hidden = self.encoder(encoded, self.hidden)
            state = sequence[:, -1]
        elif self.hidden is not None:
            state = self.hidden[-1]
        else:
            state = new_memory.new_zeros(
                new_memory.shape[0], self.emb_dim)
        return self._make_tokens(state)


class VQStyleMemory(_FixedTokenRecurrentMemory):
    """Discrete latent history state inspired by VQ-Memory."""

    def __init__(
            self, *args, codebook_size: int = 64,
            commitment_cost: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook_size = int(codebook_size)
        self.commitment_cost = float(commitment_cost)
        self.codebook = nn.Embedding(self.codebook_size, self.emb_dim)
        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / self.codebook_size,
            1.0 / self.codebook_size)

    def _quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = encoded.reshape(-1, self.emb_dim)
        codebook = self.codebook.weight
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat @ codebook.t())
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).reshape_as(encoded)
        straight_through = encoded + (quantized - encoded).detach()
        vq_loss = (
            F.mse_loss(quantized, encoded.detach())
            + self.commitment_cost * F.mse_loss(encoded, quantized.detach()))
        return straight_through, vq_loss

    def forward(
            self, memory: torch.Tensor, ep_step: Optional[torch.Tensor],
            **kwargs) -> torch.Tensor:
        encoded = self._encode_inputs(memory, ep_step, **kwargs)
        quantized, self._aux_loss = self._quantize(encoded)
        _, hidden = self.encoder(quantized)
        return self._make_tokens(hidden[-1])

    def inference(self, new_memory: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        selected = []
        for index in range(new_memory.shape[1]):
            if self.step % self.subsample_ratio == 0:
                selected.append(new_memory[:, index:index + 1])
            self.step += 1
        if selected:
            encoded = torch.cat(selected, dim=1)
            quantized, _ = self._quantize(encoded)
            sequence, self.hidden = self.encoder(quantized, self.hidden)
            state = sequence[:, -1]
        elif self.hidden is not None:
            state = self.hidden[-1]
        else:
            state = new_memory.new_zeros(
                new_memory.shape[0], self.emb_dim)
        return self._make_tokens(state)


class CycleProgressMemory(_FixedTokenRecurrentMemory):
    """History state with a training-only normalized progress objective.

    The default target is elapsed episode progress derived from ``ep_step``.
    A task-specific cycle-stage target can be supplied as ``progress_target``
    when the dataset provides it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_head = nn.Linear(self.emb_dim, 1)
        self._progress_target: Optional[torch.Tensor] = None

    def forward(
            self, memory: torch.Tensor, ep_step: Optional[torch.Tensor],
            progress_target: Optional[torch.Tensor] = None,
            **kwargs) -> torch.Tensor:
        if ep_step is None and progress_target is None:
            raise ValueError(
                "CycleProgressMemory needs ep_step or progress_target")
        encoded = self._encode_inputs(memory, ep_step, **kwargs)
        sequence, hidden = self.encoder(encoded)
        if progress_target is None:
            steps = ep_step.to(sequence.dtype)
            denom = steps.amax(dim=1, keepdim=True).clamp_min(1.0)
            progress_target = steps / denom
        target = progress_target.to(sequence.dtype)
        prediction = torch.sigmoid(self.progress_head(sequence).squeeze(-1))
        self._aux_loss = F.mse_loss(prediction, target)
        return self._make_tokens(hidden[-1])
