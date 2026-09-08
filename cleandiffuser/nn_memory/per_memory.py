from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory
from cleandiffuser.nn_memory.qformer_memory import QFormer
from cleandiffuser.utils.utils import SinusoidalEmbedding


class PERMemory(BaseNNMemory):
    """Predictive event-gated, evidence-belief memory with a fixed token budget.

    The non-parametric event bank is bounded by ``bank_size``. A small number of
    raw evidence anchors are concatenated with learned Q-Former belief tokens.
    """

    def __init__(
            self,
            mem_compress_length: int = 4,
            evidence_length: int = 1,
            emb_dim: int = 256,
            num_layers: int = 1,
            num_heads: int = 8,
            bank_size: int = 8,
            event_score_mode: str = "delta",
            event_threshold: float = 0.05,
            max_write_interval: int = 20,
            merge_similarity: float = 0.98,
            score_decay: float = 0.995,
            pos_emb_scale: float = 0.1,
            subsample_ratio: int = 1,
    ):
        super().__init__()
        if not 0 <= evidence_length < mem_compress_length:
            raise ValueError(
                "evidence_length must be in [0, mem_compress_length)")
        if bank_size < max(1, evidence_length):
            raise ValueError("bank_size must cover all evidence anchors")
        if event_score_mode not in {"delta", "predictive", "hybrid"}:
            raise ValueError(f"Unknown event score mode: {event_score_mode}")

        self.mem_compress_length = mem_compress_length
        self.evidence_length = evidence_length
        self.belief_length = mem_compress_length - evidence_length
        self.emb_dim = emb_dim
        self.bank_size = bank_size
        self.event_score_mode = event_score_mode
        self.event_threshold = event_threshold
        self.max_write_interval = max_write_interval
        self.merge_similarity = merge_similarity
        self.score_decay = score_decay
        self.pos_emb_scale = pos_emb_scale
        self.subsample_ratio = max(1, int(subsample_ratio))

        self.mem_pos_emb = SinusoidalEmbedding(emb_dim)
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Mish(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.qformer = QFormer(
            input_dim=emb_dim,
            d_model=emb_dim,
            n_queries=self.belief_length,
            n_layers=num_layers,
            nhead=num_heads,
            cache_compress="fifo",
            cache_max_length=bank_size,
        )
        self.aux_loss = torch.tensor(0.0)
        self.reset()

    def _event_scores(self, memory: torch.Tensor):
        batch, steps, _ = memory.shape
        if steps == 1:
            self.aux_loss = memory.new_zeros(())
            return torch.full(
                (batch, 1), float("inf"),
                device=memory.device, dtype=memory.dtype)

        delta = 1.0 - F.cosine_similarity(
            memory[:, 1:], memory[:, :-1], dim=-1)
        prediction = self.predictor(memory[:, :-1])
        predictive = (prediction - memory[:, 1:]).pow(2).mean(dim=-1)
        self.aux_loss = predictive.mean()

        if self.event_score_mode == "delta":
            score = delta
        elif self.event_score_mode == "predictive":
            score = predictive
        else:
            # Normalize per trajectory so the two signals have comparable scale.
            delta_n = delta / delta.mean(dim=1, keepdim=True).clamp_min(1e-6)
            pred_n = predictive / predictive.mean(
                dim=1, keepdim=True).clamp_min(1e-6)
            score = 0.5 * (delta_n + pred_n)

        first = torch.full(
            (batch, 1), float("inf"),
            device=memory.device, dtype=memory.dtype)
        return torch.cat([first, score], dim=1)

    def _select_bank(self, memory: torch.Tensor, ep_step: torch.Tensor):
        scores = self._event_scores(memory)
        select_count = min(self.bank_size, memory.shape[1])
        # Always retain the first observation and select the remaining events by
        # utility. Sorting by episode step restores temporal order for Q-Former.
        if select_count == 1:
            indices = torch.zeros(
                memory.shape[0], 1, device=memory.device, dtype=torch.long)
        else:
            event_indices = scores[:, 1:].topk(
                k=select_count - 1, dim=1).indices + 1
            indices = torch.cat([
                torch.zeros(
                    memory.shape[0], 1,
                    device=memory.device, dtype=torch.long),
                event_indices,
            ], dim=1)
        selected_steps = ep_step.gather(1, indices)
        order = selected_steps.argsort(dim=1)
        indices = indices.gather(1, order)
        selected_steps = selected_steps.gather(1, order)
        selected = memory.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
        selected_scores = scores.gather(1, indices)
        return selected, selected_steps, selected_scores

    def _compose_tokens(
            self, bank: torch.Tensor, bank_steps: torch.Tensor,
            bank_scores: Optional[torch.Tensor] = None,
            recurrent: bool = False):
        positioned = bank + self.pos_emb_scale * self.mem_pos_emb(bank_steps)
        if recurrent:
            belief = None
            for time_idx in range(positioned.shape[1]):
                belief = self.qformer(positioned[:, :time_idx + 1])
        else:
            belief = self.qformer(positioned)
        if self.evidence_length == 0:
            return belief

        if bank_scores is None:
            evidence_indices = torch.arange(
                min(self.evidence_length, bank.shape[1]),
                device=bank.device).unsqueeze(0).expand(bank.shape[0], -1)
        else:
            scores = bank_scores.clone()
            scores[:, 0] = torch.finfo(scores.dtype).max
            evidence_indices = scores.topk(
                k=min(self.evidence_length, bank.shape[1]), dim=1).indices
        evidence = positioned.gather(
            1, evidence_indices.unsqueeze(-1).expand(-1, -1, self.emb_dim))
        if evidence.shape[1] < self.evidence_length:
            evidence = F.pad(
                evidence,
                (0, 0, 0, self.evidence_length - evidence.shape[1]))
        return torch.cat([evidence, belief], dim=1)

    def forward(
            self, memory: torch.Tensor, ep_step: torch.Tensor,
            mask: torch.Tensor = None):
        self.qformer.layers[0].reset()
        for layer in self.qformer.layers[1:]:
            layer.reset()
        bank, bank_steps, scores = self._select_bank(memory, ep_step)
        return self._compose_tokens(
            bank, bank_steps, scores, recurrent=True)

    def get_aux_loss(self):
        return self.aux_loss

    def reset(self):
        self.step = 0
        self.bank = None
        self.bank_steps = None
        self.bank_scores = None
        self.bank_counts = None
        self.last_observation = None
        self.last_write_step = None
        if hasattr(self, "cp_tokens"):
            del self.cp_tokens
        if hasattr(self, "qformer"):
            for layer in self.qformer.layers:
                layer.reset()

    def _initialize_online_state(self, batch: int, device, dtype):
        self.bank = torch.zeros(
            batch, self.bank_size, self.emb_dim, device=device, dtype=dtype)
        self.bank_steps = torch.zeros(
            batch, self.bank_size, device=device, dtype=torch.long)
        self.bank_scores = torch.full(
            (batch, self.bank_size), -torch.inf,
            device=device, dtype=dtype)
        self.bank_counts = torch.zeros(batch, device=device, dtype=torch.long)
        self.last_write_step = torch.zeros(
            batch, device=device, dtype=torch.long)

    @torch.no_grad()
    def _write_online(self, observation: torch.Tensor):
        batch = observation.shape[0]
        if self.bank is None:
            self._initialize_online_state(
                batch, observation.device, observation.dtype)
        if self.bank.shape[0] != batch:
            raise ValueError("Online batch size changed without reset()")

        if self.last_observation is None:
            scores = torch.full(
                (batch,), float("inf"),
                device=observation.device, dtype=observation.dtype)
        else:
            delta = 1.0 - F.cosine_similarity(
                observation, self.last_observation, dim=-1)
            predictive = (
                self.predictor(self.last_observation) - observation
            ).pow(2).mean(dim=-1)
            if self.event_score_mode == "delta":
                scores = delta
            elif self.event_score_mode == "predictive":
                scores = predictive
            else:
                scores = delta + predictive

        self.bank_scores.mul_(self.score_decay)
        due = (self.step - self.last_write_step) >= self.max_write_interval
        should_write = (
            (scores >= self.event_threshold)
            | due
            | (self.bank_counts == 0)
        )

        wrote_any = False
        for batch_idx in range(batch):
            if not bool(should_write[batch_idx]):
                continue
            count = int(self.bank_counts[batch_idx])
            if count > 0:
                similarities = F.cosine_similarity(
                    self.bank[batch_idx, :count],
                    observation[batch_idx].unsqueeze(0), dim=-1)
                nearest = int(similarities.argmax())
            else:
                similarities = None
                nearest = -1

            if (
                    similarities is not None
                    and float(similarities[nearest]) >= self.merge_similarity):
                if nearest == 0:
                    # Slot zero is immutable episode evidence. A redundant
                    # observation refreshes its utility but not its content.
                    self.bank_scores[batch_idx, 0] = torch.maximum(
                        self.bank_scores[batch_idx, 0], scores[batch_idx])
                    self.last_write_step[batch_idx] = self.step
                    continue
                slot = nearest
            elif count < self.bank_size:
                slot = count
                self.bank_counts[batch_idx] += 1
            else:
                # Preserve slot zero as an episode anchor.
                replace_scores = self.bank_scores[batch_idx].clone()
                replace_scores[0] = torch.inf
                slot = int(replace_scores.argmin())

            self.bank[batch_idx, slot] = observation[batch_idx]
            self.bank_steps[batch_idx, slot] = self.step
            self.bank_scores[batch_idx, slot] = scores[batch_idx]
            self.last_write_step[batch_idx] = self.step
            wrote_any = True

        self.last_observation = observation
        self.step += 1
        return wrote_any

    def inference(self, new_memory: torch.Tensor):
        wrote_any = False
        for time_idx in range(new_memory.shape[1]):
            if self.step % self.subsample_ratio == 0:
                wrote_any = (
                    self._write_online(new_memory[:, time_idx]) or wrote_any)
            else:
                self.step += 1

        if not wrote_any and hasattr(self, "cp_tokens"):
            return self.cp_tokens

        max_count = max(1, int(self.bank_counts.max()))
        bank = self.bank[:, :max_count]
        steps = self.bank_steps[:, :max_count]
        scores = self.bank_scores[:, :max_count]
        # Invalid slots are copies of the trajectory anchor and receive low
        # utility, avoiding all-masked attention for short trajectories.
        valid = torch.arange(
            max_count, device=bank.device).unsqueeze(0) < self.bank_counts[:, None]
        bank = torch.where(valid.unsqueeze(-1), bank, bank[:, :1])
        steps = torch.where(valid, steps, steps[:, :1])
        scores = torch.where(
            valid, scores, torch.full_like(scores, -torch.inf))
        # Replacement reuses arbitrary physical slots; restore chronological
        # order so online beliefs follow the same distribution as training.
        sort_steps = torch.where(
            valid, steps, torch.full_like(steps, torch.iinfo(steps.dtype).max))
        order = sort_steps.argsort(dim=1)
        bank = bank.gather(
            1, order.unsqueeze(-1).expand(-1, -1, self.emb_dim))
        steps = steps.gather(1, order)
        scores = scores.gather(1, order)
        # Rebuild beliefs from the bounded event bank. Carrying Q-Former caches
        # across rebuilt bank snapshots would count the same event repeatedly.
        for layer in self.qformer.layers:
            layer.reset()
        self.cp_tokens = self._compose_tokens(
            bank, steps, scores, recurrent=True)
        return self.cp_tokens
