import torch

from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory
from cleandiffuser.utils import at_least_ndim


class AnchorRecentMemory(BaseNNMemory):
    """Keep the first (anchor) and most recent raw history embeddings."""

    def __init__(self, mem_compress_length=2, dropout=0.25):
        super().__init__()
        if int(mem_compress_length) != 2:
            raise ValueError("AnchorRecentMemory currently requires exactly two tokens")
        self.mem_compress_length = 2
        self.dropout = float(dropout)
        self.reset()

    def forward(self, memory: torch.Tensor, mask: torch.Tensor = None):
        batch_size, time_steps, _ = memory.shape
        compressed = memory if time_steps == 1 else memory[:, [0, time_steps - 1], :]
        if mask is not None and tuple(mask.shape) != (batch_size,):
            mask = None
        if self.training:
            keep = (torch.rand((batch_size,), device=memory.device) > self.dropout).float()
        else:
            keep = 1.0 if mask is None else mask
        return compressed * at_least_ndim(keep, compressed.dim())

    def reset(self):
        self.anchor = None
        self.recent = None
        self.seen = 0

    def inference(self, new_memory: torch.Tensor):
        if self.anchor is None:
            self.anchor = new_memory[:, :1, :].detach()
        self.recent = new_memory[:, -1:, :].detach()
        self.seen += new_memory.shape[1]
        if self.seen == 1:
            return self.anchor
        return torch.cat([self.anchor, self.recent], dim=1)
