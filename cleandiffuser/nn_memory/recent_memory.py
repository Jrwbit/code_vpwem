import torch

from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory
from cleandiffuser.utils import at_least_ndim


class RecentMemory(BaseNNMemory):
    """Keep the most recent raw embeddings without a learned compressor."""

    def __init__(self, mem_compress_length=2, dropout=0.25):
        super().__init__()
        self.mem_compress_length = int(mem_compress_length)
        self.dropout = float(dropout)
        self.reset()

    def forward(self, memory: torch.Tensor, mask: torch.Tensor = None):
        batch_size = memory.shape[0]
        compressed = memory[:, -self.mem_compress_length :, :]
        if mask is not None and tuple(mask.shape) != (batch_size,):
            mask = None
        if self.training:
            keep = (torch.rand((batch_size,), device=memory.device) > self.dropout).float()
        else:
            keep = 1.0 if mask is None else mask
        return compressed * at_least_ndim(keep, compressed.dim())

    def reset(self):
        self.condition_mem_cache = None

    def inference(self, new_memory: torch.Tensor):
        if self.condition_mem_cache is None:
            self.condition_mem_cache = new_memory.detach()
        else:
            self.condition_mem_cache = torch.cat(
                [self.condition_mem_cache, new_memory.detach()], dim=1
            )
        self.condition_mem_cache = self.condition_mem_cache[
            :, -self.mem_compress_length :, :
        ]
        return self.condition_mem_cache
