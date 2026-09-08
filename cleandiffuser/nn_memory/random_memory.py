import torch
import torch.nn as nn

from cleandiffuser.utils import at_least_ndim
from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory


def get_mask(mask: torch.Tensor, mask_shape: tuple, dropout: float, train: bool, device: torch.device):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1. if mask is None else mask
    return mask


class RandomMemory(BaseNNMemory):
    """
    In decision-making tasks, generating condition selections can be very diverse,
    including cumulative rewards, languages, images, demonstrations, and so on.
    It can even be a combination of these conditions. Therefore, we aim for
    nn_condition to handle diverse condition selections flexibly and
    ultimately output a tensor of shape (b, *cond_out_shape).

    Input:
        - condition: Any
        - mask : Optional, (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_out_shape)
    """

    def __init__(self, mem_compress_length=4, dropout=0.25, retention_seed=None):
        super().__init__()
        self.mem_compress_length = int(mem_compress_length)
        self.dropout = float(dropout)
        self.retention_seed = int(
            torch.initial_seed() if retention_seed is None else retention_seed)
        # Keep retention randomness off the global CPU/CUDA streams so it cannot
        # silently change diffusion action noise in paired evaluations.
        self.retention_generator = torch.Generator(device="cpu")
        self.retention_generator.manual_seed(self.retention_seed)
        self.reset()

    def forward(self, memory: torch.Tensor, mask: torch.Tensor = None):
        B, T, D = memory.shape
        # Draw independently for every trajectory instead of sharing one set of
        # time indices across the entire training batch. A (B, T) second argument
        # is the ep_step tensor used by LongMemDDPM; left padding has ep_step=0,
        # so the valid history is the final count_nonzero+1 entries.
        ep_steps = mask if mask is not None and tuple(mask.shape) == (B, T) else None
        rows = []
        for batch_idx in range(B):
            valid_length = (
                int(torch.count_nonzero(ep_steps[batch_idx]).item()) + 1
                if ep_steps is not None else T)
            valid_length = min(max(valid_length, 1), T)
            valid_start = T - valid_length
            if valid_length >= self.mem_compress_length:
                row = torch.randperm(
                    valid_length, generator=self.retention_generator
                )[:self.mem_compress_length] + valid_start
            else:
                # Fixed token count for very short histories; repeat the anchor.
                row = torch.arange(valid_start, T)
                row = torch.cat([
                    row,
                    row[:1].repeat(self.mem_compress_length - valid_length),
                ])
            rows.append(row)
        random_indices = torch.stack(rows)
        sorted_random_indices = torch.sort(random_indices, dim=1).values.to(memory.device)
        cp_memory = torch.gather(
            memory, 1, sorted_random_indices.unsqueeze(-1).expand(-1, -1, D))
        if mask is not None and tuple(mask.shape) != (B,):
            mask = None
        mask = at_least_ndim(get_mask(
            mask, (cp_memory.shape[0],), self.dropout, self.training, cp_memory.device), cp_memory.dim())
        return cp_memory * mask

    def reset(self):
        self.condition_mem_cache_list = []
        self.condition_mem_cache_step_list = []
        self.retention_trace = []
        self.step = 0
    
    def inference(self, new_memory: torch.Tensor):
        # reservoir_sampling
        B, T, D = new_memory.shape
        # new_memory (B, T, D)
        # new_memory_list = [new_memory[:, t, :] for t in range(T)]
        # 处理前 k 个元素
        for t_idx in range(T):
            i = len(self.condition_mem_cache_list)
            if i < self.mem_compress_length:
                current_step = self.step * torch.ones(B, device=new_memory.device, dtype=torch.int)
                self.condition_mem_cache_list.append(new_memory[:, t_idx, :])
                self.condition_mem_cache_step_list.append(current_step)
            else:
                # Algorithm R for zero-indexed stream position ``self.step``:
                # draw uniformly from [0, self.step], including the new item.
                j = torch.randint(
                    0, self.step + 1, (B,), generator=self.retention_generator)
                # 创建掩码，检查哪些 j 值小于 mem_compress_length
                mask = j < self.mem_compress_length

                # 使用索引更新 condition_mem_cache_list 和 condition_mem_cache_step_list
                # 选取满足条件的索引
                selected_indices = j[mask]
                selected_b = torch.arange(B)[mask].to(new_memory.device)

                # 更新缓存
                if len(selected_indices) > 0:  # 确保有满足条件的索引
                    for idx in range(len(selected_indices)):
                        slot = int(selected_indices[idx])
                        self.condition_mem_cache_list[slot][selected_b[idx]] = new_memory[selected_b[idx], t_idx, :]
                        self.condition_mem_cache_step_list[slot][selected_b[idx]] = self.step
            self.step += 1
            current_steps = torch.stack(
                self.condition_mem_cache_step_list, dim=1).detach().cpu()
            self.retention_trace.append(
                torch.sort(current_steps, dim=1).values.clone())
        # 将缓存转换为张量
        memory_tensor = torch.stack(self.condition_mem_cache_list, dim=1)
        memory_step_tensor = torch.stack(self.condition_mem_cache_step_list, dim=1)
        # 根据 step 排序
        sorted_indices = torch.argsort(memory_step_tensor, dim=1)
        batch_indices = torch.arange(B, device=memory_tensor.device).unsqueeze(1)
        memory_sorted_by_step = memory_tensor[batch_indices, sorted_indices]
        # for debug only
        memory_step_tensor_sorted_by_step = memory_step_tensor[batch_indices, sorted_indices]
        return memory_sorted_by_step

    def get_retention_trace(self):
        """Return per-update retained stream indices for audit logging."""
        return [steps.clone() for steps in self.retention_trace]
