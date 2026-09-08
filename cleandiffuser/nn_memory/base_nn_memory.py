import torch
import torch.nn as nn

from cleandiffuser.utils import at_least_ndim


def get_mask(mask: torch.Tensor, mask_shape: tuple, dropout: float, train: bool, device: torch.device):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1. if mask is None else mask
    return mask


class BaseNNMemory(nn.Module):
    """VPWEM 记忆压缩器的最小接口。

    记忆模块位于观测编码器和扩散 Transformer 之间。典型输入是已编码的历史
    ``[B, T_history, D]``，输出是固定预算的情景记忆 token
    ``[B, Tm, D]``。固定的 ``Tm`` 让下游策略的计算量不随 episode 变长。

    子类通常还提供两个在线推理接口：

    - ``inference(new_memory)``：把刚离开近期窗口的观测写入有界记忆；
    - ``reset()``：在新 episode 开始时清空所有状态，防止跨回合信息泄漏。

    ``forward`` 用于离线训练，可一次接收整段历史；``mask`` 是可选的样本级
    条件屏蔽。具体子类可以额外接收 ``ep_step`` 或动作历史等上下文。
    """

    def __init__(self, ):
        super().__init__()

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        raise NotImplementedError
