"""读取 working memory 与 episodic memory 的动作去噪 Transformer。"""

from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.utils import (FourierEmbedding, PositionalEmbedding,
                                 SinusoidalEmbedding)

from .base_nn_diffusion import BaseNNDiffusion


def init_weight(module):
    ignore_types = (
        nn.Dropout,
        SinusoidalEmbedding,
        FourierEmbedding,
        PositionalEmbedding,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.ModuleList,
        nn.Mish,
        nn.Sequential)

    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.MultiheadAttention):
        weight_names = [
            'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
        for name in weight_names:
            weight = getattr(module, name)
            if weight is not None:
                torch.nn.init.normal_(weight, mean=0.0, std=0.02)

        bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
        for name in bias_names:
            bias = getattr(module, name)
            if bias is not None:
                torch.nn.init.zeros_(bias)

    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)

    elif isinstance(module, LongMemChiTransformerPTP):
        torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        if module.obs_emb is not None:
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)

    elif isinstance(module, ignore_types):
        # no param
        pass
    else:
        raise RuntimeError("Unaccounted module {}".format(module))


class LongMemChiTransformerPTP(BaseNNDiffusion):
    """VPWEM 的策略骨干：对动作序列做 Past Token Prediction (PTP)。

    条件序列的固定排列是 ``[diffusion_step, memory(Tm), observation(To)]``。
    动作 decoder 一方面对带噪动作使用因果 self-attention，另一方面通过
    cross-attention 同时读取长期情景记忆和近期工作记忆。这里不负责产生
    ``memory``；它只消费 ``nn_memory`` 已压缩好的固定长度 token。

    ``Ta`` 是训练动作序列长度，``To`` 是近期观测数，``Tm`` 是长期记忆 token
    数。典型 PTP 配置满足 ``Ta = To + action_steps - 1``。
    """

    def __init__(
            self,
            act_dim: int, obs_dim: int, Ta: int, To: int, Tm: int,
            d_model: int = 256, nhead: int = 4, num_layers: int = 8,
            p_drop_emb: float = 0.0, p_drop_attn: float = 0.3,
            n_cond_layers: int = 0,
            memory_read_mode: str = "context",
            memory_gate_bias: float = 0.0,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(d_model, timestep_emb_type, timestep_emb_params)

        T = Ta
        T_cond = 1 + To
        self.To = To
        self.Tm = Tm
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.memory_read_mode = str(memory_read_mode)
        valid_memory_read_modes = {"context", "gated", "control_indexed"}
        if self.memory_read_mode not in valid_memory_read_modes:
            raise ValueError(
                "memory_read_mode must be one of "
                f"{sorted(valid_memory_read_modes)}, got "
                f"{self.memory_read_mode!r}")

        self.act_emb = nn.Linear(act_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, Ta, d_model))

        self.obs_emb = nn.Linear(obs_dim, d_model)
        self.mem_emb = nn.Linear(obs_dim, d_model)
        if self.memory_read_mode == "gated":
            # GMP-style read gate.  The current observation and diffusion
            # timestep decide how strongly long memory modulates the policy.
            # Initializing the final bias from configuration permits a neutral
            # or conservative starting point without changing the baseline.
            self.memory_gate = nn.Linear(2 * d_model, d_model)
            nn.init.constant_(self.memory_gate.bias, float(memory_gate_bias))
        elif self.memory_read_mode == "control_indexed":
            # Chameleon-style minimal transfer: a query derived from the
            # current control state retrieves a decision-relevant trace from
            # the fixed VPWEM memory tokens.  This is intentionally only the
            # read mechanism, not a claim to reproduce the full paper.
            self.control_read = nn.MultiheadAttention(
                d_model, nhead, dropout=p_drop_attn, batch_first=True)
            self.control_fuse = nn.Linear(2 * d_model, d_model)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 1 + Tm + To, d_model))

        self.drop = nn.Dropout(p_drop_emb)
        self.cond_encoder = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.Mish(), nn.Linear(4 * d_model, d_model))

        # encoder
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, 4 * d_model, p_drop_attn, activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_cond_layers)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.Mish(), nn.Linear(4 * d_model, d_model))

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, 4 * d_model, p_drop_attn, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_layers)

        # 动作 token 的标准因果 mask：位置 t 看不到 t 之后的带噪动作。
        mask = (torch.triu(torch.ones(Ta, Ta)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.mask = nn.Parameter(mask, requires_grad=False)

        # Cross-attention mask 对三类条件区别处理：
        #   diffusion-step token 和全部 Tm 个 memory token：所有动作位置可见；
        #   第 j 个近期 observation token：从动作位置 j 开始才可见。
        # 后一条就是 PTP 的时间对齐约束，避免未来观测泄漏到更早动作位置。
        t, s = torch.meshgrid(
            torch.arange(Ta), torch.arange(To + Tm + 1), indexing='ij')
        mask = (t >= (s - Tm - 1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.memory_mask = nn.Parameter(mask, requires_grad=False)

        # decoder head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, act_dim)

        # constant
        self.T = T
        self.T_cond = T_cond

        self.apply(init_weight)
        if self.memory_read_mode == "gated":
            nn.init.constant_(self.memory_gate.bias, float(memory_gate_bias))

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None):
        """预测 DDPM 的噪声（或按上层配置预测干净动作）。

        Args:
            x: 带噪动作，形状 ``[B, Ta, act_dim]``。
            noise: 每个样本的 diffusion step，形状 ``[B]``。
            condition: 最近 ``To`` 个观测 token，形状 ``[B, To, obs_dim]``。
            memory: 压缩后的 ``Tm`` 个情景记忆 token，形状
                ``[B, Tm, obs_dim]``。

        Returns:
            与 ``x`` 同形状的动作去噪预测 ``[B, Ta, act_dim]``。
        """
        if condition is None:
            condition = torch.zeros((x.shape[0], self.To, self.obs_dim)).to(x.device)  # (b, To, obs_dim)
        if memory is None:
            memory = torch.zeros((x.shape[0], self.Tm, self.obs_dim)).to(x.device)  # (b, Tm, obs_dim)
        t_emb = self.map_noise(noise).unsqueeze(1)  # (b, 1, d_model)

        act_emb = self.act_emb(x)
        obs_emb = self.obs_emb(condition)
        mem_emb = self.mem_emb(memory)

        if self.memory_read_mode == "gated":
            current = obs_emb.mean(dim=1)
            gate = torch.sigmoid(self.memory_gate(
                torch.cat([current, t_emb.squeeze(1)], dim=-1)))
            mem_emb = mem_emb * gate.unsqueeze(1)
            self.last_memory_gate = gate.detach()
        elif self.memory_read_mode == "control_indexed":
            control_query = obs_emb.mean(dim=1, keepdim=True)
            recalled, weights = self.control_read(
                control_query, mem_emb, mem_emb, need_weights=True)
            recalled = recalled.expand(-1, mem_emb.shape[1], -1)
            mem_emb = self.control_fuse(
                torch.cat([mem_emb, recalled], dim=-1))
            self.last_control_read_weights = weights.detach()

        # 条件顺序必须与 __init__ 中 memory_mask 的列语义保持一致。
        cond_emb = torch.cat([t_emb, mem_emb, obs_emb], dim=1)  # (b, 1+Tm+To, d_model)
        cond_pos_emb = self.cond_pos_emb[:, :cond_emb.shape[1], :]
        final_emb = self.drop(cond_emb + cond_pos_emb)
        final_emb = self.encoder(final_emb)  # (b, 1+Tm+To, d_model)

        act_pos_emb = self.pos_emb[:, :act_emb.shape[1], :]
        x = self.drop(act_emb + act_pos_emb)  # (b, Ta, d_model)
        x = self.decoder(tgt=x, memory=final_emb, tgt_mask=self.mask, memory_mask=self.memory_mask)

        x = self.ln_f(x)
        x = self.head(x)

        return x
