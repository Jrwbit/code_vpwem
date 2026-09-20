"""PTP diffusion transformer that directly reads spatial history tokens."""

from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.utils import FourierEmbedding, PositionalEmbedding, SinusoidalEmbedding
from .base_nn_diffusion import BaseNNDiffusion


def _init_weight(module):
    ignored = (nn.Dropout, SinusoidalEmbedding, FourierEmbedding,
               PositionalEmbedding, nn.TransformerEncoderLayer,
               nn.TransformerDecoderLayer, nn.TransformerEncoder,
               nn.TransformerDecoder, nn.ModuleList, nn.Mish, nn.Sequential)
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        for name in ('in_proj_weight', 'q_proj_weight', 'k_proj_weight',
                     'v_proj_weight'):
            weight = getattr(module, name)
            if weight is not None:
                nn.init.normal_(weight, mean=0.0, std=0.02)
        for name in ('in_proj_bias', 'bias_k', 'bias_v'):
            bias = getattr(module, name)
            if bias is not None:
                nn.init.zeros_(bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, DirectHistoryChiTransformerPTP):
        nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
    elif not isinstance(module, ignored):
        raise RuntimeError(f"Unaccounted module {module}")


class DirectHistoryChiTransformerPTP(BaseNNDiffusion):
    def __init__(self, act_dim, obs_dim, Ta, To, history_tokens,
                 d_model=256, nhead=4, num_layers=8,
                 p_drop_emb=0.0, p_drop_attn=0.3,
                 timestep_emb_type="positional", timestep_emb_params=None):
        super().__init__(d_model, timestep_emb_type, timestep_emb_params)
        self.T, self.To, self.Tm = Ta, To, int(history_tokens)
        self.obs_dim = obs_dim
        self.act_emb = nn.Linear(act_dim, d_model)
        self.obs_emb = nn.Linear(obs_dim, d_model)
        self.history_emb = nn.Linear(obs_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, Ta, d_model))
        self.cond_pos_emb = nn.Parameter(
            torch.zeros(1, 1 + To + self.Tm, d_model))
        self.drop = nn.Dropout(p_drop_emb)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.Mish(),
            nn.Linear(4 * d_model, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, 4 * d_model, p_drop_attn,
            activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        causal = torch.triu(torch.ones(Ta, Ta), diagonal=1).bool()
        self.register_buffer(
            'mask', causal.float().masked_fill(causal, float('-inf')))
        cross = torch.zeros(Ta, 1 + To + self.Tm)
        for obs_index in range(To):
            cross[:obs_index, 1 + obs_index] = float('-inf')
        self.register_buffer('memory_mask', cross)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, act_dim)
        self.apply(_init_weight)

    def forward(self, x, noise, condition: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None):
        batch = x.shape[0]
        if condition is None:
            condition = x.new_zeros(batch, self.To, self.obs_dim)
        if memory is None:
            memory = x.new_zeros(batch, self.Tm, self.obs_dim)
        if memory.shape[1] != self.Tm:
            raise ValueError(
                f"expected {self.Tm} history tokens, got {memory.shape[1]}")
        context = torch.cat([
            self.map_noise(noise).unsqueeze(1),
            self.obs_emb(condition), self.history_emb(memory)], dim=1)
        context = self.encoder(
            self.drop(context + self.cond_pos_emb[:, :context.shape[1]]))
        action = self.act_emb(x)
        action = self.drop(action + self.pos_emb[:, :action.shape[1]])
        action = self.decoder(
            action, context, tgt_mask=self.mask,
            memory_mask=self.memory_mask)
        return self.head(self.ln_f(action))
