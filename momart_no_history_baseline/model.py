"""No-History observation encoder and PTP diffusion policy."""

import torch
import torch.nn as nn

from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM
from cleandiffuser.nn_condition.base_nn_condition import BaseNNCondition
from cleandiffuser.nn_condition.multi_image_condition import MultiImageObsConditionMomart
from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP


class TaskConditionedObsEncoder(BaseNNCondition):
    """Add the same learned task embedding to each recent observation token."""

    def __init__(self, encoder, num_tasks, emb_dim=256):
        super().__init__()
        self.encoder = encoder
        self.task_embedding = nn.Embedding(int(num_tasks), int(emb_dim))
        nn.init.normal_(self.task_embedding.weight, std=0.02)

    def forward(self, obs, mask=None):
        obs = dict(obs)
        task_id = obs.pop("_task_id").long()
        if task_id.ndim > 1:
            task_id = task_id.reshape(task_id.shape[0], -1)[:, 0]
        features = self.encoder(obs, mask)
        task = self.task_embedding(task_id).to(dtype=features.dtype)
        while task.ndim < features.ndim:
            task = task.unsqueeze(1)
        return features + task


def build_agent(args, normalizer):
    condition = TaskConditionedObsEncoder(
        MultiImageObsConditionMomart(
            shape_meta=args.shape_meta, emb_dim=256,
            rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop,
            use_group_norm=args.use_group_norm, use_seq=True,
            keep_horizon_dims=True,
        ), args.num_tasks, emb_dim=256,
    ).to(args.device)
    diffusion = LongMemChiTransformerPTP(
        args.action_dim, 256, args.horizon, args.obs_steps, 0,
        d_model=256, nhead=4, num_layers=args.diffusion_num_layers,
        p_drop_emb=args.p_drop_emb, p_drop_attn=args.p_drop_attn,
        timestep_emb_type="positional",
    ).to(args.device)
    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device)
    return LongMemDDPM(
        nn_diffusion=diffusion, nn_condition=condition, nn_memory=None,
        device=args.device, diffusion_steps=args.sample_steps,
        x_max=x_max, x_min=-x_max, args_dict=args,
        action_range=normalizer["action"].range.to(args.device),
        action_min=normalizer["action"].min.to(args.device),
        short_cond_dropout=args.short_cond_dropout,
        long_cond_dropout=args.long_cond_dropout,
        ema_rate=args.ema_rate,
        optim_params={"lr": args.lr, "betas": (args.adam_beta1, args.adam_beta2)},
    )
