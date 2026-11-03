"""Two current object tokens, with explicit absence and PTP visibility masks."""
import math
import torch
from torch import nn
from momart_no_history_baseline.model import build_agent as build_baseline
from cleandiffuser.nn_condition.base_nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP


class ObjectTokenCondition(BaseNNCondition):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("frequencies", 2.0 ** torch.arange(16) * math.pi)
        self.position = nn.Sequential(nn.Linear(128, 256), nn.Mish(), nn.Linear(256, 256))
        self.appearance = nn.Linear(512, 256)

    def forward(self, obs, mask=None):
        obs = dict(obs)
        boxes = obs.pop("_bbox")
        roi = obs.pop("_roi")
        valid = obs.pop("_bbox_valid").reshape(-1, 1)
        task = self.encoder.task_embedding(obs["_task_id"].long().reshape(-1))
        phases = boxes.unsqueeze(-1) * self.frequencies
        position = self.position(torch.cat([phases.sin(), phases.cos()], -1).flatten(1))
        # Use the same main RGB ResNet weights, without random scene cropping.
        appearance = self.appearance(self.encoder.encoder.key_model_map["rgb"](roi))
        position = (position + task) * valid
        appearance = (appearance + task) * valid
        scene = self.encoder(obs, mask)
        extra = torch.cat([position, appearance, valid], -1).to(scene.dtype)
        # Pack channels rather than sequence length: DDPM keeps its original
        # two-observation cache and cannot truncate the object tokens.
        return torch.cat([scene, extra[:, None].expand(-1, scene.shape[1], -1)], -1)


class CurrentObjectTransformer(LongMemChiTransformerPTP):
    def __init__(self, args):
        super().__init__(args.action_dim, 256, args.horizon, args.obs_steps, 2,
                         d_model=256, nhead=4, num_layers=args.diffusion_num_layers,
                         p_drop_emb=args.p_drop_emb, p_drop_attn=args.p_drop_attn,
                         timestep_emb_type="positional")
        self.num_heads = 4
        # Newest-frame object tokens obey the same causal visibility as the
        # newest scene observation; earlier action positions cannot see them.
        with torch.no_grad():
            self.memory_mask[:args.obs_steps-1, 1:3] = float("-inf")

    def forward(self, x, noise, condition=None, memory=None):
        if memory is not None:
            raise ValueError("Current object experiment does not accept history")
        if condition is None:
            condition = x.new_zeros(x.shape[0], self.To, 769)
        if condition.shape[1:] != (self.To, 769):
            raise ValueError("Expected packed scene + position + ROI + valid")
        objects = condition[:, -1, 256:768].reshape(-1, 2, 256)
        valid = condition[:, -1, 768] > 0
        timestep = self.map_noise(noise).unsqueeze(1)
        context = torch.cat([timestep, self.mem_emb(objects),
                             self.obs_emb(condition[..., :256])], 1)
        context = self.encoder(self.drop(context + self.cond_pos_emb))
        cross_mask = self.memory_mask[None].expand(x.shape[0], -1, -1).clone()
        cross_mask[:, :, 1:3] = cross_mask[:, :, 1:3].masked_fill(
            ~valid[:, None, None], float("-inf"))
        cross_mask = cross_mask.repeat_interleave(self.num_heads, 0)
        actions = self.drop(self.act_emb(x) + self.pos_emb)
        decoded = self.decoder(actions, context, tgt_mask=self.mask, memory_mask=cross_mask)
        return self.head(self.ln_f(decoded))


def build_agent(args, normalizer):
    agent = build_baseline(args, normalizer)
    # Rebuild before any updates; retain the baseline's DDPM/EMA settings.
    import copy
    agent.model["condition"] = ObjectTokenCondition(agent.model["condition"]).to(args.device)
    agent.model["diffusion"] = CurrentObjectTransformer(args).to(args.device)
    agent.model_ema = copy.deepcopy(agent.model).requires_grad_(False)
    agent.model.train()
    agent.model_ema.eval()
    agent.optimizer = torch.optim.AdamW(agent.model.parameters(), lr=args.lr,
                                       betas=(args.adam_beta1, args.adam_beta2))
    return agent
