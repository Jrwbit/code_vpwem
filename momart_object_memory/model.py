"""Shared networks for all arms; object memory is encoded once per action query."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from cleandiffuser.nn_condition.base_nn_condition import BaseNNCondition, IdentityCondition
from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP
from momart_no_history_baseline.model import build_agent as build_baseline

OBJECT_KEYS = ('features', 'event_valid', 'ages', 'boxes', 'visible', 'valid', 'classes', 'confidence')


class ObjectMemoryEncoder(nn.Module):
    def __init__(self, feature_dim, classes, width=256):
        super().__init__()
        self.event = nn.Sequential(nn.Linear(feature_dim+1, width), nn.Mish())
        self.temporal = nn.GRUCell(width, width)
        self.class_embedding = nn.Embedding(classes, width)
        self.state = nn.Sequential(nn.Linear(6, width), nn.Mish(), nn.Linear(width, width))
        self.width = width

    def forward(self, data):
        features = data['features']
        b, n, h, _ = features.shape
        hidden = features.new_zeros(b*n, self.width)
        for j in range(h):
            event = torch.cat([features[:, :, j], torch.log1p(data['ages'][:, :, j:j+1])], -1)
            candidate = self.temporal(self.event(event).reshape(b*n, -1), hidden)
            valid = data['event_valid'][:, :, j].reshape(b*n, 1).bool()
            hidden = torch.where(valid, candidate, hidden)
        state = torch.cat([data['boxes'], data['visible'].unsqueeze(-1),
                           data['confidence'].unsqueeze(-1)], -1)
        token = (hidden.reshape(b, n, -1) + self.class_embedding(data['classes'].long())
                 + self.state(state))
        return token, data['valid'].bool()


class PackedObjectCondition(BaseNNCondition):
    """Pack tokens/validity alongside recent features for the unchanged DDPM API.

    Repeated payload rows are transport, not additional temporal observations.
    Decoder only reads payload from the latest row and masks past action slots.
    """
    def __init__(self, scene, feature_dim, classes, slots, short_dropout, long_dropout):
        super().__init__()
        self.scene = scene
        self.objects = ObjectMemoryEncoder(feature_dim, classes)
        self.slots = slots
        self.long_dropout = long_dropout
        self.scene_dropout = IdentityCondition(short_dropout)

    def forward(self, obs, mask=None):
        scene = self.scene_dropout(self.scene({k: v for k, v in obs.items() if not k.startswith('_object_')}, mask))
        data = {k: obs['_object_'+k] for k in OBJECT_KEYS}
        tokens, valid = self.objects(data)
        task = self.scene.task_embedding(obs['_task_id'].long().reshape(-1))
        tokens = tokens + task[:, None]
        if self.training and self.long_dropout:
            keep = torch.rand(tokens.shape[0], 1, device=tokens.device) >= self.long_dropout
            valid = valid & keep
        tokens = tokens * valid.unsqueeze(-1)
        payload = torch.cat([tokens.flatten(1), valid.to(tokens.dtype)], -1)
        return torch.cat([scene, payload[:, None].expand(-1, scene.shape[1], -1)], -1)


class ObjectConditionedPTP(LongMemChiTransformerPTP):
    """Original PTP decoder with actual padding and latest-bank availability."""
    def forward(self, x, noise, condition=None, memory=None):
        if condition is None:
            raise ValueError("Packed scene/object condition required")
        b = x.shape[0]
        obs = condition[:, :, :self.obs_dim]
        payload = condition[:, -1, self.obs_dim:]
        expected = self.Tm * (self.obs_dim + 1)
        if payload.shape[-1] != expected:
            raise ValueError("Packed object payload size mismatch")
        tokens = payload[:, :self.Tm*self.obs_dim].reshape(b, self.Tm, self.obs_dim)
        valid = payload[:, self.Tm*self.obs_dim:] > 0.5
        t_emb = self.map_noise(noise).unsqueeze(1)
        cond = torch.cat([t_emb, self.mem_emb(tokens), self.obs_emb(obs)], 1)
        encoded = self.encoder(self.drop(cond + self.cond_pos_emb))
        padding = torch.cat([torch.zeros(b, 1, device=x.device, dtype=torch.bool), ~valid,
                             torch.zeros(b, self.To, device=x.device, dtype=torch.bool)], 1)
        temporal = torch.isneginf(self.memory_mask).clone()
        # Latest bank includes current t: no access from positions preceding t.
        temporal[:self.To-1, 1:1+self.Tm] = True
        action = self.drop(self.act_emb(x) + self.pos_emb[:, :x.shape[1]])
        decoded = self.decoder(action, encoded, tgt_mask=torch.isneginf(self.mask),
                               memory_mask=temporal, memory_key_padding_mask=padding)
        return self.head(self.ln_f(decoded))


def build_agent(args, normalizer, record_initialization=False):
    """Initialize common scene/action parameters identically across all arms."""
    agent = build_baseline(args, normalizer)
    old = agent.model['diffusion']
    memory = args.object_memory
    new = ObjectConditionedPTP(args.action_dim, 256, args.horizon, args.obs_steps,
                              memory.slots, d_model=256, nhead=4,
                              num_layers=args.diffusion_num_layers,
                              p_drop_emb=args.p_drop_emb, p_drop_attn=args.p_drop_attn,
                              n_cond_layers=0, timestep_emb_type='positional').to(args.device)
    # Explicit parameter mapping: positional rows change when Tm grows.
    state = dict(old.state_dict())
    state.pop('memory_mask')
    positions = state.pop('cond_pos_emb')
    result = new.load_state_dict(state, strict=False)
    assert set(result.missing_keys) == {'memory_mask', 'cond_pos_emb'}
    assert not result.unexpected_keys
    with torch.no_grad():
        new.cond_pos_emb[:, :1].copy_(positions[:, :1])
        new.cond_pos_emb[:, 1+memory.slots:].copy_(positions[:, 1:])
    scene = agent.model['condition']
    packed = PackedObjectCondition(scene, memory.feature_dim, len(memory.classes),
                                   memory.slots, args.short_cond_dropout, args.long_cond_dropout).to(args.device)
    agent.model['diffusion'] = new
    agent.model['condition'] = packed
    agent.nn_condition_drop = IdentityCondition(0.)
    agent.model_ema = copy.deepcopy(agent.model).requires_grad_(False)
    agent.model_ema.eval()
    agent.optimizer = torch.optim.AdamW(agent.model.parameters(), lr=args.lr,
                                       betas=(args.adam_beta1, args.adam_beta2))
    if record_initialization:
        digest = hashlib.sha256()
        for key, value in sorted(agent.model.state_dict().items()):
            digest.update(key.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        record = dict(initial_model_sha256=digest.hexdigest(),
                      trainable_parameters=sum(p.numel() for p in agent.model.parameters() if p.requires_grad))
        (Path(args.work_dir)/'initialization.json').write_text(json.dumps(record,indent=2)+'\n')
    return agent
