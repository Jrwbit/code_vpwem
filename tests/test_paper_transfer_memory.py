from types import SimpleNamespace

import pytest
import torch

from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM
from cleandiffuser.dataset.dataset_utils import dynamic_batch_collate_fn
from cleandiffuser.nn_diffusion.longmem_chitransformerptp import (
    LongMemChiTransformerPTP,
)
from cleandiffuser.nn_memory.paper_transfer_memory import (
    AEMStyleMemory,
    CycleProgressMemory,
    DSSPStyleMemory,
    VQStyleMemory,
)


@pytest.mark.parametrize(
    "memory_cls,kwargs",
    [
        (DSSPStyleMemory, {}),
        (VQStyleMemory, {"codebook_size": 8}),
        (CycleProgressMemory, {}),
    ],
)
def test_transfer_memories_have_fixed_budget_and_aux_loss(memory_cls, kwargs):
    module = memory_cls(
        mem_compress_length=3, emb_dim=8, subsample_ratio=2, **kwargs)
    history = torch.randn(4, 7, 8, requires_grad=True)
    steps = torch.arange(7).repeat(4, 1)
    output = module(history, steps)
    assert output.shape == (4, 3, 8)
    loss = output.square().mean() + module.get_aux_loss()
    loss.backward()
    assert torch.isfinite(loss)

    module.reset()
    online = module.inference(history.detach()[:, :4])
    assert online.shape == (4, 3, 8)
    assert module.step == 4


def test_aem_requires_aligned_action_history():
    module = AEMStyleMemory(
        mem_compress_length=2, emb_dim=8, action_dim=3)
    history = torch.randn(2, 5, 8)
    steps = torch.arange(5).repeat(2, 1)
    with pytest.raises(ValueError, match="action_history"):
        module(history, steps)
    output = module(
        history, steps, action_history=torch.randn(2, 5, 3))
    assert output.shape == (2, 2, 8)
    assert module.get_aux_loss() > 0


def test_dynamic_collate_aligns_action_and_observation_memory():
    samples = []
    for length in (2, 4):
        samples.append({
            "obs": {
                "rgb_emb_mem": torch.randn(length, 8),
                "ep_step": torch.arange(length),
                "actual_num_frames": length,
            },
            "action": torch.randn(3, 2),
            "memory_action": torch.randn(length, 2),
        })
    batch = dynamic_batch_collate_fn(samples)
    assert batch["obs"]["rgb_emb_mem"].shape == (2, 4, 8)
    assert batch["memory_action"].shape == (2, 4, 2)


def test_aem_action_context_reaches_longmem_training_loss():
    memory = AEMStyleMemory(
        mem_compress_length=2, emb_dim=8, action_dim=3)
    diffusion = LongMemChiTransformerPTP(
        act_dim=3, obs_dim=8, Ta=4, To=2, Tm=2,
        d_model=8, nhead=2, num_layers=1, p_drop_attn=0.0)
    args = SimpleNamespace(
        inference_action_steps=1,
        pred_reg_coeff=0.0,
        memory_aux_coeff=0.1,
        counterfactual_memory_coeff=0.0,
        counterfactual_memory_margin=0.1,
        counterfactual_memory_probability=1.0,
        counterfactual_negative_mode="none",
        counterfactual_objective="ranking",
    )
    agent = LongMemDDPM(
        nn_diffusion=diffusion, nn_memory=memory, diffusion_steps=4,
        x_max=torch.ones(1, 4, 3), x_min=-torch.ones(1, 4, 3),
        args_dict=args, device="cpu")
    total, _, compressed, _ = agent.loss(
        x0=torch.randn(3, 4, 3),
        condition=torch.randn(3, 2, 8),
        memory=torch.randn(3, 6, 8),
        ep_step=torch.arange(6).repeat(3, 1),
        memory_context={"action_history": torch.randn(3, 6, 3)},
    )
    assert compressed.shape == (3, 2, 8)
    assert torch.isfinite(total)


def test_longmem_sample_initializes_action_cache_from_action_dim():
    memory = DSSPStyleMemory(mem_compress_length=2, emb_dim=8)
    diffusion = LongMemChiTransformerPTP(
        act_dim=3, obs_dim=8, Ta=4, To=2, Tm=2,
        d_model=8, nhead=2, num_layers=1, p_drop_attn=0.0)
    args = SimpleNamespace(
        inference_action_steps=1,
        pred_reg_coeff=0.0,
        memory_aux_coeff=0.0,
        counterfactual_memory_coeff=0.0,
        counterfactual_memory_margin=0.1,
        counterfactual_memory_probability=1.0,
        counterfactual_negative_mode="none",
        counterfactual_objective="ranking",
    )
    agent = LongMemDDPM(
        nn_diffusion=diffusion, nn_memory=memory, diffusion_steps=2,
        x_max=torch.ones(1, 4, 3), x_min=-torch.ones(1, 4, 3),
        args_dict=args, device="cpu")
    sample, _ = agent.sample(
        prior=torch.zeros(2, 4, 3), n_samples=2, sample_steps=2,
        condition_cfg=torch.randn(2, 4, 8))
    assert sample.shape == (2, 4, 3)
    assert agent.action_dim == 3
    assert len(agent.action_cache_list) == 2


@pytest.mark.parametrize("read_mode", ["context", "gated", "control_indexed"])
def test_memory_read_modes_support_diffusion_backward(read_mode):
    memory = DSSPStyleMemory(mem_compress_length=2, emb_dim=8)
    diffusion = LongMemChiTransformerPTP(
        act_dim=3,
        obs_dim=8,
        Ta=4,
        To=2,
        Tm=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        p_drop_attn=0.0,
        memory_read_mode=read_mode,
    )
    args = SimpleNamespace(
        inference_action_steps=1,
        pred_reg_coeff=0.0,
        memory_aux_coeff=0.1,
        counterfactual_memory_coeff=0.0,
        counterfactual_memory_margin=0.1,
        counterfactual_memory_probability=1.0,
        counterfactual_negative_mode="none",
        counterfactual_objective="ranking",
    )
    agent = LongMemDDPM(
        nn_diffusion=diffusion,
        nn_memory=memory,
        diffusion_steps=4,
        x_max=torch.ones(1, 4, 3),
        x_min=-torch.ones(1, 4, 3),
        args_dict=args,
        device="cpu",
    )
    total, _, compressed, action = agent.loss(
        x0=torch.randn(3, 4, 3),
        condition=torch.randn(3, 2, 8),
        memory=torch.randn(3, 6, 8),
        ep_step=torch.arange(6).repeat(3, 1),
    )
    total.backward()
    assert compressed.shape == (3, 2, 8)
    assert torch.isfinite(total)
    assert total >= action
    if read_mode == "gated":
        assert diffusion.last_memory_gate.shape == (3, 8)
    if read_mode == "control_indexed":
        assert diffusion.last_control_read_weights.shape == (3, 1, 2)


def test_invalid_memory_read_mode_fails_fast():
    with pytest.raises(ValueError, match="memory_read_mode"):
        LongMemChiTransformerPTP(
            act_dim=2, obs_dim=8, Ta=3, To=1, Tm=2,
            d_model=8, nhead=2, num_layers=1,
            memory_read_mode="not-a-mode")
