import torch
from types import SimpleNamespace

from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM
from cleandiffuser.nn_memory.per_memory import PERMemory
from cleandiffuser.nn_diffusion.longmem_chitransformerptp import (
    LongMemChiTransformerPTP,
)


def make_memory(**kwargs):
    return PERMemory(
        mem_compress_length=4,
        evidence_length=1,
        emb_dim=8,
        num_layers=1,
        num_heads=2,
        bank_size=4,
        event_threshold=0.1,
        max_write_interval=3,
        **kwargs,
    )


def test_training_forward_has_fixed_evidence_belief_budget():
    module = make_memory()
    history = torch.randn(3, 10, 8, requires_grad=True)
    steps = torch.arange(10).repeat(3, 1)
    output = module(history, steps)
    assert output.shape == (3, 4, 8)
    (output.square().mean() + module.get_aux_loss()).backward()
    assert module.predictor[0].weight.grad is not None


def test_event_bank_keeps_episode_anchor_and_is_bounded():
    module = make_memory()
    observations = torch.randn(2, 20, 8)
    module.inference(observations)
    assert torch.equal(module.bank[:, 0], observations[:, 0])
    assert int(module.bank_counts.max()) <= module.bank_size
    assert module.inference(observations[:, :1]).shape == (2, 4, 8)


def test_static_history_writes_only_on_max_interval():
    module = make_memory(event_score_mode="delta")
    observation = torch.ones(1, 10, 8)
    module.inference(observation)
    # Periodic refreshes do not consume the bank or overwrite episode evidence.
    assert int(module.bank_counts.item()) == 1
    assert int(module.bank_steps[0, 0]) == 0
    assert torch.equal(module.bank[0, 0], observation[0, 0])


def test_reset_clears_online_episode_state():
    module = make_memory()
    module.inference(torch.randn(2, 3, 8))
    module.reset()
    assert module.bank is None
    assert module.last_observation is None
    assert module.step == 0


def test_online_candidates_respect_subsample_ratio():
    module = PERMemory(
        mem_compress_length=4,
        evidence_length=1,
        emb_dim=8,
        num_layers=1,
        num_heads=2,
        bank_size=4,
        event_threshold=-1.0,
        max_write_interval=100,
        subsample_ratio=3,
    )
    observations = torch.randn(1, 10, 8)
    module.inference(observations)
    assert module.step == 10
    assert int(module.bank_counts.item()) == 4
    assert torch.equal(module.last_observation, observations[:, 9])


def test_longmem_loss_combines_predictive_and_counterfactual_objectives():
    memory = make_memory()
    diffusion = LongMemChiTransformerPTP(
        act_dim=2,
        obs_dim=8,
        Ta=3,
        To=1,
        Tm=4,
        d_model=8,
        nhead=2,
        num_layers=1,
        p_drop_attn=0.0,
    )
    args = SimpleNamespace(
        inference_action_steps=1,
        pred_reg_coeff=0.0,
        memory_aux_coeff=0.5,
        counterfactual_memory_coeff=0.5,
        counterfactual_memory_margin=0.1,
        counterfactual_memory_probability=1.0,
    )
    agent = LongMemDDPM(
        nn_diffusion=diffusion,
        nn_memory=memory,
        diffusion_steps=4,
        x_max=torch.ones(1, 3, 2),
        x_min=-torch.ones(1, 3, 2),
        args_dict=args,
        device="cpu",
    )
    total, _, _, action = agent.loss(
        x0=torch.randn(3, 3, 2),
        condition=torch.randn(3, 1, 8),
        memory=torch.randn(3, 7, 8),
        ep_step=torch.arange(7).repeat(3, 1),
    )
    assert torch.isfinite(total)
    assert total >= action
    assert agent._last_memory_losses["memory_aux_loss"] > 0
    assert "counterfactual_memory_loss" in agent._last_memory_losses
