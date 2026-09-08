import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from cleandiffuser.dataset.ncm_pairing import (
    NCMPairedBatchSampler,
    NCMHybridBatchSampler,
    load_ncm_manifest,
    partner_positions,
)
from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM


def write_manifest(path):
    np.savez_compressed(
        path,
        schema_version=np.asarray(1),
        metadata_json=np.asarray(json.dumps({"test": True})),
        batches=np.asarray([[0, 1, 2, 3]]),
        branch=np.asarray([0, 1, 1, 2]),
        episode_id=np.asarray([10, 11, 12, 13]),
        decision_step=np.asarray([6, 6, 6, 6]),
        matched_partner=np.asarray([1, 0, 3, 2]),
        unmatched_partner=np.asarray([2, 3, 0, 1]),
        alias_weight=np.ones(4, dtype=np.float32),
    )


def test_manifest_and_batch_sampler_are_deterministic(tmp_path):
    path = tmp_path / "pairs.npz"
    write_manifest(path)
    manifest = load_ncm_manifest(path, dataset_size=4)
    first = NCMPairedBatchSampler(manifest["batches"], seed=7)
    second = NCMPairedBatchSampler(manifest["batches"], seed=7)
    first_epoch = list(first)
    second_epoch = list(second)
    assert first_epoch == second_epoch
    assert sorted(first_epoch[0]) == [0, 1, 2, 3]


def test_manifest_rejects_same_branch_partner(tmp_path):
    path = tmp_path / "bad.npz"
    write_manifest(path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["matched_partner"] = np.asarray([1, 2, 1, 2])
    # Rows 1 and 2 share branch 1; validation must reject the pair.
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="does not change hidden branch"):
        load_ncm_manifest(path, dataset_size=4)


def test_partner_positions_do_not_depend_on_batch_order():
    samples = torch.tensor([30, 10, 40, 20])
    partners = torch.tensor([40, 20, 30, 10])
    positions = partner_positions(samples, partners)
    assert positions.tolist() == [2, 3, 0, 1]
    assert samples[positions].tolist() == partners.tolist()


def test_hybrid_sampler_covers_full_data_and_declared_pair_fraction():
    pair_batches = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]])
    sampler = NCMHybridBatchSampler(
        pair_batches=pair_batches,
        dataset_size=40,
        batch_size=4,
        pair_fraction=0.2,
        seed=17,
    )
    rows = list(iter(sampler))
    assert len(rows) == 10
    pair_sets = {frozenset(row) for row in pair_batches.tolist()}
    observed_pair_rows = sum(frozenset(row) in pair_sets for row in rows)
    assert observed_pair_rows == 2
    assert all(len(row) == 4 and len(set(row)) == 4 for row in rows)
    for row in rows:
        if frozenset(row) not in pair_sets:
            assert set(row).isdisjoint(set(pair_batches.reshape(-1)))


def test_hybrid_sampler_is_deterministic_across_equal_arms():
    kwargs = dict(
        pair_batches=np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]),
        dataset_size=40,
        batch_size=4,
        pair_fraction=0.2,
        seed=23,
    )
    left = NCMHybridBatchSampler(**kwargs)
    right = NCMHybridBatchSampler(**kwargs)
    assert list(iter(left)) == list(iter(right))
    assert list(iter(left)) == list(iter(right))


class FirstTokenMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, memory, ep_step=None):
        return memory[:, :1] * self.scale


class RecordingDropoutDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.dropout = nn.Dropout(0.5)
        self.dropout_outputs = []
        self.memory_inputs = []

    def forward(self, xt, t, condition=None, memory=None):
        stochastic = self.dropout(torch.ones_like(xt))
        self.dropout_outputs.append(stochastic.detach().clone())
        self.memory_inputs.append(memory.detach().clone())
        memory_value = memory.mean(dim=(1, 2))[:, None, None]
        return stochastic * self.scale + memory_value


class MutatingCondition(nn.Module):
    """Mimic the in-place mapping rewrite used by the image encoder."""

    def forward(self, values):
        value = values["obs"]
        assert value.ndim == 3
        batch, horizon, width = value.shape
        values["obs"] = value.flatten(end_dim=1)
        return values["obs"].reshape(batch, horizon, width)


class ConditionAgnosticDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, xt, t, condition=None, memory=None):
        return xt * self.scale


class ZeroOutputDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, xt, t, condition=None, memory=None):
        return torch.zeros_like(xt) + self.bias


def test_repeated_loss_does_not_mutate_caller_condition_mapping():
    args = SimpleNamespace(
        inference_action_steps=1,
        pred_reg_coeff=0.0,
        memory_aux_coeff=0.0,
        counterfactual_memory_coeff=0.0,
        counterfactual_memory_margin=0.1,
        counterfactual_memory_probability=1.0,
        counterfactual_negative_mode="none",
    )
    agent = LongMemDDPM(
        nn_diffusion=ConditionAgnosticDiffusion(),
        nn_condition=MutatingCondition(),
        diffusion_steps=4,
        x_max=torch.ones(1, 2, 1),
        x_min=-torch.ones(1, 2, 1),
        args_dict=args,
        device="cpu",
    )
    condition = {"obs": torch.zeros(4, 2, 3)}
    x0 = torch.randn(4, 2, 1)
    agent.loss(x0=x0, condition=condition)
    agent.loss(x0=x0, condition=condition)
    assert condition["obs"].shape == (4, 2, 3)


def test_matched_counterfactual_is_per_sample_and_shares_decoder_dropout():
    args = SimpleNamespace(
        inference_action_steps=1,
        pred_reg_coeff=0.0,
        memory_aux_coeff=0.0,
        counterfactual_memory_coeff=0.5,
        counterfactual_memory_margin=0.1,
        counterfactual_memory_probability=1.0,
        counterfactual_negative_mode="matched",
    )
    agent = LongMemDDPM(
        nn_diffusion=RecordingDropoutDiffusion(),
        nn_memory=FirstTokenMemory(),
        diffusion_steps=4,
        x_max=torch.ones(1, 2, 1),
        x_min=-torch.ones(1, 2, 1),
        args_dict=args,
        device="cpu",
        long_cond_dropout=0.5,
    )
    torch.manual_seed(7)
    agent.loss(
        x0=torch.randn(4, 2, 1),
        condition=torch.zeros(4, 1, 1),
        memory=(torch.arange(4, dtype=torch.float32) + 1).view(4, 1, 1),
        ep_step=torch.zeros(4, 1),
        counterfactual_index=torch.tensor([1, 0, 3, 2]),
        counterfactual_weight=torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    calls = agent.model["diffusion"].dropout_outputs
    assert len(calls) == 2
    assert torch.equal(calls[0], calls[1])
    correct_present = agent.model["diffusion"].memory_inputs[0].flatten(1).any(1)
    wrong_present = agent.model["diffusion"].memory_inputs[1].flatten(1).any(1)
    assert torch.equal(correct_present, wrong_present)
    assert agent._last_memory_losses["counterfactual_pairs"] == 2
    assert torch.isfinite(agent._last_memory_losses["counterfactual_gap"])


def test_donor_action_objective_uses_partner_action_as_positive_target():
    args = SimpleNamespace(
        inference_action_steps=1,
        pred_reg_coeff=0.0,
        memory_aux_coeff=0.0,
        counterfactual_memory_coeff=1.0,
        counterfactual_memory_margin=0.1,
        counterfactual_memory_probability=1.0,
        counterfactual_negative_mode="matched",
        counterfactual_objective="donor_action",
    )
    agent = LongMemDDPM(
        nn_diffusion=ZeroOutputDiffusion(),
        nn_memory=FirstTokenMemory(),
        diffusion_steps=4,
        x_max=torch.ones(1, 1, 1),
        x_min=-torch.ones(1, 1, 1),
        args_dict=args,
        device="cpu",
        predict_noise=False,
    )
    total, _, _, correct = agent.loss(
        x0=torch.tensor([[[1.0]], [[2.0]]]),
        condition=torch.zeros(2, 1, 1),
        memory=torch.tensor([[[10.0]], [[20.0]]]),
        ep_step=torch.zeros(2, 1),
        counterfactual_index=torch.tensor([1, 0]),
        counterfactual_weight=torch.tensor([1.0, 0.0]),
    )

    # Correct targets contribute mean((1^2, 2^2)) = 2.5.  Only anchor 0
    # is active, and its donor target is action 2.0, contributing 4.0.
    assert torch.isclose(correct, torch.tensor(2.5))
    assert torch.isclose(
        agent._last_memory_losses["counterfactual_donor_action_loss"],
        torch.tensor(4.0),
    )
    assert torch.isclose(total, torch.tensor(6.5))
