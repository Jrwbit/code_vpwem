import pytest
import torch

from cleandiffuser.diffusion.longmem_ddpm import apply_memory_intervention


def test_none_preserves_memory_object():
    memory = torch.randn(3, 4, 8)
    assert apply_memory_intervention(memory, "none") is memory


def test_zero_removes_memory_content():
    memory = torch.randn(3, 4, 8)
    result = apply_memory_intervention(memory, "zero")
    assert torch.count_nonzero(result) == 0
    assert result.shape == memory.shape


def test_shuffle_batch_uses_other_trajectories():
    memory = torch.arange(3 * 2 * 2).reshape(3, 2, 2)
    result = apply_memory_intervention(memory, "shuffle_batch")
    assert torch.equal(result[0], memory[-1])
    assert torch.equal(result[1], memory[0])


def test_reverse_tokens_preserves_batch_and_reverses_time():
    memory = torch.arange(2 * 3).reshape(2, 3, 1)
    result = apply_memory_intervention(memory, "reverse_tokens")
    assert torch.equal(result[:, 0], memory[:, -1])


def test_unknown_intervention_fails_loudly():
    with pytest.raises(ValueError, match="Unknown memory intervention"):
        apply_memory_intervention(torch.ones(2, 2, 2), "invalid")
