import torch

from cleandiffuser.nn_condition.momart_history_condition import (
    TaskConditionedEmbeddingEncoder,
)


def _obs(batch=3, time=5, dim=8):
    return {
        "dp_emb": torch.zeros(batch, time, dim),
        "_task_id": torch.tensor([0, 1, 0]),
    }


def test_embedding_encoder_adds_task_condition_to_every_token():
    encoder = TaskConditionedEmbeddingEncoder(num_tasks=2, emb_dim=8)
    output = encoder(_obs())
    assert output.shape == (3, 5, 8)
    assert torch.allclose(output[0, 0], output[0, -1])
    assert not torch.allclose(output[0, 0], output[1, 0])


def test_direct_embedding_history_has_fixed_length():
    encoder = TaskConditionedEmbeddingEncoder(
        num_tasks=2, emb_dim=8, history_frames=7)
    output = encoder.encode_spatial_history(_obs())
    assert output.shape == (3, 7, 8)


def test_direct_token2_embedding_history_is_pooled():
    encoder = TaskConditionedEmbeddingEncoder(
        num_tasks=2, emb_dim=8, history_frames=7,
        history_output_tokens=2)
    output = encoder.encode_spatial_history(_obs())
    assert output.shape == (3, 2, 8)
