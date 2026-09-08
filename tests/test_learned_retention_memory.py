import torch

from cleandiffuser.nn_memory.learned_retention_memory import (
    LearnedRetentionMemory, RetentionHead, retention_features)


def test_retention_features_are_causal_and_shaped():
    memory = torch.randn(3, 7, 16)
    steps = torch.arange(7).repeat(3, 1)
    features = retention_features(memory, steps)
    assert features.shape == (3, 7, 34)
    assert torch.isfinite(features).all()


def test_learned_retention_offline_and_online_shapes():
    module = LearnedRetentionMemory(
        mem_compress_length=2,
        emb_dim=16,
        num_layers=1,
        num_heads=4,
        bank_size=4,
        subsample_ratio=1,
    ).eval()
    memory = torch.randn(2, 9, 16)
    steps = torch.arange(9).repeat(2, 1)
    offline = module(memory, steps)
    assert offline.shape == (2, 2, 16)

    module.reset()
    online = None
    for index in range(memory.shape[1]):
        online = module.inference(memory[:, index:index + 1])
    assert online.shape == (2, 2, 16)
    assert module.bank.shape == (2, 4, 16)


def test_retention_head_checkpoint_round_trip(tmp_path):
    head = RetentionHead(emb_dim=16)
    path = tmp_path / "head.pt"
    torch.save({"retention_head": head.state_dict()}, path)
    module = LearnedRetentionMemory(
        mem_compress_length=2,
        emb_dim=16,
        num_layers=1,
        num_heads=4,
        bank_size=4,
        retention_head_path=str(path),
    )
    for expected, actual in zip(head.parameters(), module.retention_head.parameters()):
        assert torch.equal(expected, actual)
