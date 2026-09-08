import torch

from cleandiffuser.nn_memory.random_memory import RandomMemory


def test_training_indices_are_independent_across_batch():
    memory = torch.arange(8, dtype=torch.float32).view(1, 8, 1).repeat(32, 1, 1)
    module = RandomMemory(mem_compress_length=2, dropout=0.0, retention_seed=7)
    module.train()

    selected = module(memory).squeeze(-1)

    assert torch.unique(selected, dim=0).shape[0] > 1
    assert torch.all(selected[:, 0] < selected[:, 1])


def test_training_does_not_sample_left_padding():
    memory = torch.tensor([
        [[-9.0], [-9.0], [-9.0], [10.0], [11.0]],
        [[-9.0], [20.0], [21.0], [22.0], [23.0]],
    ])
    ep_steps = torch.tensor([[0, 0, 0, 0, 5], [0, 2, 4, 6, 8]])
    module = RandomMemory(mem_compress_length=2, dropout=0.0, retention_seed=3)
    module.eval()

    selected = module(memory, ep_steps).squeeze(-1)

    assert torch.all(selected >= 10)


def test_inference_can_reject_first_overflow_item():
    # At stream index 2 with k=2, Algorithm R must draw from {0, 1, 2}; j=2
    # means retaining the two old items. Seed 0 produces j=2 on the first draw.
    module = RandomMemory(mem_compress_length=2, retention_seed=0)
    first = torch.tensor([[[0.0], [1.0]]])
    overflow = torch.tensor([[[2.0]]])

    module.inference(first)
    retained = module.inference(overflow)

    assert retained.squeeze().tolist() == [0.0, 1.0]


def test_algorithm_r_is_marginally_uniform():
    stream_length = 10
    trials = 12000
    counts = torch.zeros(stream_length, dtype=torch.int64)
    for seed in range(trials):
        module = RandomMemory(mem_compress_length=2, retention_seed=seed)
        stream = torch.arange(stream_length, dtype=torch.float32).view(1, -1, 1)
        retained = module.inference(stream).to(torch.int64).flatten()
        counts.scatter_add_(0, retained, torch.ones_like(retained))

    expected = trials * 2 / stream_length
    # A generous deterministic bound: catches the previous early-item bias
    # while avoiding a flaky statistical assertion.
    assert torch.max(torch.abs(counts.float() - expected)) < expected * 0.08


def test_retention_rng_does_not_advance_global_torch_rng():
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)

    module = RandomMemory(mem_compress_length=2, dropout=0.0, retention_seed=99)
    module.eval()
    module(torch.zeros(3, 8, 1))
    module.inference(torch.zeros(3, 8, 1))

    assert torch.equal(torch.rand(4), expected)
