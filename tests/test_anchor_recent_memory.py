import torch

from cleandiffuser.nn_memory.anchor_recent_memory import AnchorRecentMemory


def test_anchor_recent_forward_keeps_endpoints():
    memory = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    module = AnchorRecentMemory().eval()
    assert torch.equal(module(memory), memory[:, [0, 4]])


def test_anchor_recent_stream_keeps_first_and_latest():
    module = AnchorRecentMemory().eval()
    module.inference(torch.tensor([[[1.0], [2.0]]]))
    output = module.inference(torch.tensor([[[3.0]]]))
    assert torch.equal(output, torch.tensor([[[1.0], [3.0]]]))
