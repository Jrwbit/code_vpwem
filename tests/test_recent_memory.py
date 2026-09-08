import torch

from cleandiffuser.nn_memory.recent_memory import RecentMemory


def test_recent_forward_keeps_last_two():
    memory = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    module = RecentMemory(mem_compress_length=2).eval()
    assert torch.equal(module(memory), memory[:, -2:])


def test_recent_inference_is_streaming_and_resettable():
    module = RecentMemory(mem_compress_length=2).eval()
    module.inference(torch.tensor([[[1.0], [2.0]]]))
    output = module.inference(torch.tensor([[[3.0]]]))
    assert torch.equal(output, torch.tensor([[[2.0], [3.0]]]))
    module.reset()
    assert torch.equal(module.inference(torch.tensor([[[4.0]]])), torch.tensor([[[4.0]]]))
