from types import SimpleNamespace

import numpy as np

from pipelines.eval_fair_history_momart_multitask_emb import (
    env_args,
    env_factory,
    make_dpptp_async_envs,
    make_longmem_async_envs,
    normalize_observations,
)


def test_no_history_uses_dpptp_observation_window():
    assert env_factory("no_history") is make_dpptp_async_envs


def test_history_methods_use_longmem_observation_window():
    for method in ("vpwem", "direct", "direct_token2"):
        assert env_factory(method) is make_longmem_async_envs


def test_no_history_env_receives_training_observation_steps():
    cli = SimpleNamespace(
        dataset_path="/tmp/example.hdf5",
        num_envs=10,
        eval_episodes=30,
        num_video_episodes=1,
        work_dir="/tmp/eval",
        seed=20260917,
    )
    config = SimpleNamespace(obs_steps=2, action_steps=8)

    args = env_args(cli, config)

    assert args.obs_steps == 2
    assert args.action_steps == 8


def test_observation_conversion_materializes_contiguous_float32_tensor():
    class IdentityNormalizer:
        def normalize(self, value):
            return value

    dataset = SimpleNamespace(
        normalizer={"obs": {"rgb": IdentityNormalizer()}})
    base = np.zeros((2, 2, 3, 8, 8), dtype=np.uint8)
    strided = base[..., ::2]

    normalized = normalize_observations(
        {"rgb": strided}, dataset, device="cpu")

    assert normalized["rgb"].is_contiguous()
    assert str(normalized["rgb"].dtype) == "torch.float32"
