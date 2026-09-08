#!/usr/bin/env python3
import argparse
import pathlib

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cleandiffuser.dataset.dataset_utils import dynamic_batch_collate_fn
from cleandiffuser.dataset.rmbench_dataset import LongMemRMBenchImageDataset
from pipelines.longmemdpptp_rmbench_image_emb import (
    _concat_lowdim_obs,
    _move_normalizer_to_device,
    build_agent,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    args_cli = parser.parse_args()

    run_dir = pathlib.Path(args_cli.run_dir)
    args = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    args.device = args_cli.device
    normalizer = torch.load(
        run_dir / "models" / "rmbench_swap_blocks_normalizer.pt",
        map_location="cpu",
        weights_only=False,
    )
    _move_normalizer_to_device(normalizer, args.device)

    dataset = LongMemRMBenchImageDataset(
        dataset_dir=args.dataset_path,
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        horizon=args.horizon,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        dataset_memory_type=args.dataset_memory_type,
        subsample_ratio=args.dataset_memory_subsample_ratio,
        maximum_memory_length=args.qformer_cache_max_length,
        load_lowdim_to_memory=args.load_lowdim_to_memory,
        device="cpu",
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=dynamic_batch_collate_fn,
            )
        )
    )

    agent = build_agent(args, normalizer)
    agent.load(args_cli.checkpoint)
    condition = _concat_lowdim_obs(batch["obs"], normalizer, args)[:, : args.obs_steps]
    memory = _concat_lowdim_obs(batch["obs"], normalizer, args, suffix="_mem")
    ep_step = batch["obs"]["ep_step"].int().to(args.device)
    prior = torch.zeros((2, args.horizon, args.action_dim), device=args.device)
    agent.reset_cache()
    with torch.no_grad():
        sample, _ = agent.sample(
            prior=prior,
            n_samples=2,
            sample_steps=args.sample_steps,
            condition_cfg=condition,
            mask_cfg=None,
            memory_cfg=memory,
            ep_step_cfg=ep_step,
            temperature=args.temperature,
        )
        delta = normalizer["action"].unnormalize(sample)
        current = batch["obs"]["agent_pos"][:, args.obs_steps - 1].to(args.device)
        actions = delta + current[:, None, :]

    if not torch.isfinite(actions).all():
        raise RuntimeError("Sampled actions contain NaN or Inf")
    max_delta = torch.amax(torch.abs(actions[:, 1] - current), dim=-1).cpu().tolist()
    print(f"checkpoint_load=ok sample_shape={tuple(actions.shape)}")
    print(
        f"action_min={float(actions.min()):.6f} "
        f"action_max={float(actions.max()):.6f} "
        f"execute_step_max_delta={max_delta}"
    )


if __name__ == "__main__":
    main()
