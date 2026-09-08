#!/usr/bin/env python3
"""Evaluate the official state PPO oracle on current RememberColor3-v0."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mikasa_robo_suite  # noqa: F401 - registers environments
from baselines.ppo.ppo_memtasks import AgentStateOnly, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-episode-steps", type=int, default=60)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.max_episode_steps != 60:
        raise ValueError("RememberColor3 official horizon is 60")

    env = gym.make(
        "RememberColor3-v0",
        num_envs=args.episodes,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )
    wrappers, registered_timeout = env_info("RememberColor3-v0")
    if int(registered_timeout) != args.max_episode_steps:
        raise RuntimeError(
            f"registered horizon {registered_timeout} != requested horizon 60")
    for wrapper_class, wrapper_kwargs in wrappers:
        env = wrapper_class(env, **wrapper_kwargs)
    env = FlattenRGBDObservationWrapper(
        env, rgb=False, depth=False, state=True,
        oracle=False, joints=False)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env, args.episodes, ignore_terminations=True, record_metrics=True)

    try:
        agent = AgentStateOnly(env).to(env.base_env.device)
        state_dict = torch.load(checkpoint, map_location=env.base_env.device)
        agent.load_state_dict(state_dict, strict=True)
        agent.eval()
        obs, reset_info = env.reset(seed=args.seed)

        gt = _numpy(reset_info["oracle_info"]).reshape(args.episodes, -1)[:, 0]
        success_once = np.zeros(args.episodes, dtype=bool)
        placed_once = np.zeros(args.episodes, dtype=bool)
        placed_and_static_once = np.zeros(args.episodes, dtype=bool)
        success_time = np.full(args.episodes, -1, dtype=np.int32)
        min_correct_distance = np.full(args.episodes, np.inf, dtype=np.float32)
        final_success = np.zeros(args.episodes, dtype=bool)
        final_placed = np.zeros(args.episodes, dtype=bool)
        final_static = np.zeros(args.episodes, dtype=bool)

        for step in range(args.max_episode_steps):
            with torch.no_grad():
                action = agent.get_action(
                    {key: value.to(env.base_env.device)
                     for key, value in obs.items()},
                    deterministic=True,
                )
            obs, _, _, _, info = env.step(action)
            success = _numpy(info["success"])[:args.episodes].astype(bool)
            placed = _numpy(info["is_obj_placed"])[:args.episodes].astype(bool)
            static = _numpy(info["is_robot_static"])[:args.episodes].astype(bool)
            distance = np.linalg.norm(
                _numpy(info["obj_to_goal_pos"])[:args.episodes], axis=-1)
            first = success & (success_time < 0)
            success_time[first] = step + 1
            success_once |= success
            placed_once |= placed
            placed_and_static_once |= placed & static
            min_correct_distance = np.minimum(min_correct_distance, distance)
            final_success, final_placed, final_static = success, placed, static

        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        episodes = []
        for episode in range(args.episodes):
            episodes.append({
                "episode": episode,
                "ground_truth_branch": int(gt[episode]),
                "success_once": bool(success_once[episode]),
                "success_time": int(success_time[episode]),
                "placed_once": bool(placed_once[episode]),
                "placed_and_static_once": bool(
                    placed_and_static_once[episode]),
                "min_correct_target_distance": float(
                    min_correct_distance[episode]),
                "final_success": bool(final_success[episode]),
                "final_placed": bool(final_placed[episode]),
                "final_static": bool(final_static[episode]),
            })
        result = {
            "diagnostics_schema_version": 1,
            "env_name": "RememberColor3-v0",
            "observation_mode": "state",
            "oracle_checkpoint": str(checkpoint),
            "oracle_checkpoint_sha256": digest,
            "episodes_count": args.episodes,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
            "saved_images": False,
            "success_rate": float(success_once.mean()),
            "placed_rate": float(placed_once.mean()),
            "placed_and_static_rate": float(placed_and_static_once.mean()),
            "final_success_rate": float(final_success.mean()),
            "mean_success_time": float(success_time[success_time >= 0].mean())
            if success_once.any() else None,
            "mean_min_correct_target_distance": float(
                min_correct_distance.mean()),
            "ground_truth_counts": {
                str(branch): int((gt == branch).sum()) for branch in range(3)
            },
            "episodes": episodes,
        }
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps({
            "output": str(output),
            "success_rate": result["success_rate"],
            "placed_rate": result["placed_rate"],
            "placed_and_static_rate": result["placed_and_static_rate"],
            "mean_success_time": result["mean_success_time"],
        }, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
