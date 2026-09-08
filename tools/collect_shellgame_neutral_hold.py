#!/usr/bin/env python3
"""Collect MIKASA demonstrations with a physical neutral prefix.

For ShellGameTouch the registered protocol holds through step 5 and releases
the oracle at step 6. RememberColor uses the same physical construction with a
task-specific reveal/decision step. The legacy shifted-action replay remains
available only to reproduce the explicitly marked ShellGame diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mikasa_robo_suite  # noqa: F401
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info
from baselines.ppo.ppo_memtasks import (
    AgentStateOnly,
    FlattenRGBDObservationWrapper,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="ShellGameTouch-v0")
    parser.add_argument(
        "--branch-attribute",
        help=("Unwrapped environment tensor containing the hidden branch. "
              "Defaults to the known attribute for ShellGameTouch or "
              "RememberColor."))
    parser.add_argument("--num-branches", type=int)
    parser.add_argument(
        "--source-dataset", required=True,
        help=("Original fixed-horizon demonstrations. Their actions are replayed one "
              "step later after the physical neutral hold."))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--oracle-checkpoint",
        help=("Released MIKASA state-oracle checkpoint. When supplied, actions "
              "after the neutral prefix are generated in closed loop and all "
              "reward/success/done labels come from the physical rollout."))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--neutral-through-step", type=int, default=5)
    parser.add_argument("--episode-timeout", type=int, default=90)
    return parser.parse_args()


def script_sha256():
    digest = hashlib.sha256()
    with open(__file__, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numpy_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def wrap_vector_env(raw_env, env_name: str, batch_size: int, *, state: bool):
    wrappers, timeout = env_info(env_name)
    for wrapper, kwargs in wrappers:
        raw_env = wrapper(raw_env, **kwargs)
    env = FlattenRGBDObservationWrapper(
        raw_env, rgb=not state, depth=False, state=state,
        oracle=False, joints=not state)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env, batch_size, ignore_terminations=True, record_metrics=True)
    return env, timeout


def build_envs(env_name: str, batch_size: int, with_state_oracle: bool):
    common = dict(
        control_mode="pd_joint_delta_pos", render_mode="all",
        sim_backend="gpu", reward_mode="normalized_dense")
    raw_rgb = gym.make(
        env_name, num_envs=batch_size, obs_mode="rgb", **common)
    raw_rgb_base = raw_rgb
    rgb, timeout = wrap_vector_env(
        raw_rgb, env_name, batch_size, state=False)
    state_env = None
    if with_state_oracle:
        raw_state = gym.make(
            env_name, num_envs=batch_size,
            obs_mode="state", **common)
        state_env, state_timeout = wrap_vector_env(
            raw_state, env_name, batch_size, state=True)
        if int(state_timeout) != int(timeout):
            raise RuntimeError("RGB/state environment timeouts disagree")
    return rgb, raw_rgb_base, state_env, timeout


def load_source_batch(source: Path, episode_ids, episode_timeout: int):
    actions, rewards, successes, dones = [], [], [], []
    first_rgb, first_joints = [], []
    for episode_id in episode_ids:
        path = source / f"train_data_{episode_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as episode:
            if len(episode["action"]) != episode_timeout:
                raise ValueError(
                    f"{path} has {len(episode['action'])} actions; "
                    f"expected {episode_timeout}")
            actions.append(episode["action"])
            rewards.append(episode["reward"])
            successes.append(episode["success"])
            dones.append(episode["done"])
            first_rgb.append(episode["rgb"][0])
            first_joints.append(episode["joints"][0])
    return {
        "action": np.stack(actions, axis=1),
        "reward": np.stack(rewards, axis=1),
        "success": np.stack(successes, axis=1),
        "done": np.stack(dones, axis=1),
        "rgb0": np.stack(first_rgb),
        "joints0": np.stack(first_joints),
    }


def write_marker(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("episodes and batch-size must be positive")
    if args.episodes % args.batch_size:
        raise ValueError("episodes must be divisible by batch-size")
    if args.neutral_through_step < 0:
        raise ValueError("neutral-through-step must be non-negative")
    if args.neutral_through_step >= args.episode_timeout - 1:
        raise ValueError("neutral prefix must leave at least one oracle action")
    branch_attribute = args.branch_attribute
    if branch_attribute is None:
        if args.env_name == "ShellGameTouch-v0":
            branch_attribute = "cup_with_ball_number"
        elif args.env_name.startswith("RememberColor"):
            branch_attribute = "true_color_indices"
        else:
            raise ValueError(
                "--branch-attribute is required for this environment")
    num_branches = args.num_branches
    if num_branches is None:
        if args.env_name == "ShellGameTouch-v0":
            num_branches = 3
        elif args.env_name.startswith("RememberColor"):
            match = re.fullmatch(r"RememberColor(\d+)-v\d+", args.env_name)
            num_branches = int(match.group(1)) if match else None
    if not num_branches or num_branches < 2:
        raise ValueError("--num-branches must resolve to at least two")

    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_dataset).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(source)
    oracle_checkpoint = (
        Path(args.oracle_checkpoint).expanduser()
        if args.oracle_checkpoint else None)
    if oracle_checkpoint is not None and not oracle_checkpoint.is_file():
        raise FileNotFoundError(oracle_checkpoint)
    closed_loop_oracle = oracle_checkpoint is not None

    marker_path = output / "NCM_PHYSICAL_HOLD.json"
    labels_path = output / "branch_labels.npy"
    physical_success_path = output / (
        "physical_rollout_success.npy" if closed_loop_oracle
        else "physical_replay_success.npy")
    labels = (np.load(labels_path, allow_pickle=False).astype(np.int8)
              if labels_path.exists()
              else np.full(args.episodes, -1, dtype=np.int8))
    if len(labels) != args.episodes:
        raise ValueError("Existing branch_labels.npy has an incompatible length")
    physical_success = (
        np.load(physical_success_path, allow_pickle=False).astype(np.int8)
        if physical_success_path.exists()
        else np.full(args.episodes, -1, dtype=np.int8))
    if len(physical_success) != args.episodes:
        raise ValueError(
            "Existing physical_replay_success.npy has an incompatible length")
    marker = {
        "schema_version": 2 if closed_loop_oracle else 1,
        "complete": False,
        "env_name": args.env_name,
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "episode_timeout": args.episode_timeout,
        "neutral_through_step": args.neutral_through_step,
        "first_policy_action_step": args.neutral_through_step + 1,
        "neutral_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "demonstrator": (
            "released_mikasa_state_oracle_closed_loop" if closed_loop_oracle
            else "physical_replay_of_source_actions_shifted_one_step"),
        "dataset_role": (
            "physical_closed_loop_decision_supervision_at_step_"
            f"{args.neutral_through_step + 1}"
            if closed_loop_oracle
            else "hybrid_decision_supervision_at_step_"
                 f"{args.neutral_through_step + 1}_only"),
        "branch_attribute": branch_attribute,
        "num_branches": num_branches,
        "observation_provenance": "physical_simulator_rollout",
        "action_provenance": (
            "released_mikasa_state_oracle_closed_loop"
            if closed_loop_oracle
            else "source_demonstrator_shifted_one_step"),
        "reward_success_done_provenance": (
            "physical_simulator_rollout" if closed_loop_oracle else
            "source_demonstrator_shifted_one_step_for_episode_retention"),
        "source_dataset": str(source),
        "source_action_shift": 0 if closed_loop_oracle else 1,
        "oracle_checkpoint": (
            str(oracle_checkpoint) if oracle_checkpoint is not None else None),
        "oracle_checkpoint_sha256": (
            file_sha256(oracle_checkpoint)
            if oracle_checkpoint is not None else None),
        "collector_sha256": script_sha256(),
    }
    write_marker(marker_path, marker)

    env_rgb, raw_rgb_base, env_state, registered_timeout = build_envs(
        args.env_name, args.batch_size,
        with_state_oracle=closed_loop_oracle)
    if int(registered_timeout) != args.episode_timeout:
        raise ValueError(
            f"Registered timeout {registered_timeout} != {args.episode_timeout}")
    agent = None
    if closed_loop_oracle:
        oracle_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        agent = AgentStateOnly(env_state)
        checkpoint = torch.load(
            oracle_checkpoint, map_location=oracle_device,
            weights_only=False)
        agent.load_state_dict(checkpoint, strict=True)
        agent.to(oracle_device).eval()
    try:
        for batch_seed in range(args.episodes // args.batch_size):
            first_id = batch_seed * args.batch_size
            episode_ids = range(first_id, first_id + args.batch_size)
            complete = all(
                (output / f"train_data_{episode_id}.npz").is_file()
                and labels[episode_id] >= 0
                and physical_success[episode_id] >= 0
                for episode_id in episode_ids)
            if complete:
                print(f"Skipping complete batch {batch_seed}", flush=True)
                continue

            obs_state = None
            if env_state is not None:
                obs_state, _ = env_state.reset(seed=batch_seed)
            obs_rgb, _ = env_rgb.reset(seed=batch_seed)
            source_batch = load_source_batch(
                source, episode_ids, args.episode_timeout)
            observed_rgb0 = numpy_value(obs_rgb["rgb"])
            observed_joints0 = numpy_value(obs_rgb["joints"])
            rgb_mae = float(np.abs(
                observed_rgb0.astype(np.float32)
                - source_batch["rgb0"].astype(np.float32)).mean())
            joints_linf = float(np.abs(
                observed_joints0.astype(np.float32)
                - source_batch["joints0"].astype(np.float32)).max())
            if rgb_mae > 1e-3 or joints_linf > 1e-5:
                raise RuntimeError(
                    "Source demonstrations do not align with the deterministic "
                    f"reset: rgb_mae={rgb_mae:.6g}, "
                    f"joints_linf={joints_linf:.6g}")
            branch_value = getattr(raw_rgb_base.unwrapped, branch_attribute)
            batch_branches = numpy_value(branch_value).astype(np.int8).reshape(-1)
            if len(batch_branches) != args.batch_size:
                raise RuntimeError("Could not read one branch per vector environment")

            rgb_list, joints_list, action_list = [], [], []
            reward_list, success_list, done_list = [], [], []
            physical_success_list = []
            for step in range(args.episode_timeout):
                rgb_list.append(numpy_value(obs_rgb["rgb"]))
                joints_list.append(numpy_value(obs_rgb["joints"]))
                if step <= args.neutral_through_step:
                    action = torch.zeros(
                        source_batch["action"].shape[-1],
                        device=obs_rgb["joints"].device,
                        dtype=obs_rgb["joints"].dtype).expand(
                            args.batch_size, -1).clone()
                    # pd_joint_delta_pos uses the final dimension as an
                    # absolute gripper command. +1 keeps it open; a literal
                    # all-zero vector is not a physical hold for the gripper.
                    action[:, -1] = 1.0
                elif closed_loop_oracle:
                    with torch.no_grad():
                        action = agent.get_action(
                            obs_state, deterministic=True)
                else:
                    action = torch.as_tensor(
                        source_batch["action"][step - 1],
                        device=obs_rgb["joints"].device,
                        dtype=obs_rgb["joints"].dtype)

                if env_state is not None:
                    obs_state, state_reward, state_terminated, \
                        state_truncated, state_info = env_state.step(action)
                obs_rgb, reward, terminated, truncated, info = env_rgb.step(action)
                action_list.append(numpy_value(action))
                physical_success_list.append(
                    numpy_value(info["success"]).astype(np.int64))
                if closed_loop_oracle:
                    reward_list.append(numpy_value(reward))
                    success_list.append(
                        numpy_value(info["success"]).astype(np.int64))
                    done_list.append(numpy_value(
                        torch.logical_or(terminated, truncated)).astype(
                            np.int64))
                    if not np.allclose(
                            numpy_value(state_reward), numpy_value(reward),
                            atol=1e-5, rtol=1e-5):
                        raise RuntimeError(
                            "State/RGB physical rollouts desynchronized")
                elif step <= args.neutral_through_step:
                    reward_list.append(np.zeros(args.batch_size, dtype=np.float32))
                    success_list.append(np.zeros(args.batch_size, dtype=np.int64))
                    done_list.append(np.zeros(args.batch_size, dtype=np.int64))
                else:
                    source_step = step - 1
                    reward_list.append(source_batch["reward"][source_step])
                    success_list.append(source_batch["success"][source_step])
                    done_list.append(source_batch["done"][source_step])

            arrays = {
                "rgb": np.asarray(rgb_list),
                "joints": np.asarray(joints_list),
                "action": np.asarray(action_list),
                "reward": np.asarray(reward_list),
                "success": np.asarray(success_list),
                "done": np.asarray(done_list),
            }
            for slot, episode_id in enumerate(episode_ids):
                payload = {
                    key: value[:, slot] for key, value in arrays.items()
                }
                np.savez_compressed(
                    output / f"train_data_{episode_id}.npz", **payload)
                labels[episode_id] = batch_branches[slot]
                physical_success[episode_id] = int(bool(np.any(
                    np.asarray(physical_success_list)[:, slot])))
            np.save(labels_path, labels)
            np.save(physical_success_path, physical_success)
            physical_successes = int(np.any(
                np.asarray(physical_success_list), axis=0).sum())
            print(
                f"Saved batch {batch_seed}; episodes "
                f"{first_id}-{first_id + args.batch_size - 1}; "
                f"physical replay successes={physical_successes}/"
                f"{args.batch_size}", flush=True)
    finally:
        env_rgb.close()
        if env_state is not None:
            env_state.close()

    if np.any(labels < 0) or np.any(physical_success < 0):
        raise RuntimeError("Collection ended with missing sidecar labels")
    marker["complete"] = True
    marker["branch_counts"] = np.bincount(
        labels, minlength=num_branches).tolist()
    successful = 0
    for episode_id in range(args.episodes):
        with np.load(output / f"train_data_{episode_id}.npz") as episode:
            successful += int(bool(np.any(episode["success"])))
    marker["successful_episodes"] = successful
    marker[
        "physical_rollout_successful_episodes" if closed_loop_oracle
        else "physical_replay_successful_episodes"
    ] = int(physical_success.sum())
    write_marker(marker_path, marker)
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
