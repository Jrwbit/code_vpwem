#!/usr/bin/env python3
"""Replay ten stored RememberColor3 expert trajectories in the current env.

The original dataset was collected in batches of 250.  By default this script
recreates that vector width and seed, then audits indices 0..9 while harmless
zero actions are sent to the unused vector slots.  This preserves the original
seed-to-vector-index mapping needed for an environment-version parity check.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mikasa_robo_suite
from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info

from cleandiffuser.utils.remembercolor_diagnostics import (
    COLOR_NAMES,
    array_comparison,
    infer_primary_color_branch,
)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _make_collector_env_pair(batch_size):
    common_kwargs = dict(
        num_envs=batch_size,
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )
    # Match collect_batched_data_from_ckpt: create both raw environments first,
    # then finish wrapping state before RGB.
    env_state = gym.make(
        "RememberColor3-v0",
        obs_mode="state",
        **common_kwargs,
    )
    env_rgb = gym.make(
        "RememberColor3-v0",
        obs_mode="rgb",
        **common_kwargs,
    )

    wrappers, state_timeout = env_info("RememberColor3-v0")
    for wrapper_class, wrapper_kwargs in wrappers:
        env_state = wrapper_class(env_state, **wrapper_kwargs)
    env_state = FlattenRGBDObservationWrapper(
        env_state, rgb=False, depth=False, state=True,
        oracle=False, joints=False)

    wrappers, rgb_timeout = env_info("RememberColor3-v0")
    for wrapper_class, wrapper_kwargs in wrappers:
        env_rgb = wrapper_class(env_rgb, **wrapper_kwargs)
    env_rgb = FlattenRGBDObservationWrapper(
        env_rgb, rgb=True, depth=False, state=False,
        oracle=False, joints=True)
    if isinstance(env_state.action_space, gym.spaces.Dict):
        env_state = FlattenActionSpaceWrapper(env_state)
    if isinstance(env_rgb.action_space, gym.spaces.Dict):
        env_rgb = FlattenActionSpaceWrapper(env_rgb)
    env_state = ManiSkillVectorEnv(
        env_state, batch_size, ignore_terminations=True, record_metrics=True)
    env_rgb = ManiSkillVectorEnv(
        env_rgb, batch_size, ignore_terminations=True, record_metrics=True)
    return env_state, env_rgb, state_timeout, rgb_timeout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="/home/jrw/vpwem_data/mikasa/RememberColor3-v0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--source-batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=60)
    parser.add_argument(
        "--compare-frames", type=int, nargs="+", default=[0, 5, 10, 59])
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.episodes <= args.source_batch_size:
        raise ValueError("episodes must be within the recreated source batch")
    if args.max_episode_steps != 60:
        raise ValueError("RC3 expert parity must use the registered 60-step horizon")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    trajectories = []
    for index in range(args.episodes):
        path = dataset_dir / f"train_data_{index}.npz"
        with np.load(path) as archive:
            trajectories.append({key: archive[key].copy() for key in archive.files})
    actions = np.stack([trajectory["action"] for trajectory in trajectories], axis=1)
    if actions.shape[0] < args.max_episode_steps:
        raise ValueError(f"stored trajectory is too short: {actions.shape}")

    # Recreate the original collector's construction order, not merely its RGB
    # environment.  The environment uses global torch randomness for placement,
    # so omitting the parallel state environment would make pixel reset parity
    # ambiguous even when the package version is identical.
    env_state, env_rgb, state_timeout, registered_timeout = (
        _make_collector_env_pair(args.source_batch_size))
    if int(registered_timeout) != args.max_episode_steps:
        raise RuntimeError(
            f"registered timeout changed: {registered_timeout} != {args.max_episode_steps}")
    if state_timeout != registered_timeout:
        raise RuntimeError("state and RGB collector horizons disagree")

    try:
        _, _ = env_state.reset(seed=args.seed)
        obs, reset_info = env_rgb.reset(seed=args.seed)
        current_gt = _numpy(reset_info["oracle_info"])[:args.episodes].reshape(
            args.episodes, -1)[:, 0].astype(np.int64)
        success_once = np.zeros(args.episodes, dtype=bool)
        current_reward = np.zeros(
            (args.max_episode_steps, args.episodes), dtype=np.float32)
        current_success = np.zeros(
            (args.max_episode_steps, args.episodes), dtype=bool)
        frame_comparisons = {index: {} for index in range(args.episodes)}
        compare_frames = set(args.compare_frames)
        final_info = None

        for step in range(args.max_episode_steps):
            current_rgb = _numpy(obs["rgb"])[:args.episodes]
            current_joints = _numpy(obs["joints"])[:args.episodes]
            if step in compare_frames:
                for episode in range(args.episodes):
                    reference_rgb = trajectories[episode]["rgb"][step]
                    reference_joints = trajectories[episode]["joints"][step]
                    frame_comparisons[episode][step] = {
                        "rgb": array_comparison(reference_rgb, current_rgb[episode]),
                        "joints": array_comparison(
                            reference_joints, current_joints[episode]),
                        "reference_rgb_sha256": _sha256(reference_rgb),
                        "current_rgb_sha256": _sha256(current_rgb[episode]),
                        "reference_cue_branch": int(
                            infer_primary_color_branch(reference_rgb[..., :3]))
                        if step == 0 else None,
                        "current_cue_branch": int(
                            infer_primary_color_branch(current_rgb[episode][..., :3]))
                        if step == 0 else None,
                    }

            action_batch = np.zeros(
                (args.source_batch_size, actions.shape[-1]), dtype=np.float32)
            action_batch[:args.episodes] = actions[step]
            action_tensor = torch.as_tensor(
                action_batch, device=env_rgb.base_env.device)
            # Preserve the collector's state-then-RGB stepping order.
            env_state.step(action_tensor)
            obs, reward, terminated, truncated, info = env_rgb.step(action_tensor)
            final_info = info
            current_reward[step] = _numpy(reward)[:args.episodes]
            current_success[step] = _numpy(info["success"])[:args.episodes].astype(bool)
            success_once |= current_success[step]

        stored_reward = np.stack(
            [trajectory["reward"][:args.max_episode_steps]
             for trajectory in trajectories], axis=1)
        stored_success = np.stack(
            [trajectory["success"][:args.max_episode_steps]
             for trajectory in trajectories], axis=1).astype(bool)
        stored_done = np.stack(
            [trajectory["done"][:args.max_episode_steps]
             for trajectory in trajectories], axis=1).astype(bool)
        current_placed = _numpy(final_info["is_obj_placed"])[:args.episodes].astype(bool)
        current_static = _numpy(final_info["is_robot_static"])[:args.episodes].astype(bool)
        current_distance = np.linalg.norm(
            _numpy(final_info["obj_to_goal_pos"])[:args.episodes], axis=-1)
        current_terminated = _numpy(terminated)[:args.episodes].astype(bool)
        current_truncated = _numpy(truncated)[:args.episodes].astype(bool)

        per_episode = []
        for episode in range(args.episodes):
            cue_branch = frame_comparisons[episode][0]["reference_cue_branch"]
            per_episode.append({
                "episode": episode,
                "source_batch_seed": args.seed,
                "source_vector_index": episode,
                "dataset_cue_branch": cue_branch,
                "dataset_cue_color": COLOR_NAMES[cue_branch],
                "current_ground_truth_branch": int(current_gt[episode]),
                "current_ground_truth_color": COLOR_NAMES[current_gt[episode]],
                "cue_matches_current_ground_truth": bool(
                    cue_branch == current_gt[episode]),
                "frame_comparisons": frame_comparisons[episode],
                "dataset_success_once": bool(stored_success[:, episode].any()),
                "current_success_once": bool(success_once[episode]),
                "dataset_first_success_time": int(np.flatnonzero(
                    stored_success[:, episode])[0])
                if stored_success[:, episode].any() else -1,
                "current_first_success_time": int(np.flatnonzero(
                    current_success[:, episode])[0] + 1)
                if current_success[:, episode].any() else -1,
                "current_final_success": bool(current_success[-1, episode]),
                "current_final_is_obj_placed": bool(current_placed[episode]),
                "current_final_is_robot_static": bool(current_static[episode]),
                "current_final_correct_target_distance": float(
                    current_distance[episode]),
                "dataset_done_at_60": bool(stored_done[-1, episode]),
                "current_terminated_at_60": bool(current_terminated[episode]),
                "current_truncated_at_60": bool(current_truncated[episode]),
                "reward_comparison": array_comparison(
                    stored_reward[:, episode], current_reward[:, episode]),
                "success_sequence_exact": bool(np.array_equal(
                    stored_success[:, episode], current_success[:, episode])),
            })

        mikasa_path = Path(mikasa_robo_suite.__file__).resolve().parent
        result = {
            "diagnostics_schema_version": 1,
            "purpose": "current-environment replay of stored RC3 expert actions",
            "dataset_dir": str(dataset_dir),
            "dataset_files": [f"train_data_{idx}.npz"
                              for idx in range(args.episodes)],
            "source_batch_size": args.source_batch_size,
            "replayed_episodes": args.episodes,
            "reset_seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
            "compare_frames": args.compare_frames,
            "mikasa_path": str(mikasa_path),
            "mikasa_git_revision": _git_revision(mikasa_path),
            "mikasa_package_version": _package_version("mikasa-robo-suite"),
            "maniskill_version": _package_version("mani-skill"),
            "torch_version": torch.__version__,
            "dataset_success_rate": float(stored_success.any(axis=0).mean()),
            "current_env_expert_replay_success_rate": float(success_once.mean()),
            "cue_gt_match_rate": float(np.mean([
                item["cue_matches_current_ground_truth"] for item in per_episode])),
            "frame0_rgb_exact_rate": float(np.mean([
                item["frame_comparisons"][0]["rgb"]["exact"]
                for item in per_episode])),
            "frame0_joints_exact_rate": float(np.mean([
                item["frame_comparisons"][0]["joints"]["exact"]
                for item in per_episode])),
            "episodes": per_episode,
        }
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(result), handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps({
            "output": str(output),
            "dataset_success_rate": result["dataset_success_rate"],
            "current_env_expert_replay_success_rate": (
                result["current_env_expert_replay_success_rate"]),
            "cue_gt_match_rate": result["cue_gt_match_rate"],
            "frame0_rgb_exact_rate": result["frame0_rgb_exact_rate"],
            "frame0_joints_exact_rate": result["frame0_joints_exact_rate"],
        }, indent=2), flush=True)
    finally:
        env_state.close()
        env_rgb.close()


if __name__ == "__main__":
    main()
