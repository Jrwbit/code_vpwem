#!/usr/bin/env python3
"""Collect 1,000 successful RC3 demonstrations in the current environment.

This follows the released MIKASA collector protocol: a state-observation PPO
oracle drives a separately rendered RGB environment with the same reset seed
and actions.  Unlike the released collector, failed (or too-late) rollouts are
rejected so every consecutively numbered file is usable by the VPWEM dataset
loader.  The old dataset is never read or modified.
"""

from __future__ import annotations

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
from baselines.ppo.ppo_memtasks import AgentStateOnly, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info


ENV_NAME = "RememberColor3-v0"
REQUIRED_KEYS = ("rgb", "joints", "action", "reward", "success", "done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=60)
    parser.add_argument(
        "--post-success-steps",
        type=int,
        default=8,
        help="Required tail used by LongMemMikasaDataset after first success.",
    )
    parser.add_argument("--max-source-batches", type=int, default=20)
    return parser.parse_args()


def numpy_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
    return np.asarray(value)


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def write_json_atomic(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def write_npz_atomic(path: Path, payload):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, path)


def wrap_env(raw_env, batch_size: int, *, state: bool):
    wrappers, timeout = env_info(ENV_NAME)
    for wrapper_class, wrapper_kwargs in wrappers:
        raw_env = wrapper_class(raw_env, **wrapper_kwargs)
    env = FlattenRGBDObservationWrapper(
        raw_env,
        rgb=not state,
        depth=False,
        state=state,
        oracle=False,
        joints=not state,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(
        env, batch_size, ignore_terminations=True, record_metrics=True
    ), int(timeout)


def make_env_pair(batch_size: int):
    common = dict(
        num_envs=batch_size,
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )
    # Preserve the released collector's creation and wrapping order.
    raw_state = gym.make(ENV_NAME, obs_mode="state", **common)
    raw_rgb = gym.make(ENV_NAME, obs_mode="rgb", **common)
    state_env, state_timeout = wrap_env(raw_state, batch_size, state=True)
    rgb_env, rgb_timeout = wrap_env(raw_rgb, batch_size, state=False)
    if state_timeout != rgb_timeout:
        raise RuntimeError("state/RGB registered horizons disagree")
    return state_env, rgb_env, state_timeout


def allocate_batch(obs_rgb, max_steps: int, batch_size: int, action_dim: int):
    rgb = numpy_value(obs_rgb["rgb"])
    joints = numpy_value(obs_rgb["joints"])
    return {
        "rgb": np.empty((max_steps, *rgb.shape), dtype=rgb.dtype),
        "joints": np.empty((max_steps, *joints.shape), dtype=joints.dtype),
        "action": np.empty(
            (max_steps, batch_size, action_dim), dtype=np.float32
        ),
        "reward": np.empty((max_steps, batch_size), dtype=np.float32),
        "success": np.empty((max_steps, batch_size), dtype=np.int64),
        "done": np.empty((max_steps, batch_size), dtype=np.int64),
    }


def audit_episode(path: Path, max_steps: int, post_success_steps: int):
    with np.load(path, allow_pickle=False) as episode:
        if tuple(episode.files) != REQUIRED_KEYS:
            raise RuntimeError(
                f"{path}: keys {episode.files} != {list(REQUIRED_KEYS)}"
            )
        for key in REQUIRED_KEYS:
            if len(episode[key]) != max_steps:
                raise RuntimeError(
                    f"{path}: {key} length {len(episode[key])} != {max_steps}"
                )
        success_indices = np.flatnonzero(episode["success"])
        if not len(success_indices):
            raise RuntimeError(f"{path}: retained episode has no success")
        if int(success_indices[0]) + 1 + post_success_steps > max_steps:
            raise RuntimeError(f"{path}: insufficient tail after first success")
        if episode["rgb"].shape[1:] != (128, 128, 6):
            raise RuntimeError(f"{path}: unexpected RGB shape {episode['rgb'].shape}")
        if episode["joints"].shape[1:] != (25,):
            raise RuntimeError(
                f"{path}: unexpected joints shape {episode['joints'].shape}"
            )
        if episode["action"].shape[1:] != (8,):
            raise RuntimeError(
                f"{path}: unexpected action shape {episode['action'].shape}"
            )
        for key in ("joints", "action", "reward"):
            if not np.isfinite(episode[key]).all():
                raise RuntimeError(f"{path}: non-finite {key}")
        return int(success_indices[0]) + 1


def load_resume(marker_path: Path, args, checkpoint_sha: str):
    if not marker_path.exists():
        return [], args.seed_start, None
    with marker_path.open(encoding="utf-8") as stream:
        marker = json.load(stream)
    expected = {
        "env_name": ENV_NAME,
        "target_successful_episodes": args.episodes,
        "source_batch_size": args.batch_size,
        "max_episode_steps": args.max_episode_steps,
        "post_success_steps": args.post_success_steps,
        "oracle_checkpoint_sha256": checkpoint_sha,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(
                f"incompatible resume marker field {key}: "
                f"{marker.get(key)!r} != {value!r}"
            )
    records = marker.get("episodes", [])
    next_seed = args.seed_start
    if records:
        next_seed = max(int(row["source_batch_seed"]) for row in records) + 1
    return records, next_seed, marker


def main():
    args = parse_args()
    if args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("episodes and batch-size must be positive")
    if args.max_episode_steps != 60:
        raise ValueError("RememberColor3 official horizon is 60")
    if not 0 <= args.post_success_steps < args.max_episode_steps:
        raise ValueError("invalid post-success tail")
    if args.max_source_batches <= 0:
        raise ValueError("max-source-batches must be positive")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker_path = output / "RC3_CURRENT_ORACLE_DATASET.json"
    checkpoint_sha = file_sha256(checkpoint)
    records, source_seed, prior_marker = load_resume(
        marker_path, args, checkpoint_sha
    )
    if len(records) > args.episodes:
        raise RuntimeError("resume marker contains too many episodes")
    for episode_id in range(len(records)):
        path = output / f"train_data_{episode_id}.npz"
        if not path.is_file():
            raise RuntimeError(f"resume marker references missing {path}")
        audit_episode(path, args.max_episode_steps, args.post_success_steps)
    unexpected = output / f"train_data_{len(records)}.npz"
    if unexpected.exists():
        raise RuntimeError(f"untracked output file blocks safe resume: {unexpected}")

    mikasa_path = Path(mikasa_robo_suite.__file__).resolve().parent
    marker = {
        "schema_version": 1,
        "complete": False,
        "env_name": ENV_NAME,
        "observation_provenance": "current_simulator_rgb_closed_loop",
        "action_provenance": "released_mikasa_state_oracle_closed_loop",
        "selection": "success_once_and_full_8_step_training_tail",
        "target_successful_episodes": args.episodes,
        "source_batch_size": args.batch_size,
        "seed_start": args.seed_start,
        "max_episode_steps": args.max_episode_steps,
        "post_success_steps": args.post_success_steps,
        "oracle_checkpoint": str(checkpoint),
        "oracle_checkpoint_sha256": checkpoint_sha,
        "collector_sha256": file_sha256(Path(__file__).resolve()),
        "mikasa_path": str(mikasa_path),
        "mikasa_git_revision": git_revision(mikasa_path),
        "mikasa_package_version": package_version("mikasa-robo-suite"),
        "gymnasium_version": package_version("gymnasium"),
        "maniskill_version": package_version("mani-skill"),
        "torch_version": torch.__version__,
        "source_rollouts": int(
            prior_marker.get("source_rollouts", 0) if prior_marker else 0
        ),
        "source_successes": int(
            prior_marker.get("source_successes", 0) if prior_marker else 0
        ),
        "rejected_failures": int(
            prior_marker.get("rejected_failures", 0) if prior_marker else 0
        ),
        "rejected_late_successes": int(
            prior_marker.get("rejected_late_successes", 0)
            if prior_marker else 0
        ),
        "episodes": records,
    }
    # Recover counters exactly from prior source-batch summaries when resuming.
    prior_batches = prior_marker.get("source_batches", []) if prior_marker else []
    marker["source_batches"] = prior_batches
    write_json_atomic(marker_path, marker)

    state_env, rgb_env, registered_timeout = make_env_pair(args.batch_size)
    if registered_timeout != args.max_episode_steps:
        state_env.close()
        rgb_env.close()
        raise RuntimeError(
            f"registered horizon {registered_timeout} != {args.max_episode_steps}"
        )
    device = state_env.base_env.device
    agent = AgentStateOnly(state_env).to(device)
    checkpoint_state = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    agent.load_state_dict(checkpoint_state, strict=True)
    agent.eval()
    action_dim = int(np.prod(state_env.single_action_space.shape))
    batches_this_run = 0

    try:
        while len(records) < args.episodes:
            if batches_this_run >= args.max_source_batches:
                raise RuntimeError(
                    "max-source-batches exhausted before collecting target"
                )
            obs_state, state_reset_info = state_env.reset(seed=source_seed)
            obs_rgb, rgb_reset_info = rgb_env.reset(seed=source_seed)
            state_gt = numpy_value(state_reset_info["oracle_info"]).reshape(
                args.batch_size, -1
            )[:, 0].astype(np.int64)
            rgb_gt = numpy_value(rgb_reset_info["oracle_info"]).reshape(
                args.batch_size, -1
            )[:, 0].astype(np.int64)
            if not np.array_equal(state_gt, rgb_gt):
                raise RuntimeError("state/RGB reset branches desynchronized")

            arrays = allocate_batch(
                obs_rgb, args.max_episode_steps, args.batch_size, action_dim
            )
            placed_and_static_once = np.zeros(args.batch_size, dtype=bool)
            for step in range(args.max_episode_steps):
                arrays["rgb"][step] = numpy_value(obs_rgb["rgb"])
                arrays["joints"][step] = numpy_value(obs_rgb["joints"])
                with torch.no_grad():
                    action = agent.get_action(obs_state, deterministic=True)
                obs_state, state_reward, state_terminated, state_truncated, state_info = (
                    state_env.step(action)
                )
                obs_rgb, reward, terminated, truncated, info = rgb_env.step(action)
                action_np = numpy_value(action)
                reward_np = numpy_value(reward)
                success_np = numpy_value(info["success"]).astype(bool)
                arrays["action"][step] = action_np
                arrays["reward"][step] = reward_np
                arrays["success"][step] = success_np.astype(np.int64)
                arrays["done"][step] = numpy_value(
                    torch.logical_or(terminated, truncated)
                ).astype(np.int64)
                placed_and_static_once |= (
                    numpy_value(info["is_obj_placed"]).astype(bool)
                    & numpy_value(info["is_robot_static"]).astype(bool)
                )

                state_success = numpy_value(state_info["success"]).astype(bool)
                if not np.array_equal(state_success, success_np):
                    raise RuntimeError(
                        f"state/RGB success desynchronized at step {step}"
                    )
                if not np.allclose(
                    numpy_value(state_reward), reward_np, atol=1e-5, rtol=1e-5
                ):
                    raise RuntimeError(
                        f"state/RGB reward desynchronized at step {step}"
                    )

            first_success = np.full(args.batch_size, -1, dtype=np.int32)
            for slot in range(args.batch_size):
                indices = np.flatnonzero(arrays["success"][:, slot])
                if len(indices):
                    first_success[slot] = int(indices[0]) + 1
            success_mask = first_success > 0
            full_tail_mask = (
                success_mask
                & (first_success + args.post_success_steps <= args.max_episode_steps)
            )
            selected_slots = np.flatnonzero(full_tail_mask)
            remaining = args.episodes - len(records)
            selected_slots = selected_slots[:remaining]

            first_output_id = len(records)
            for slot in selected_slots:
                episode_id = len(records)
                payload = {
                    key: np.ascontiguousarray(value[:, slot])
                    for key, value in arrays.items()
                }
                path = output / f"train_data_{episode_id}.npz"
                write_npz_atomic(path, payload)
                records.append(
                    {
                        "episode": episode_id,
                        "source_batch_seed": source_seed,
                        "source_vector_index": int(slot),
                        "ground_truth_branch": int(rgb_gt[slot]),
                        "first_success_time": int(first_success[slot]),
                        "placed_and_static_once": bool(
                            placed_and_static_once[slot]
                        ),
                    }
                )

            batch_summary = {
                "source_batch_seed": source_seed,
                "rollouts": args.batch_size,
                "successes": int(success_mask.sum()),
                "late_successes": int((success_mask & ~full_tail_mask).sum()),
                "accepted": int(len(selected_slots)),
                "first_output_episode": first_output_id,
                "last_output_episode": len(records) - 1,
                "ground_truth_counts": np.bincount(
                    rgb_gt, minlength=3
                ).astype(int).tolist(),
            }
            marker["source_batches"].append(batch_summary)
            marker["source_rollouts"] += args.batch_size
            marker["source_successes"] += int(success_mask.sum())
            marker["rejected_failures"] += int((~success_mask).sum())
            marker["rejected_late_successes"] += int(
                (success_mask & ~full_tail_mask).sum()
            )
            marker["episodes"] = records
            write_json_atomic(marker_path, marker)
            print(
                f"seed={source_seed} success={int(success_mask.sum())}/"
                f"{args.batch_size} accepted={len(selected_slots)} "
                f"total={len(records)}/{args.episodes}",
                flush=True,
            )
            source_seed += 1
            batches_this_run += 1
    finally:
        state_env.close()
        rgb_env.close()

    success_times = []
    total_bytes = 0
    for episode_id in range(args.episodes):
        path = output / f"train_data_{episode_id}.npz"
        success_times.append(
            audit_episode(path, args.max_episode_steps, args.post_success_steps)
        )
        total_bytes += path.stat().st_size
    if (output / f"train_data_{args.episodes}.npz").exists():
        raise RuntimeError("unexpected extra numbered episode")

    labels = np.asarray(
        [row["ground_truth_branch"] for row in records], dtype=np.int8
    )
    np.save(output / "branch_labels.npy", labels)
    marker["complete"] = True
    marker["successful_episodes"] = args.episodes
    marker["branch_counts"] = np.bincount(labels, minlength=3).astype(int).tolist()
    marker["mean_first_success_time"] = float(np.mean(success_times))
    marker["dataset_bytes"] = total_bytes
    write_json_atomic(marker_path, marker)
    print(json.dumps({
        "output": str(output),
        "successful_episodes": args.episodes,
        "branch_counts": marker["branch_counts"],
        "mean_first_success_time": marker["mean_first_success_time"],
        "dataset_bytes": total_bytes,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
