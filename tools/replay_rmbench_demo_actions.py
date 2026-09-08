#!/usr/bin/env python3
"""Replay RMBench demonstration qpos actions in the current simulator.

This is a parity diagnostic, not a policy evaluation.  Each requested episode
is reset with the seed stored next to the official demonstration and receives
the recorded joint states as absolute qpos commands.  The default offset of 1
matches DP training: observation/qpos[t] -> action/qpos[t + 1].
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(frame) -> np.ndarray:
    import cv2

    encoded = np.frombuffer(bytes(frame), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Could not decode stored RGB frame")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _pose(actor) -> dict[str, list[float]]:
    pose = actor.get_pose()
    return {
        "p": np.asarray(pose.p, dtype=float).tolist(),
        "q": np.asarray(pose.q, dtype=float).tolist(),
    }


def _robot_qpos(task) -> np.ndarray:
    left = task.robot.get_left_arm_jointState()
    if not task.is_dual_arm:
        return np.asarray(left, dtype=np.float64)
    return np.asarray(left + task.robot.get_right_arm_jointState(), dtype=np.float64)


def _scene_state(task) -> dict:
    state = {
        "robot_qpos": _robot_qpos(task).tolist(),
        "basket_poses": [_pose(actor) for actor in task.baskets],
        "block1_pose": _pose(task.block1),
        "block2_pose": _pose(task.block2),
        "button_pose": _pose(task.button.actor if hasattr(task.button, "actor") else task.button),
        "with_block_basket_idx_1": int(task.with_block_basket_idx_1),
        "with_block_basket_idx_2": int(task.with_block_basket_idx_2),
        "empty_basket_idx": int(task.empty_basket_idx),
        "press_cnt": int(task.press_cnt),
        "press_flag": bool(task.press_flag),
        "flag": bool(task.flag),
    }
    return state


def _build_task_args(rmbench_root: Path, task_config: str) -> dict:
    from envs import CONFIGS_PATH

    with (rmbench_root / "task_config" / f"{task_config}.yml").open(
        "r", encoding="utf-8"
    ) as stream:
        args = yaml.safe_load(stream)
    args.update(
        task_name="swap_blocks",
        task_config=task_config,
        eval_mode=True,
        eval_video_log=False,
        render_freq=0,
        save_data=False,
        need_plan=False,
    )

    with Path(CONFIGS_PATH, "_embodiment_config.yml").open(
        "r", encoding="utf-8"
    ) as stream:
        embodiments = yaml.safe_load(stream)
    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise NotImplementedError("This diagnostic currently expects the dual-arm RMBench embodiment")
    robot_file = embodiments[embodiment[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["dual_arm"] = True
    with (rmbench_root / robot_file / "config.yml").resolve().open(
        "r", encoding="utf-8"
    ) as stream:
        embodiment_config = yaml.safe_load(stream)
    args["left_embodiment_config"] = embodiment_config
    args["right_embodiment_config"] = embodiment_config

    with Path(CONFIGS_PATH, "_camera_config.yml").open("r", encoding="utf-8") as stream:
        cameras = yaml.safe_load(stream)
    camera = cameras[args["camera"]["head_camera_type"]]
    args["head_camera_h"] = camera["h"]
    args["head_camera_w"] = camera["w"]
    return args


def _init_comparison(task, hdf5: h5py.File) -> dict:
    observation = task.get_obs()
    current_qpos = np.asarray(observation["joint_action"]["vector"], dtype=np.float64)
    stored_qpos = np.asarray(hdf5["joint_action/vector"][0], dtype=np.float64)
    current_rgb = np.asarray(observation["observation"]["head_camera"]["rgb"])
    stored_rgb = _load_rgb(hdf5["observation/head_camera/rgb"][0])
    rgb_delta = current_rgb.astype(np.int16) - stored_rgb.astype(np.int16)
    return {
        "qpos_max_abs_error": float(np.max(np.abs(current_qpos - stored_qpos))),
        "qpos_l2_error": float(np.linalg.norm(current_qpos - stored_qpos)),
        "qpos_current": current_qpos.tolist(),
        "qpos_stored": stored_qpos.tolist(),
        "head_rgb_shape_current": list(current_rgb.shape),
        "head_rgb_shape_stored": list(stored_rgb.shape),
        "head_rgb_exact": bool(np.array_equal(current_rgb, stored_rgb)),
        "head_rgb_mean_abs_error": float(np.mean(np.abs(rgb_delta))),
        "head_rgb_max_abs_error": int(np.max(np.abs(rgb_delta))),
    }


def _checkpoint(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def replay_episode(task_class, task_args: dict, episode: int, seed: int, hdf5_path: Path, action_offset: int) -> dict:
    task = task_class()
    started = time.time()
    result = {
        "episode": int(episode),
        "seed": int(seed),
        "action_offset": int(action_offset),
        "hdf5_path": str(hdf5_path),
        "hdf5_sha256": _sha256(hdf5_path),
        "status": "running",
    }
    try:
        task.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **task_args)
        with h5py.File(hdf5_path, "r") as hdf5:
            actions = np.asarray(hdf5["joint_action/vector"][:], dtype=np.float64)
            result["stored_frames"] = int(actions.shape[0])
            result["initial"] = _init_comparison(task, hdf5)
            result["initial_scene"] = _scene_state(task)

            errors = []
            executed = 0
            for target_index in range(action_offset, len(actions)):
                task.take_action(actions[target_index], action_type="qpos")
                executed += 1
                current = _robot_qpos(task)
                errors.append(float(np.max(np.abs(current - actions[target_index]))))
                if task.eval_success:
                    break

            result["actions_executed"] = executed
            result["last_target_index"] = int(action_offset + executed - 1)
            result["tracking_max_abs_error_max"] = float(max(errors, default=0.0))
            result["tracking_max_abs_error_mean"] = float(np.mean(errors)) if errors else 0.0
            result["eval_success"] = bool(task.eval_success)
            result["check_success"] = bool(task.check_success())
            result["max_reward"] = float(task.max_reward)
            result["final_scene"] = _scene_state(task)
            result["status"] = "complete"
    except Exception as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    finally:
        result["elapsed_seconds"] = float(time.time() - started)
        try:
            task.close_env()
        except Exception as close_error:
            result["close_error"] = f"{type(close_error).__name__}: {close_error}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench-root", type=Path, default=Path("/home/jrw/code_rmbench"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--action-offset", type=int, default=1)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rmbench_root = args.rmbench_root.resolve()
    dataset_root = args.dataset_root.resolve()
    os.chdir(rmbench_root)
    sys.path.insert(0, str(rmbench_root))
    sys.path.insert(0, str(rmbench_root / "policy"))
    sys.path.insert(0, str(rmbench_root / "description" / "utils"))

    task_module = importlib.import_module("envs.swap_blocks")
    task_class = getattr(task_module, "swap_blocks")
    task_args = _build_task_args(rmbench_root, args.task_config)
    seeds = [int(value) for value in (dataset_root / "seed.txt").read_text().split()]
    summary = {
        "diagnostic": "rmbench_stored_expert_action_replay",
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "rmbench_root": str(rmbench_root),
        "dataset_root": str(dataset_root),
        "action_semantics": "observation/qpos[t] -> command/qpos[t + action_offset]",
        "episodes": [],
    }
    _checkpoint(summary, args.output)
    for episode in args.episodes:
        if episode < 0 or episode >= len(seeds):
            raise IndexError(f"episode {episode} outside seed list of length {len(seeds)}")
        path = dataset_root / "data" / f"episode{episode}.hdf5"
        print(f"[replay] episode={episode} seed={seeds[episode]} path={path}", flush=True)
        result = replay_episode(
            task_class=task_class,
            task_args=task_args,
            episode=episode,
            seed=seeds[episode],
            hdf5_path=path,
            action_offset=args.action_offset,
        )
        summary["episodes"].append(result)
        _checkpoint(summary, args.output)
        print(
            f"[replay] episode={episode} status={result['status']} "
            f"success={result.get('check_success')} reward={result.get('max_reward')} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
