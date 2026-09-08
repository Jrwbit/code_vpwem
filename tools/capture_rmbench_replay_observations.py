#!/usr/bin/env python3
"""Capture live observations along an exact RMBench demonstration replay.

The resulting NPZ is intentionally policy-independent.  It pairs current-live
RGB/qpos with the stored demonstration RGB/qpos at requested frame indices, so
checkpoint diagnostics can be run later without keeping the renderer alive.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def _decode_rgb(value) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(bytes(value))).convert("RGB"), dtype=np.uint8)


def _checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench-root", type=Path, default=Path("/home/jrw/code_rmbench"))
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.rmbench_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "envs" / "curobo" / "src"))
    sys.path.insert(0, str(root / "policy"))
    sys.path.insert(0, str(root / "description" / "utils"))

    # Reuse the already-audited environment construction from the expert replay
    # diagnostic instead of duplicating RMBench's setup details here.
    tools_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(tools_dir))
    from replay_rmbench_demo_actions import _build_task_args
    from envs.swap_blocks import swap_blocks

    requested = np.asarray(sorted(set(args.frames)), dtype=np.int64)
    if requested[0] < 0:
        raise ValueError("Frame indices must be non-negative")

    with h5py.File(args.hdf5, "r") as source:
        qpos = np.asarray(source["joint_action/vector"][:], dtype=np.float64)
        if requested[-1] >= len(qpos):
            raise IndexError(f"Requested frame {requested[-1]} but demonstration has {len(qpos)} frames")
        stored_rgb = np.stack(
            [_decode_rgb(source["observation/head_camera/rgb"][int(index)]) for index in requested]
        )
        stored_qpos = qpos[requested]
        stored_camera = {
            key: np.asarray(source[f"observation/head_camera/{key}"][requested])
            for key in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl")
            if f"observation/head_camera/{key}" in source
        }

    task = swap_blocks()
    started = time.time()
    summary = {
        "diagnostic": "rmbench_live_observation_capture_during_stored_action_replay",
        "status": "running",
        "rmbench_root": str(root),
        "hdf5": str(args.hdf5.resolve()),
        "episode": args.episode,
        "seed": args.seed,
        "task_config": args.task_config,
        "requested_frames": requested.tolist(),
    }
    _checkpoint(summary, args.output_json)
    live_rgb: list[np.ndarray] = []
    live_qpos: list[np.ndarray] = []
    tracking_error: list[float] = []
    wanted = {int(frame): position for position, frame in enumerate(requested)}
    captured: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    captured_camera: dict[str, dict[int, np.ndarray]] = {
        key: {} for key in stored_camera
    }
    try:
        task_args = _build_task_args(root, args.task_config)
        task.setup_demo(now_ep_num=args.episode, seed=args.seed, is_test=True, **task_args)

        for frame in range(int(requested[-1]) + 1):
            if frame > 0:
                task.take_action(qpos[frame], action_type="qpos")
            observation = task.get_obs()
            current_qpos = np.asarray(observation["joint_action"]["vector"], dtype=np.float64)
            tracking_error.append(float(np.max(np.abs(current_qpos - qpos[frame]))))
            if frame in wanted:
                head_camera = observation["observation"]["head_camera"]
                current_rgb = np.asarray(
                    head_camera["rgb"], dtype=np.uint8
                ).copy()
                captured[frame] = (current_rgb, current_qpos.copy())
                for key in captured_camera:
                    if key in head_camera:
                        captured_camera[key][frame] = np.asarray(head_camera[key]).copy()
                print(
                    f"[capture] frame={frame} "
                    f"qpos_max_error={tracking_error[-1]:.6g}",
                    flush=True,
                )

        for frame in requested:
            rgb, joints = captured[int(frame)]
            live_rgb.append(rgb)
            live_qpos.append(joints)

        live_rgb_array = np.stack(live_rgb)
        live_qpos_array = np.stack(live_qpos)
        live_camera = {
            key: np.stack([values[int(frame)] for frame in requested])
            for key, values in captured_camera.items()
            if len(values) == len(requested)
        }
        rgb_error = np.abs(live_rgb_array.astype(np.int16) - stored_rgb.astype(np.int16))
        qpos_error = np.abs(live_qpos_array - stored_qpos)
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "frame_indices": requested,
            "live_rgb": live_rgb_array,
            "live_qpos": live_qpos_array,
            "stored_rgb": stored_rgb,
            "stored_qpos": stored_qpos,
            "demonstration_qpos": qpos,
        }
        for key, value in stored_camera.items():
            arrays[f"stored_{key}"] = value
        for key, value in live_camera.items():
            arrays[f"live_{key}"] = value
        np.savez_compressed(args.output_npz, **arrays)
        camera_comparison = {}
        for key, stored_value in stored_camera.items():
            if key not in live_camera:
                camera_comparison[key] = {"present_live": False}
                continue
            error = np.abs(live_camera[key].astype(np.float64) - stored_value.astype(np.float64))
            camera_comparison[key] = {
                "present_live": True,
                "exact": bool(np.array_equal(live_camera[key], stored_value)),
                "mean_abs_error": float(np.mean(error)),
                "max_abs_error": float(np.max(error)),
            }
        summary.update(
            status="complete",
            output_npz=str(args.output_npz.resolve()),
            live_rgb_shape=list(live_rgb_array.shape),
            qpos_max_abs_error_at_captures=float(np.max(qpos_error)),
            qpos_mean_abs_error_at_captures=float(np.mean(qpos_error)),
            replay_tracking_max_abs_error=float(np.max(tracking_error)),
            replay_tracking_mean_abs_error=float(np.mean(tracking_error)),
            live_vs_stored_rgb_mean_abs_error=float(np.mean(rgb_error)),
            live_vs_stored_rgb_max_abs_error=int(np.max(rgb_error)),
            camera_matrices_live_vs_stored=camera_comparison,
            per_frame=[
                {
                    "frame": int(frame),
                    "qpos_max_abs_error": float(np.max(qpos_error[position])),
                    "rgb_mean_abs_error": float(np.mean(rgb_error[position])),
                    "rgb_max_abs_error": int(np.max(rgb_error[position])),
                }
                for position, frame in enumerate(requested)
            ],
        )
    except Exception as error:
        summary.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        summary["elapsed_seconds"] = float(time.time() - started)
        try:
            task.close_env()
        except Exception as error:
            summary["close_error"] = f"{type(error).__name__}: {error}"
        _checkpoint(summary, args.output_json)


if __name__ == "__main__":
    main()
