#!/usr/bin/env python3
"""Audit an RMBench HDF5 directory against a converted replay zarr."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
import zarr
from PIL import Image


def _decode_rgb(value) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(bytes(value))).convert("RGB"), dtype=np.uint8)


def _episode_slices(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.r_[0, episode_ends[:-1]]
    return [(int(start), int(end)) for start, end in zip(starts, episode_ends)]


def _lag_stats(qpos: np.ndarray, action: np.ndarray, slices, lag: int) -> dict:
    left = []
    right = []
    for start, end in slices:
        if lag >= 0:
            left.append(qpos[start : end - lag] if lag else qpos[start:end])
            right.append(action[start + lag : end] if lag else action[start:end])
        else:
            offset = -lag
            left.append(qpos[start + offset : end])
            right.append(action[start : end - offset])
    left = np.concatenate(left)
    right = np.concatenate(right)
    error = left - right
    return {
        "lag_action_index_minus_qpos_index": lag,
        "mse": float(np.mean(error**2)),
        "max_abs_error": float(np.max(np.abs(error))),
        "exact_row_fraction": float(np.mean(np.all(left == right, axis=1))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay = zarr.open(str(args.zarr), mode="r")
    episode_ends = np.asarray(replay["meta/episode_ends"][:], dtype=np.int64)
    slices = _episode_slices(episode_ends)
    qpos = np.asarray(replay["data/agent_pos"][:], dtype=np.float32)
    action = np.asarray(replay["data/action"][:], dtype=np.float32)

    # The converter enumerates all files with ``sorted(glob(...))``.  Rebuild
    # that full lexicographic order even when this audit downloaded only a
    # small reference subset.
    source_names = sorted(f"episode{episode}.hdf5" for episode in range(len(slices)))
    result = {
        "zarr": str(args.zarr.resolve()),
        "hdf5_root": str(args.hdf5_root.resolve()),
        "zarr_episode_count": len(slices),
        "zarr_transition_count": int(len(action)),
        "source_ordering": "Python lexicographic Path sort",
        "lag_stats": [_lag_stats(qpos, action, slices, lag) for lag in range(-3, 4)],
        "episodes": [],
    }

    for episode in args.episodes:
        filename = f"episode{episode}.hdf5"
        if not (args.hdf5_root / filename).is_file():
            raise FileNotFoundError(args.hdf5_root / filename)
        zarr_episode = source_names.index(filename)
        start, end = slices[zarr_episode]
        with h5py.File(args.hdf5_root / filename, "r") as source:
            source_qpos = np.asarray(source["joint_action/vector"][:], dtype=np.float32)
            qpos_equal = np.array_equal(qpos[start:end], source_qpos[:-1])
            action_equal = np.array_equal(action[start:end], source_qpos[1:])
            frame_checks = []
            for frame in sorted({0, (end - start) // 2, end - start - 1}):
                source_rgb = _decode_rgb(source["observation/head_camera/rgb"][frame])
                replay_rgb = np.asarray(replay["data/head_cam"][start + frame])
                frame_checks.append(
                    {
                        "frame": frame,
                        "exact": bool(np.array_equal(source_rgb, replay_rgb)),
                        "mean_abs_error": float(
                            np.mean(np.abs(source_rgb.astype(np.int16) - replay_rgb.astype(np.int16)))
                        ),
                    }
                )
        result["episodes"].append(
            {
                "numeric_episode": episode,
                "filename": filename,
                "zarr_episode_index": zarr_episode,
                "source_frames": int(len(source_qpos)),
                "zarr_transitions": end - start,
                "agent_pos_equals_source_qpos_without_last": bool(qpos_equal),
                "action_equals_source_qpos_without_first": bool(action_equal),
                "head_rgb_checks": frame_checks,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
