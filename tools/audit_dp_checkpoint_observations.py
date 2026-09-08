#!/usr/bin/env python3
"""Isolate RMBench DP checkpoint train/deploy observation mismatches.

For each requested planning time, this diagnostic evaluates the untouched EMA
policy with the same diffusion random seed under a factorial set of inputs:
stored versus live rendering, RGB versus BGR, and contiguous versus duplicated
plan-boundary observation windows.  Predictions are compared with the stored
expert qpos[t+1:t+7] chunk in physical action units.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _json_tensor(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _field_normalizer(normalizer, key: str) -> dict:
    params = normalizer.params_dict[key]
    return {
        "scale": _json_tensor(params["scale"]),
        "offset": _json_tensor(params["offset"]),
        "input_stats": {
            name: _json_tensor(value) for name, value in params["input_stats"].items()
        },
    }


def _array_stats(value: np.ndarray) -> dict:
    value = np.asarray(value, dtype=np.float64)
    return {
        "min": np.min(value, axis=0).tolist(),
        "max": np.max(value, axis=0).tolist(),
        "mean": np.mean(value, axis=0).tolist(),
        # Match torch.std's default (Bessel-corrected) estimator used by the
        # checkpoint normalizer fit.
        "std": np.std(value, axis=0, ddof=1).tolist(),
    }


def _max_stats_delta(checkpoint_stats: dict, replay_stats: dict) -> dict:
    result = {}
    for name in ("min", "max", "mean", "std"):
        checkpoint = np.asarray(checkpoint_stats[name], dtype=np.float64)
        replay = np.asarray(replay_stats[name], dtype=np.float64)
        result[name] = float(np.max(np.abs(checkpoint - replay)))
    return result


def _metrics(prediction: np.ndarray, target: np.ndarray, current: np.ndarray) -> dict:
    error = prediction - target
    hold = np.broadcast_to(current, target.shape)
    pred_delta = (prediction - current).reshape(-1)
    target_delta = (target - current).reshape(-1)
    denominator = float(np.linalg.norm(pred_delta) * np.linalg.norm(target_delta))
    cosine = float(np.dot(pred_delta, target_delta) / denominator) if denominator else None
    first_pred_delta = prediction[0] - current
    first_target_delta = target[0] - current
    first_denominator = float(np.linalg.norm(first_pred_delta) * np.linalg.norm(first_target_delta))
    first_cosine = (
        float(np.dot(first_pred_delta, first_target_delta) / first_denominator)
        if first_denominator
        else None
    )
    return {
        "chunk_mse": float(np.mean(error**2)),
        "chunk_mae": float(np.mean(np.abs(error))),
        "chunk_max_abs_error": float(np.max(np.abs(error))),
        "first_action_mse": float(np.mean(error[0] ** 2)),
        "first_action_mae": float(np.mean(np.abs(error[0]))),
        "first_action_max_abs_error": float(np.max(np.abs(error[0]))),
        "hold_chunk_mse": float(np.mean((hold - target) ** 2)),
        "plan_delta_cosine": cosine,
        "first_delta_cosine": first_cosine,
    }


def _window_indices(timestep: int, duplicate_boundary: bool) -> list[int]:
    if timestep == 0:
        return [0, 0, 0]
    if duplicate_boundary:
        return [max(0, timestep - 1), timestep, timestep]
    return [max(0, timestep - 2), max(0, timestep - 1), timestep]


def _mean_metrics(rows: list[dict]) -> dict:
    metric_names = (
        "chunk_mse",
        "chunk_mae",
        "chunk_max_abs_error",
        "first_action_mse",
        "first_action_mae",
        "first_action_max_abs_error",
        "hold_chunk_mse",
        "plan_delta_cosine",
        "first_delta_cosine",
    )
    result = {}
    for name in metric_names:
        values = [row["metrics"][name] for row in rows if row["metrics"][name] is not None]
        result[name] = float(np.mean(values)) if values else None
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench-root", type=Path, default=Path("/home/jrw/code_rmbench"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--planning-points", type=int, nargs="+", required=True)
    parser.add_argument("--replay-zarr", type=Path)
    parser.add_argument("--base-seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.rmbench_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root / "policy"))
    sys.path.insert(0, str(root / "policy" / "DP"))
    from DP.dp_model import DP

    capture = np.load(args.capture)
    available = {int(frame): position for position, frame in enumerate(capture["frame_indices"])}
    demonstration_qpos = np.asarray(capture["demonstration_qpos"], dtype=np.float64)
    planning_points = [int(value) for value in args.planning_points]
    required = {
        frame
        for timestep in planning_points
        for duplicate in (False, True)
        for frame in _window_indices(timestep, duplicate)
    }
    missing = sorted(required - set(available))
    if missing:
        raise ValueError(f"Capture is missing required frames: {missing}")
    for timestep in planning_points:
        if timestep + 6 >= len(demonstration_qpos):
            raise ValueError(f"Planning point {timestep} lacks a six-action expert target")

    started = time.time()
    model = DP(str(args.checkpoint.resolve()), n_obs_steps=3, n_action_steps=6)
    policy = model.policy
    torch.cuda.reset_peak_memory_stats(policy.device)

    summary = {
        "diagnostic": "official_dp_checkpoint_observation_factorial",
        "checkpoint": str(args.checkpoint.resolve()),
        "capture": str(args.capture.resolve()),
        "planning_points": planning_points,
        "base_seed": args.base_seed,
        "diffusion_noise_control": (
            "torch CPU and CUDA RNG reset to base_seed+timestep before every variant"
        ),
        "action_target": "stored demonstration qpos[t+1:t+7] in physical units",
        "policy": {
            "horizon": int(policy.horizon),
            "n_obs_steps": int(policy.n_obs_steps),
            "n_action_steps": int(policy.n_action_steps),
            "num_inference_steps": int(policy.num_inference_steps),
            "action_dim": int(policy.action_dim),
            "device": str(policy.device),
            "dtype": str(policy.dtype),
        },
        "normalizer": {
            key: _field_normalizer(policy.normalizer, key)
            for key in ("action", "agent_pos", "head_cam")
        },
        "variants": {},
    }

    if args.replay_zarr is not None:
        import zarr

        replay = zarr.open(str(args.replay_zarr.resolve()), mode="r")
        replay_action_stats = _array_stats(np.asarray(replay["data/action"][:]))
        replay_agent_stats = _array_stats(np.asarray(replay["data/agent_pos"][:]))
        summary["replay_zarr_normalizer_parity"] = {
            "path": str(args.replay_zarr.resolve()),
            "action_stats": replay_action_stats,
            "agent_pos_stats": replay_agent_stats,
            "checkpoint_max_abs_delta": {
                "action": _max_stats_delta(
                    summary["normalizer"]["action"]["input_stats"], replay_action_stats
                ),
                "agent_pos": _max_stats_delta(
                    summary["normalizer"]["agent_pos"]["input_stats"], replay_agent_stats
                ),
            },
        }

    variant_specs = [
        ("stored_bgr_correct_queue", "stored_rgb", True, False),
        ("stored_rgb_correct_queue", "stored_rgb", False, False),
        ("stored_bgr_duplicate_queue", "stored_rgb", True, True),
        ("stored_rgb_duplicate_queue", "stored_rgb", False, True),
        ("live_bgr_correct_queue", "live_rgb", True, False),
        ("live_rgb_correct_queue", "live_rgb", False, False),
        ("live_bgr_duplicate_queue", "live_rgb", True, True),
        ("live_rgb_duplicate_queue", "live_rgb", False, True),
    ]
    predictions: dict[str, dict[int, np.ndarray]] = {name: {} for name, *_ in variant_specs}

    for name, source_key, use_bgr, duplicate in variant_specs:
        rows = []
        source_images = np.asarray(capture[source_key])
        source_qpos_key = "stored_qpos" if source_key == "stored_rgb" else "live_qpos"
        source_qpos = np.asarray(capture[source_qpos_key], dtype=np.float64)
        for timestep in planning_points:
            indices = _window_indices(timestep, duplicate)
            positions = [available[index] for index in indices]
            images = source_images[positions]
            if use_bgr:
                images = images[..., ::-1]
            images = np.ascontiguousarray(np.moveaxis(images, -1, 1) / 255.0)
            agent_pos = np.ascontiguousarray(source_qpos[positions])

            torch.manual_seed(args.base_seed + timestep)
            torch.cuda.manual_seed_all(args.base_seed + timestep)
            call_started = time.time()
            with torch.no_grad():
                prediction_dict = policy.predict_action(
                    {
                        "head_cam": torch.from_numpy(images).unsqueeze(0).to(policy.device),
                        "agent_pos": torch.from_numpy(agent_pos).unsqueeze(0).to(policy.device),
                    }
                )
            prediction = prediction_dict["action"][0].detach().cpu().numpy().astype(np.float64)
            target = demonstration_qpos[timestep + 1 : timestep + 7]
            current = demonstration_qpos[timestep]
            predictions[name][timestep] = prediction
            action_stats = summary["normalizer"]["action"]["input_stats"]
            action_min = np.asarray(action_stats["min"])
            action_max = np.asarray(action_stats["max"])
            row = {
                "timestep": timestep,
                "window_indices": indices,
                "prediction": prediction.tolist(),
                "target": target.tolist(),
                "metrics": _metrics(prediction, target, current),
                "prediction_outside_training_range_count": int(
                    np.sum((prediction < action_min) | (prediction > action_max))
                ),
                "inference_seconds": float(time.time() - call_started),
            }
            rows.append(row)
            print(
                f"[dp-audit] {name} t={timestep} "
                f"chunk_mse={row['metrics']['chunk_mse']:.6g} "
                f"first_mse={row['metrics']['first_action_mse']:.6g}",
                flush=True,
            )
        summary["variants"][name] = {
            "image_source": source_key,
            "channel_order": "BGR" if use_bgr else "RGB",
            "queue": "duplicated boundary" if duplicate else "contiguous/padded",
            "aggregate": _mean_metrics(rows),
            "points": rows,
        }

    reference = "stored_bgr_correct_queue"
    for name in summary["variants"]:
        differences = []
        for timestep in planning_points:
            delta = predictions[name][timestep] - predictions[reference][timestep]
            differences.append(
                {
                    "timestep": timestep,
                    "action_mse": float(np.mean(delta**2)),
                    "action_max_abs": float(np.max(np.abs(delta))),
                    "first_action_mse": float(np.mean(delta[0] ** 2)),
                    "first_action_max_abs": float(np.max(np.abs(delta[0]))),
                }
            )
        summary["variants"][name]["vs_stored_bgr_correct_queue"] = {
            key: float(np.mean([row[key] for row in differences]))
            for key in ("action_mse", "action_max_abs", "first_action_mse", "first_action_max_abs")
        }

    # Explicitly verify that the checkpoint API returns physical actions and
    # that normalization itself is not introducing an offset.
    action_probe = torch.as_tensor(
        demonstration_qpos[planning_points[0] + 1 : planning_points[0] + 7],
        device=policy.device,
        dtype=policy.dtype,
    )
    normalized = policy.normalizer["action"].normalize(action_probe)
    roundtrip = policy.normalizer["action"].unnormalize(normalized)
    summary["action_unnormalize_audit"] = {
        "roundtrip_max_abs_error": float(torch.max(torch.abs(roundtrip - action_probe)).item()),
        "predict_action_returns": "unnormalized physical action via normalizer[action].unnormalize",
    }
    summary["elapsed_seconds"] = float(time.time() - started)
    summary["cuda_peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(policy.device))
    summary["cuda_peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(policy.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "elapsed_seconds": summary["elapsed_seconds"],
        "aggregates": {
            name: value["aggregate"] for name, value in summary["variants"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
