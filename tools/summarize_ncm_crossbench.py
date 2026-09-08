#!/usr/bin/env python3
"""Collect NCM cross-benchmark results without mixing diagnostic budgets.

The experiment runners intentionally emit several different result formats.  This
collector keeps them in separate tables so a 5k diagnostic checkpoint cannot be
mistaken for a full-budget result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DIAGNOSTIC_STEPS = {5_000, 10_000, 20_000}
FULL_BUDGETS = {
    "remember_color3": 600_000,
    "remember_color5": 400_000,
    "shell": 600_000,
    "intercept_grab": 400_000,
    "square_mh": 1_000_000,
    "transport_ph": 1_000_000,
    "longsquare_ph": 1_000_000,
}


def scalar(value: Any) -> float:
    try:
        return float(np.asarray(value).item())
    except (TypeError, ValueError):
        return math.nan


def load_npy_dict(path: Path) -> dict[str, Any]:
    value = np.load(path, allow_pickle=True)
    if value.shape == ():
        value = value.item()
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a dict, got {type(value).__name__}")
    return value


def checkpoint_from_eval_dir(path: Path) -> int:
    match = re.fullmatch(r"eval_(\d+)(k)?(?:_chunk\d+)?", path.parent.name)
    if not match:
        raise ValueError(f"cannot parse checkpoint from {path.parent}")
    return int(match.group(1)) * (1_000 if match.group(2) else 1)


def task_from_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    first = relative.parts[0]
    return {
        "remember_color3": "remember_color3",
        "remember_color5": "remember_color5",
        "shell": "shell",
        "intercept_grab": "intercept_grab",
    }.get(first, first)


def budget_class(task: str, checkpoint: int) -> str:
    if checkpoint in DIAGNOSTIC_STEPS:
        return "diagnostic"
    if checkpoint == FULL_BUDGETS.get(task):
        return "full"
    return "intermediate"


def collect_rollouts(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/eval_*/result.npy")):
        checkpoint = checkpoint_from_eval_dir(path)
        task = task_from_path(path, root)
        result = load_npy_dict(path)
        episode_success = result.get("episode_success")
        rows.append(
            {
                "task": task,
                "experiment": str(path.parent.parent.relative_to(root)),
                "checkpoint": checkpoint,
                "budget_class": budget_class(task, checkpoint),
                "episodes": (
                    int(np.asarray(episode_success).size)
                    if episode_success is not None
                    else ""
                ),
                "mean_success": scalar(result.get("mean_success")),
                "mean_reward": scalar(result.get("mean_reward")),
                "eval_seed": result.get("eval_seed", ""),
                "result_path": str(path),
            }
        )
    return rows


def collect_causal(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/eval_*/ncm_eval.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = checkpoint_from_eval_dir(path)
        task = task_from_path(path, root)
        base = {
            "task": task,
            "experiment": str(path.parent.parent.relative_to(root)),
            "checkpoint": checkpoint,
            "budget_class": budget_class(task, checkpoint),
            "success_gap": data.get("success_gap_correct_minus_matched_wrong"),
            "branch_gap": data.get("branch_gap_correct_minus_matched_wrong"),
            "wrong_attraction_uplift": data.get("wrong_attraction_uplift"),
            "first_direction_wrong_uplift": data.get(
                "first_direction_wrong_uplift"
            ),
            "flip_to_wrong_given_correct": data.get(
                "flip_to_wrong_given_correct"
            ),
            "eval_seed": data.get("eval_seed_start", ""),
            "checkpoint_sha256": data.get("checkpoint_sha256", ""),
            "result_path": str(path),
        }
        for mode, metrics in data.get("modes", {}).items():
            rows.append(
                {
                    **base,
                    "mode": mode,
                    "mean_success": metrics.get("mean_success"),
                    "branch_acc": metrics.get("branch_acc"),
                    "branch_acc_macro": metrics.get("branch_acc_macro"),
                    "commit_rate": metrics.get("commit_rate"),
                    "wrong_branch_choice_rate": metrics.get(
                        "wrong_branch_choice_rate"
                    ),
                    "first_direction_to_wrong_rate": metrics.get(
                        "first_direction_to_wrong_rate"
                    ),
                }
            )
    return rows


def collect_offline(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/offline_*/offline_gap.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        match = re.fullmatch(r"offline_(\d+)", path.parent.name)
        if not match:
            continue
        checkpoint = int(match.group(1))
        task = task_from_path(path, root)
        rows.append(
            {
                "task": task,
                "experiment": str(path.parent.parent.relative_to(root)),
                "checkpoint": checkpoint,
                "budget_class": budget_class(task, checkpoint),
                "delta_cf": data.get("delta_cf"),
                "delta_cf_se": data.get("delta_cf_se"),
                "r_cf": data.get("r_cf"),
                "r_cf_se": data.get("r_cf_se"),
                "correct_action_loss": data.get("correct_action_loss"),
                "wrong_action_loss": data.get("wrong_action_loss"),
                "margin_violation": data.get("margin_violation"),
                "batches": data.get("batches"),
                "mc_samples": data.get("mc_samples"),
                "manifest_sha256": data.get("manifest_sha256", ""),
                "checkpoint_sha256": data.get("checkpoint_sha256", ""),
                "result_path": str(path),
            }
        )
    return rows


def collect_controls(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("robomimic_controls/*/result_*_memory-*.npy")):
        match = re.fullmatch(r"result_(\d+)_memory-(.+)", path.stem)
        if not match:
            continue
        checkpoint, intervention = int(match.group(1)), match.group(2)
        task = path.parent.name
        result = load_npy_dict(path)
        episode_success = result.get("episode_success")
        rows.append(
            {
                "task": task,
                "checkpoint": checkpoint,
                "budget_class": budget_class(task, checkpoint),
                "memory_intervention": intervention,
                "episodes": (
                    int(np.asarray(episode_success).size)
                    if episode_success is not None
                    else ""
                ),
                "mean_success": scalar(result.get("mean_success")),
                "mean_reward": scalar(result.get("mean_reward")),
                "eval_seed": result.get("eval_seed", ""),
                "result_path": str(path),
            }
        )
    return rows


def collect_parity_diagnostics(root: Path) -> list[dict[str, Any]]:
    """Collect protocol/parity checks separately from policy rollout curves."""

    rows: list[dict[str, Any]] = []
    rc3_root = root / "remember_color3" / "parity600" / "diagnostics"

    replay_path = rc3_root / "current_env_expert_replay10.json"
    if replay_path.is_file():
        data = json.loads(replay_path.read_text(encoding="utf-8"))
        episodes = int(data.get("replayed_episodes", 0))
        rate = scalar(data.get("current_env_expert_replay_success_rate"))
        rows.append(
            {
                "task": "remember_color3",
                "diagnostic": "old_demo_current_env_action_replay",
                "protocol": "official_60_step_success_once",
                "episodes": episodes,
                "successes": round(rate * episodes) if math.isfinite(rate) else "",
                "success_rate": rate,
                "reference_success_rate": scalar(data.get("dataset_success_rate")),
                "blocker": "data_environment_dynamics_mismatch",
                "result_path": str(replay_path),
            }
        )

    oracle_path = rc3_root / "current_env_state_oracle100.json"
    if oracle_path.is_file():
        data = json.loads(oracle_path.read_text(encoding="utf-8"))
        episodes = int(data.get("episodes_count", 0))
        rate = scalar(data.get("success_rate"))
        rows.append(
            {
                "task": "remember_color3",
                "diagnostic": "current_env_state_oracle",
                "protocol": "official_60_step_success_once",
                "episodes": episodes,
                "successes": round(rate * episodes) if math.isfinite(rate) else "",
                "success_rate": rate,
                "reference_success_rate": "",
                "blocker": "",
                "result_path": str(oracle_path),
            }
        )

    policy_path = rc3_root / "policy_10000_chunk4_eval30" / "result_episodes.json"
    if policy_path.is_file():
        data = json.loads(policy_path.read_text(encoding="utf-8"))
        episode_success = np.asarray(data.get("episode_success", []))
        rows.append(
            {
                "task": "remember_color3",
                "diagnostic": "old_demo_policy_10k_current_env",
                "protocol": "official_60_step_chunk4",
                "episodes": int(episode_success.size),
                "successes": int(episode_success.sum()),
                "success_rate": scalar(data.get("mean_success")),
                "branch_accuracy": scalar(data.get("branch_accuracy")),
                "reference_success_rate": "",
                "blocker": "data_environment_version_mismatch",
                "result_path": str(policy_path),
            }
        )

    swap_path = root / "swap_blocks" / "corrected_ood10" / "episodes.tsv"
    if swap_path.is_file():
        with swap_path.open(encoding="utf-8", newline="") as stream:
            swap_rows = list(csv.DictReader(stream, delimiter="\t"))
        for phase in ("dp", "vpwem"):
            phase_rows = [row for row in swap_rows if row.get("phase") == phase]
            if not phase_rows:
                continue
            successes = sum(int(row["success"]) for row in phase_rows)
            rows.append(
                {
                    "task": "swap_blocks",
                    "diagnostic": f"corrected_ood_{phase}",
                    "protocol": "oidn_continuous_queue_execute1",
                    "episodes": len(phase_rows),
                    "successes": successes,
                    "success_rate": successes / len(phase_rows),
                    "reference_success_rate": "",
                    "blocker": "coverage_generalization",
                    "result_path": str(swap_path),
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def control_deltas(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["checkpoint"]), {})[
            row["memory_intervention"]
        ] = row
    output = []
    for (task, checkpoint), modes in sorted(grouped.items()):
        if "none" not in modes or "zero" not in modes:
            continue
        none, zero = modes["none"], modes["zero"]
        output.append(
            {
                "task": task,
                "checkpoint": checkpoint,
                "normal_success": none["mean_success"],
                "zero_success": zero["mean_success"],
                "normal_minus_zero": none["mean_success"]
                - zero["mean_success"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.root / "summary"

    rollouts = collect_rollouts(args.root)
    causal = collect_causal(args.root)
    offline = collect_offline(args.root)
    controls = collect_controls(args.root)
    deltas = control_deltas(controls)
    parity = collect_parity_diagnostics(args.root)
    tables = {
        "rollout_curves": rollouts,
        "causal_interventions": causal,
        "offline_gaps": offline,
        "robomimic_controls": controls,
        "robomimic_control_deltas": deltas,
        "parity_diagnostics": parity,
    }
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    (output / "counts.json").write_text(
        json.dumps({key: len(value) for key, value in tables.items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), **{k: len(v) for k, v in tables.items()}}))


if __name__ == "__main__":
    main()
