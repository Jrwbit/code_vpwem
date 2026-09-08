"""Diagnostic RMBench DP deployment with the duplicated plan-boundary frame removed.

This module intentionally delegates model loading and observation encoding to the
official ``policy/DP`` implementation.  The only behavioral difference is that
the observation already inserted after the last action of a chunk is not inserted
again at the start of the next planning call.
"""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path

import numpy as np

from DP.deploy_policy import encode_obs, get_model as _get_official_model


def _same_current_observation(left, right) -> bool:
    """Use qpos for exact plan-boundary deduplication.

    Both observations are captured without another environment action between
    them.  A small tolerance avoids treating insignificant simulator float noise
    as a new frame.
    """

    return bool(
        np.allclose(
            np.asarray(left["agent_pos"]),
            np.asarray(right["agent_pos"]),
            rtol=0.0,
            atol=1e-7,
        )
    )


def _padded_last(frame_ids, count):
    values = list(frame_ids)[-count:]
    if values and len(values) < count:
        values = [values[0]] * (count - len(values)) + values
    return values


def _initialize_trace(model) -> None:
    maxlen = model.runner.obs.maxlen
    model._fixed_queue_frame_ids = deque(maxlen=maxlen)
    model._fixed_queue_next_frame_id = 0
    model._fixed_queue_trace_prints = 0
    model._rollout_diagnostics = {
        "fixed_queue_plan_count": 0,
        "fixed_queue_dedup_count": 0,
        "fixed_queue_actual_adjacent_duplicates": 0,
        "fixed_queue_original_adjacent_duplicates": 0,
    }
    model._reference_qpos = None
    model._reference_action_index = 0
    model._reference_plan_trace = []
    reference_path = os.environ.get("RMBENCH_DP_REFERENCE_HDF5")
    if reference_path:
        import h5py

        with h5py.File(reference_path, "r") as source:
            model._reference_qpos = np.asarray(
                source["joint_action/vector"][:], dtype=np.float64
            )
        model._rollout_diagnostics.update(
            reference_action_count=0,
            reference_action_mse_sum=0.0,
            reference_action_max_abs_error=0.0,
            reference_actual_qpos_max_abs_error=0.0,
            reference_first_action_error_gt_0p01=-1,
            reference_first_action_error_gt_0p03=-1,
            reference_first_actual_qpos_error_gt_0p01=-1,
            reference_first_actual_qpos_error_gt_0p03=-1,
        )


def _write_reference_trace(model, task_env) -> None:
    output = os.environ.get("RMBENCH_DP_TRACE_OUTPUT")
    if not output or model._reference_qpos is None:
        return
    diagnostics = dict(model._rollout_diagnostics)
    count = int(diagnostics.get("reference_action_count", 0))
    diagnostics["reference_action_mse_mean"] = (
        float(diagnostics["reference_action_mse_sum"] / count) if count else None
    )
    payload = {
        "reference_hdf5": os.environ.get("RMBENCH_DP_REFERENCE_HDF5"),
        "reference_semantics": "policy action k compared with stored qpos[k+1]",
        "denoiser": os.environ.get("RMBENCH_RT_DENOISER", "oidn"),
        "task_take_action_count": int(task_env.take_action_cnt),
        "eval_success": bool(task_env.eval_success),
        "max_reward": float(task_env.max_reward),
        "diagnostics": diagnostics,
        "plans": model._reference_plan_trace,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record_threshold(diagnostics, key: str, value: float, threshold: float, action_index: int):
    if value > threshold and diagnostics[key] < 0:
        diagnostics[key] = int(action_index)


def get_model(usr_args):
    # Allow the result namespace to describe the diagnostic while still loading
    # the untouched official checkpoint from its original setting directory.
    model_args = dict(usr_args)
    model_args["ckpt_setting"] = usr_args.get("dp_ckpt_setting", usr_args["ckpt_setting"])
    model = _get_official_model(model_args)
    _initialize_trace(model)
    return model


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    if not hasattr(model, "_fixed_queue_frame_ids"):
        _initialize_trace(model)

    frame_ids = model._fixed_queue_frame_ids
    raw_would_duplicate = bool(model.runner.obs) and _same_current_observation(
        model.runner.obs[-1], obs
    )

    if raw_would_duplicate:
        # The official rollout calls get_action(obs), which appends obs again.
        # Reuse the already-cached frame instead.
        actions = model.get_action(None)
        model._rollout_diagnostics["fixed_queue_dedup_count"] += 1
    else:
        actions = model.get_action(obs)
        frame_ids.append(model._fixed_queue_next_frame_id)
        model._fixed_queue_next_frame_id += 1

    n_obs = int(model.runner.n_obs_steps)
    actual_ids = _padded_last(frame_ids, n_obs)
    original_ids = _padded_last(
        list(frame_ids) + ([frame_ids[-1]] if raw_would_duplicate else []), n_obs
    )
    model._rollout_diagnostics["fixed_queue_plan_count"] += 1
    model._rollout_diagnostics["fixed_queue_actual_adjacent_duplicates"] += int(
        any(a == b for a, b in zip(actual_ids, actual_ids[1:]))
    )
    model._rollout_diagnostics["fixed_queue_original_adjacent_duplicates"] += int(
        any(a == b for a, b in zip(original_ids, original_ids[1:]))
    )

    if model._fixed_queue_trace_prints < 3:
        print(
            "[fixed_queue_trace] "
            f"plan={model._rollout_diagnostics['fixed_queue_plan_count']} "
            f"dedup={int(raw_would_duplicate)} "
            f"actual_input_frame_ids={actual_ids} "
            f"official_input_frame_ids={original_ids}",
            flush=True,
        )
        model._fixed_queue_trace_prints += 1

    plan_reference = {
        "plan": int(model._rollout_diagnostics["fixed_queue_plan_count"]),
        "input_frame_ids": actual_ids,
        "first_policy_action_index": int(model._reference_action_index + 1),
        "actions": [],
    }
    for action in actions:
        target = None
        action_index = model._reference_action_index + 1
        if model._reference_qpos is not None and action_index < len(model._reference_qpos):
            target = model._reference_qpos[action_index]
            action_error = np.asarray(action, dtype=np.float64) - target
            action_mse = float(np.mean(action_error**2))
            action_max = float(np.max(np.abs(action_error)))
            diagnostics = model._rollout_diagnostics
            diagnostics["reference_action_count"] += 1
            diagnostics["reference_action_mse_sum"] += action_mse
            diagnostics["reference_action_max_abs_error"] = max(
                diagnostics["reference_action_max_abs_error"], action_max
            )
            _record_threshold(
                diagnostics,
                "reference_first_action_error_gt_0p01",
                action_max,
                0.01,
                action_index,
            )
            _record_threshold(
                diagnostics,
                "reference_first_action_error_gt_0p03",
                action_max,
                0.03,
                action_index,
            )
        else:
            action_mse = None
            action_max = None

        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)
        frame_ids.append(model._fixed_queue_next_frame_id)
        model._fixed_queue_next_frame_id += 1
        if target is not None:
            actual_error = np.asarray(obs["agent_pos"], dtype=np.float64) - target
            actual_max = float(np.max(np.abs(actual_error)))
            diagnostics["reference_actual_qpos_max_abs_error"] = max(
                diagnostics["reference_actual_qpos_max_abs_error"], actual_max
            )
            _record_threshold(
                diagnostics,
                "reference_first_actual_qpos_error_gt_0p01",
                actual_max,
                0.01,
                action_index,
            )
            _record_threshold(
                diagnostics,
                "reference_first_actual_qpos_error_gt_0p03",
                actual_max,
                0.03,
                action_index,
            )
        else:
            actual_max = None
        plan_reference["actions"].append(
            {
                "action_index": int(action_index),
                "policy_vs_reference_mse": action_mse,
                "policy_vs_reference_max_abs": action_max,
                "actual_qpos_vs_reference_max_abs": actual_max,
            }
        )
        model._reference_action_index += 1

    if model._reference_qpos is not None:
        model._reference_plan_trace.append(plan_reference)
        _write_reference_trace(model, TASK_ENV)


def reset_model(model):
    model.reset_obs()
    _initialize_trace(model)
