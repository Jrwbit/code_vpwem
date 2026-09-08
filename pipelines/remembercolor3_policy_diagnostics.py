"""RememberColor3 policy evaluator with policy-free per-episode diagnostics.

Oracle geometry is added only to ``info`` after the normal MIKASA state
wrappers.  ``FlattenRGBDObservationWrapper`` still emits exactly the policy's
``rgb``, ``rgb_wrist`` and ``joints`` inputs.
"""

import json
import os
import time

import gymnasium as gym
import hydra
import numpy as np
import torch

import longmemdpptp_mikasa_image_eval as base_eval
from cleandiffuser.env.gymnasium_wrapper import (
    FlattenRGBDObservationWrapper,
    MultiStepWrapper,
)
from cleandiffuser.utils.remembercolor_diagnostics import (
    COLOR_NAMES,
    EpisodeBranchTracker,
)
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info


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


class RememberColorGeometryInfoWrapper(gym.Wrapper):
    """Expose target geometry in info without changing observations."""

    def _augment(self, info):
        info = dict(info)
        base = self.unwrapped
        keys = sorted(base.cubes.keys())
        positions = torch.stack([base.cubes[key].pose.p for key in keys], dim=1)
        tcp_position = base.agent.tcp.pose.p
        distances = torch.linalg.norm(
            positions - tcp_position.unsqueeze(1), dim=-1)
        info.update({
            "all_target_positions": positions,
            "tcp_position": tcp_position,
            "all_target_distances": distances,
        })
        return info

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        return observation, self._augment(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated, truncated, self._augment(info)


class DiagnosticsMultiStepWrapper(MultiStepWrapper):
    DIAGNOSTIC_INFO_KEYS = MultiStepWrapper.DIAGNOSTIC_INFO_KEYS + (
        "is_obj_placed",
        "is_robot_static",
        "tcp_position",
        "all_target_distances",
    )


def make_async_envs(args):
    if str(args.env_name) != "RememberColor3-v0":
        raise ValueError("this evaluator is intentionally restricted to RememberColor3-v0")
    print(f"Starting {args.num_envs} diagnostic RememberColor3 environments...")
    env = gym.make(
        args.env_name,
        num_envs=args.num_envs,
        obs_mode="rgb",
        render_mode="all",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        sim_backend="gpu",
    )
    state_wrappers_list, episode_timeout = env_info(args.env_name)
    print(f"Registered episode timeout: {episode_timeout}")
    for wrapper_class, wrapper_kwargs in state_wrappers_list:
        env = wrapper_class(env, **wrapper_kwargs)
    env = RememberColorGeometryInfoWrapper(env)
    env = FlattenRGBDObservationWrapper(env, rgb=True, joints=True)
    env = DiagnosticsMultiStepWrapper(
        env,
        n_obs_steps=args.action_steps,
        n_action_steps=args.inference_action_steps,
        max_episode_steps=args.max_episode_steps,
    )
    return env


def _last_substeps(info, key, count):
    value = _numpy(info[key])
    if value.ndim < 2:
        raise ValueError(f"diagnostic {key} has unexpected shape {value.shape}")
    return value[:, -count:]


def inference(args, envs, normalizer, agent, video_path):
    del video_path  # This independent diagnostic never writes policy video frames.
    if bool(args.save_video):
        raise ValueError("save_video must remain false for this diagnostic evaluator")
    solver = {
        "ddpm": None,
        "ddim": "ddim",
        "dpm": "ode_dpmpp_2",
        "edm": "euler",
    }[str(args.diffusion)]
    decision_start = int(getattr(args, "diagnostics_decision_start", 10))
    expected_obs_keys = set(args.shape_meta.obs.keys())

    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_reset_seeds = []
    episode_vector_indices = []
    tracker_results = []
    all_all_actions = []
    inference_times = []

    n_loops = int(args.eval_episodes) // int(args.num_envs)
    for loop_idx in range(n_loops):
        reset_seed = int(args.seed) + loop_idx * int(args.num_envs)
        obs, reset_info = envs.reset(seed=reset_seed)
        if set(obs.keys()) != expected_obs_keys:
            raise AssertionError(
                "diagnostic wrapper changed policy inputs: "
                f"actual={sorted(obs)}, expected={sorted(expected_obs_keys)}")

        ground_truth = _last_substeps(reset_info, "oracle_info", 1)
        ground_truth = ground_truth.reshape(args.num_envs, -1)[:, 0].astype(np.int64)
        tracker = EpisodeBranchTracker(
            ground_truth, n_branches=3, decision_start=decision_start)
        ep_reward = np.zeros(args.num_envs, dtype=np.float32)
        success_once = np.zeros(args.num_envs, dtype=bool)
        t = 0
        loop_actions = []

        agent.reset_cache()
        agent.model["memory"].reset()
        agent.model_ema["memory"].reset()

        while t < int(args.max_episode_steps):
            started = time.time()
            obs_dict = {}
            for key, obs_seq in obs.items():
                if isinstance(obs_seq, np.ndarray):
                    obs_seq = torch.from_numpy(obs_seq)
                obs_seq = obs_seq.to(args.device)
                obs_dict[key] = normalizer["obs"][key].normalize(obs_seq)
            with torch.no_grad():
                prior = torch.zeros(
                    (args.num_envs, args.horizon, args.action_dim),
                    device=args.device,
                )
                normalized_action, _ = agent.sample(
                    prior=prior,
                    n_samples=args.num_envs,
                    sample_steps=args.sample_steps,
                    solver=solver,
                    condition_cfg=obs_dict,
                    w_cfg=1.0,
                    temperature=args.temperature,
                    use_ema=True,
                )
            action_prediction = normalizer["action"].unnormalize(
                normalized_action.detach())
            action_prediction = _numpy(action_prediction)
            start = int(args.obs_steps) - 1
            action = action_prediction[
                :, start:start + int(args.inference_action_steps), :]
            loop_actions.append(action)
            inference_times.append(time.time() - started)

            obs, reward, done, info = envs.step(action)
            actual_substeps = min(
                action.shape[1], int(args.max_episode_steps) - t)
            success_chunk = _last_substeps(info, "success", actual_substeps).astype(bool)
            tracker.update(
                elapsed_steps=_last_substeps(
                    info, "elapsed_steps", actual_substeps),
                branch_distances=_last_substeps(
                    info, "all_target_distances", actual_substeps),
                is_static=_last_substeps(
                    info, "is_robot_static", actual_substeps),
                is_placed=_last_substeps(
                    info, "is_obj_placed", actual_substeps),
                success=success_chunk,
            )
            success_once |= success_chunk.any(axis=1)
            ep_reward += np.asarray(reward, dtype=np.float32)
            t += actual_substeps
            if np.all(done):
                break

        tracked = tracker.finalize()
        tracker_results.append(tracked)
        episode_rewards.extend(ep_reward.tolist())
        episode_steps.extend([t] * args.num_envs)
        episode_success.extend(success_once.tolist())
        episode_reset_seeds.extend([reset_seed] * args.num_envs)
        episode_vector_indices.extend(range(args.num_envs))
        all_all_actions.append(np.concatenate(loop_actions, axis=1))
        print(
            f"[Diagnostic {loop_idx + 1}/{n_loops}] seed={reset_seed} "
            f"success={success_once.astype(int).tolist()} "
            f"branch={tracked['branch_choice'].tolist()} "
            f"gt={ground_truth.tolist()}",
            flush=True,
        )

    tracked = {
        key: np.concatenate([result[key] for result in tracker_results], axis=0)
        for key in tracker_results[0]
    }
    all_actions = np.concatenate(all_all_actions, axis=0)
    result = {
        "diagnostics_schema_version": 1,
        "branch_choice_definition": (
            f"argmin TCP-to-branch distance over elapsed_steps>={decision_start}"),
        "policy_observation_keys": np.asarray(sorted(expected_obs_keys)),
        "oracle_used_as_policy_input": False,
        "eval_seed": int(args.seed),
        "episode_id": np.arange(len(episode_success), dtype=np.int64),
        "episode_reset_seed": np.asarray(episode_reset_seeds, dtype=np.int64),
        "episode_vector_index": np.asarray(episode_vector_indices, dtype=np.int64),
        "episode_reward": np.asarray(episode_rewards, dtype=np.float32),
        "episode_steps": np.asarray(episode_steps, dtype=np.int32),
        "episode_success": np.asarray(episode_success, dtype=np.int8),
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_step": float(np.mean(episode_steps)),
        "mean_success": float(np.mean(episode_success)),
        "branch_accuracy": float(np.mean(tracked["branch_choice_correct"])),
        "final_closest_accuracy": float(np.mean(tracked["final_closest_correct"])),
        "avg_infer_time": float(np.mean(inference_times)),
        "mlp_corr_pred_actions_full_traj_online_fixed": float(
            base_eval.batch_mlp_corr(all_actions)),
        **tracked,
    }
    result["ground_truth_color"] = np.asarray([
        COLOR_NAMES[index] for index in result["ground_truth_branch"]])
    result["branch_choice_color"] = np.asarray([
        COLOR_NAMES[index] if index >= 0 else "Unknown"
        for index in result["branch_choice"]])

    configured_result_path = str(getattr(args, "result_path", ""))
    if configured_result_path.lower() in {"", "false", "none"}:
        json_path = os.path.join(args.work_dir, "rc3_policy_diagnostics.json")
    else:
        stem, _ = os.path.splitext(os.path.expanduser(configured_result_path))
        json_path = stem + "_episodes.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(result), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved diagnostic JSON: {json_path}", flush=True)
    print({
        "mean_success": result["mean_success"],
        "branch_accuracy": result["branch_accuracy"],
        "final_closest_accuracy": result["final_closest_accuracy"],
        "mean_min_correct_target_distance": float(np.mean(
            result["min_correct_target_distance"])),
    }, flush=True)
    return result


@hydra.main(
    config_path="../configs/dp/mikasa/chi_transformer_ptp_longmem",
    config_name="remember_color_3",
)
def pipeline(args):
    base_eval.make_async_envs = make_async_envs
    base_eval.inference = inference
    # Call the undecorated implementation so Hydra resolves this script's own
    # explicit config path instead of the imported module's caller location.
    return base_eval.pipeline.__wrapped__(args)


if __name__ == "__main__":
    pipeline()
