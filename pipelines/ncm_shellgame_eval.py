import collections
import hashlib
import json
import os
import pathlib
import time
from pathlib import Path

import hydra
import gymnasium as gym
import numpy as np
import torch

from cleandiffuser.dataset.mikasa_dataset_longmem import LongMemMikasaDataset
from cleandiffuser.env.gymnasium_wrapper import (
    FlattenRGBDObservationWrapper,
    MultiStepWrapper,
)
from cleandiffuser.utils import report_parameters
from utils import Logger, set_seed

import mikasa_robo_suite  # noqa: F401
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info


INTERVENTIONS = ("correct", "zero", "random", "matched_wrong")


class TargetPositionInfoWrapper(gym.Wrapper):
    """Expose eval-only object geometry through ``info``, never policy obs."""

    def _attach(self, info):
        info = dict(info)
        base = self.env.unwrapped
        positions = [
            base.cubes[key].pose.p.detach().cpu()
            for key in sorted(base.cubes)
        ]
        info["all_target_positions"] = torch.stack(positions, dim=1)
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._attach(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._attach(info)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_envs(args):
    env = gym.make(
        args.env_name, num_envs=args.num_envs, obs_mode="rgb",
        render_mode="all", control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense", sim_backend="gpu")
    wrappers, timeout = env_info(args.env_name)
    for wrapper, kwargs in wrappers:
        env = wrapper(env, **kwargs)
    if args.env_name.startswith("RememberColor"):
        env = TargetPositionInfoWrapper(env)
    env = FlattenRGBDObservationWrapper(env, rgb=True, joints=True)
    env = MultiStepWrapper(
        env, n_obs_steps=args.obs_steps, n_action_steps=1,
        max_episode_steps=timeout)
    env.benchmark_episode_timeout = int(timeout)
    return env


def normalized_observation(args, obs, normalizer):
    result = {}
    for key, value in obs.items():
        value = torch.as_tensor(value, dtype=torch.float32, device=args.device)
        normalized = normalizer["obs"][key].normalize(value)
        result[key] = torch.as_tensor(
            normalized, dtype=torch.float32, device=args.device)
    return result


def current_embeddings(pairing_encoder, condition):
    cloned = {key: value.clone() for key, value in condition.items()}
    encoded = pairing_encoder(cloned).flatten(start_dim=1)
    return torch.nn.functional.normalize(encoded, dim=-1)


def matched_wrong_donors(pairing_encoder, condition, branches):
    features = current_embeddings(pairing_encoder, condition)
    similarity = features @ features.T
    branches = torch.as_tensor(
        branches, dtype=torch.long, device=similarity.device).reshape(-1)
    invalid = branches[:, None].eq(branches[None, :])
    similarity = similarity.masked_fill(invalid, -torch.inf)
    if torch.any(torch.isneginf(similarity).all(dim=1)):
        raise RuntimeError(
            "An eval batch has no different-branch donor for at least one episode")
    donor = similarity.argmax(dim=1)
    selected_similarity = similarity.gather(1, donor[:, None]).squeeze(1)
    return donor, selected_similarity


def target_positions_from_info(args, obs, info):
    tcp = np.asarray(obs["joints"][:, -1, :3], dtype=np.float32)
    gt = np.asarray(info["oracle_info"][:, -1], dtype=np.int64).reshape(-1)
    if args.env_name.startswith("RememberColor"):
        targets = np.asarray(
            info["all_target_positions"][:, -1], dtype=np.float32)
        if targets.shape[1] != int(args.ncm_num_branches):
            raise RuntimeError(
                f"Expected {args.ncm_num_branches} targets, got "
                f"{targets.shape[1]}")
        return gt, tcp, targets

    obj_to_goal = np.asarray(
        info["obj_to_goal_pos"][:, -1], dtype=np.float32)
    target = tcp + obj_to_goal
    cups = np.repeat(target[:, None, :], 3, axis=1)
    for cup in range(3):
        cups[:, cup, 1] += (cup - gt) * 0.2
    return gt, tcp, cups


def rollout_mode(
        args, envs, normalizer, agent, pairing_encoder, intervention):
    if intervention not in INTERVENTIONS:
        raise ValueError(intervention)
    if int(args.inference_action_steps) != 1:
        raise ValueError("NCM causal eval requires inference_action_steps=1")
    if args.eval_episodes % args.num_envs:
        raise ValueError("eval_episodes must be divisible by num_envs")

    set_seed(int(args.diffusion_seed))
    episode_success, episode_reward = [], []
    episode_gt, episode_donor, episode_choice = [], [], []
    episode_first_direction, episode_pair_similarity = [], []
    episode_reset_seed, episode_env_slot = [], []
    infer_times = []

    decision_step = int(args.ncm_decision_step)
    intervention_steps = int(args.ncm_intervention_steps)
    commit_threshold = float(args.ncm_commit_threshold)
    timeout = int(envs.benchmark_episode_timeout)

    for batch_id in range(args.eval_episodes // args.num_envs):
        reset_seed = int(args.eval_seed_start) + batch_id * args.num_envs
        obs, info = envs.reset(seed=reset_seed)
        agent.reset_cache()
        agent.model["memory"].reset()
        agent.model_ema["memory"].reset()
        previous_action_normalized = None

        rewards = np.zeros(args.num_envs, dtype=np.float32)
        success = np.zeros(args.num_envs, dtype=bool)
        choice = np.full(args.num_envs, -1, dtype=np.int8)
        donor_branch = np.full(args.num_envs, -1, dtype=np.int8)
        pair_similarity = np.full(args.num_envs, np.nan, dtype=np.float32)
        donor_index = None
        gt_branch = None
        cup_positions = None
        initial_distance = None
        min_decision_distance = None

        for step in range(timeout):
            condition = normalized_observation(args, obs, normalizer)
            if step == decision_step:
                gt_branch, tcp, cup_positions = target_positions_from_info(
                    args, obs, info)
                distances = np.linalg.norm(
                    tcp[:, None, :] - cup_positions, axis=-1)
                initial_distance = distances.copy()
                min_decision_distance = distances.copy()
                if intervention == "matched_wrong":
                    if pairing_encoder is None:
                        raise RuntimeError(
                            "matched_wrong requires a pairing encoder")
                    with torch.no_grad():
                        donor_index, similarity = matched_wrong_donors(
                            pairing_encoder, condition, gt_branch)
                    donor_numpy = donor_index.detach().cpu().numpy()
                    donor_branch = gt_branch[donor_numpy].astype(np.int8)
                    pair_similarity = similarity.detach().cpu().numpy().astype(
                        np.float32)

            active_intervention = (
                step >= decision_step
                and step < decision_step + intervention_steps)
            memory_intervention = "none"
            memory_index = None
            if active_intervention:
                if intervention == "zero":
                    memory_intervention = "zero"
                elif intervention == "random":
                    memory_intervention = "shuffle_batch"
                elif intervention == "matched_wrong":
                    memory_index = donor_index

            start_time = time.time()
            with torch.no_grad():
                prior = torch.zeros(
                    (args.num_envs, args.horizon, args.action_dim),
                    device=args.device)
                normalized_action, _ = agent.sample(
                    prior=prior,
                    n_samples=args.num_envs,
                    sample_steps=args.sample_steps,
                    solver=None,
                    condition_cfg=condition,
                    action_context_cfg=previous_action_normalized,
                    w_cfg=1.0,
                    temperature=args.temperature,
                    use_ema=True,
                    memory_intervention=memory_intervention,
                    memory_index=memory_index,
                )
            infer_times.append(time.time() - start_time)
            action_prediction = normalizer["action"].unnormalize(
                normalized_action.detach())
            action_prediction = torch.as_tensor(
                action_prediction).detach().cpu().numpy()
            start = args.obs_steps - 1
            action = action_prediction[:, start:start + 1, :]
            if step < decision_step:
                # The policy forward above is intentionally retained so the
                # online memory observes the cue. Only motor execution is held.
                action = np.zeros_like(action)
                action[..., -1] = 1.0

            executed_action = torch.as_tensor(
                action, dtype=torch.float32, device=args.device)
            previous_action_normalized = normalizer["action"].normalize(
                executed_action).detach()

            obs, reward, done, info = envs.step(action)
            rewards += np.asarray(reward, dtype=np.float32)
            success_now = np.asarray(info["success"][:, -1], dtype=bool)
            success |= success_now

            if cup_positions is not None:
                tcp = np.asarray(obs["joints"][:, -1, :3], dtype=np.float32)
                distances = np.linalg.norm(
                    tcp[:, None, :] - cup_positions, axis=-1)
                undecided = choice < 0
                committed = distances.min(axis=1) <= commit_threshold
                newly_committed = undecided & committed
                choice[newly_committed] = distances[newly_committed].argmin(
                    axis=1).astype(np.int8)
                if step < decision_step + intervention_steps:
                    min_decision_distance = np.minimum(
                        min_decision_distance, distances)
            if np.all(done):
                break

        if gt_branch is None:
            raise RuntimeError("Rollout ended before the NCM decision step")
        approach = initial_distance - min_decision_distance
        first_direction = approach.argmax(axis=1).astype(np.int8)
        if intervention == "random":
            donor_branch = np.roll(gt_branch, 1).astype(np.int8)
        elif intervention in ("correct", "zero"):
            donor_branch.fill(-1)

        episode_success.extend(success.astype(np.int8).tolist())
        episode_reward.extend(rewards.tolist())
        episode_gt.extend(gt_branch.astype(np.int8).tolist())
        episode_donor.extend(donor_branch.tolist())
        episode_choice.extend(choice.tolist())
        episode_first_direction.extend(first_direction.tolist())
        episode_pair_similarity.extend(pair_similarity.tolist())
        episode_reset_seed.extend([reset_seed] * args.num_envs)
        episode_env_slot.extend(range(args.num_envs))

    gt = np.asarray(episode_gt, dtype=np.int8)
    donor = np.asarray(episode_donor, dtype=np.int8)
    choice = np.asarray(episode_choice, dtype=np.int8)
    success = np.asarray(episode_success, dtype=np.int8)
    committed = choice >= 0
    success_by_branch = {
        str(branch): float(success[gt == branch].mean())
        for branch in range(int(args.ncm_num_branches))
    }
    branch_acc_by_branch = {
        str(branch): float(np.mean(choice[gt == branch] == branch))
        for branch in range(int(args.ncm_num_branches))
    }
    result = {
        "intervention": intervention,
        "mean_success": float(success.mean()),
        "branch_acc": float(np.mean(choice == gt)),
        "mean_success_macro": float(np.mean(list(
            success_by_branch.values()))),
        "branch_acc_macro": float(np.mean(list(
            branch_acc_by_branch.values()))),
        "success_by_gt_branch": success_by_branch,
        "branch_acc_by_gt_branch": branch_acc_by_branch,
        "commit_rate": float(committed.mean()),
        "conditional_branch_acc": float(
            np.mean(choice[committed] == gt[committed])) if committed.any() else 0.0,
        "wrong_branch_choice_rate": float(
            np.mean(choice == donor)) if np.any(donor >= 0) else float("nan"),
        "first_direction_to_wrong_rate": float(np.mean(
            np.asarray(episode_first_direction) == donor
        )) if np.any(donor >= 0) else float("nan"),
        "avg_infer_time": float(np.mean(infer_times)),
        "episode_success": success,
        "episode_reward": np.asarray(episode_reward, dtype=np.float32),
        "episode_gt_branch": gt,
        "episode_donor_branch": donor,
        "episode_branch_choice": choice,
        "episode_committed": committed.astype(np.int8),
        "episode_first_direction": np.asarray(
            episode_first_direction, dtype=np.int8),
        "episode_pair_similarity": np.asarray(
            episode_pair_similarity, dtype=np.float32),
        "episode_reset_seed": np.asarray(episode_reset_seed, dtype=np.int64),
        "episode_env_slot": np.asarray(episode_env_slot, dtype=np.int16),
    }
    return result


def build_agent(args, normalizer):
    from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM
    from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
    from cleandiffuser.nn_diffusion.longmem_chitransformerptp import (
        LongMemChiTransformerPTP,
    )

    condition = MultiImageObsConditionRobomimic(
        shape_meta=args.shape_meta, emb_dim=256,
        rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
        crop_shape=args.crop_shape, random_crop=args.random_crop,
        use_group_norm=args.use_group_norm, use_seq=args.use_seq,
        keep_horizon_dims=True).to(args.device)
    diffusion = LongMemChiTransformerPTP(
        args.action_dim, 256, args.horizon, args.obs_steps,
        args.mem_compress_length, d_model=256, nhead=4,
        num_layers=args.diffusion_num_layers,
        p_drop_emb=args.p_drop_emb, p_drop_attn=args.p_drop_attn,
        memory_read_mode=str(getattr(
            args, "memory_read_mode", "context")),
        memory_gate_bias=float(getattr(args, "memory_gate_bias", 0.0)),
        timestep_emb_type="positional").to(args.device)
    if args.mem_compress_method == "qformer":
        from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
        memory = QFormerMemory(
            mem_compress_length=args.mem_compress_length,
            emb_dim=args.qformer_emb_dim,
            num_layers=args.qformer_num_layers,
            cache_compress=args.qformer_cache_compress,
            cache_max_length=args.qformer_cache_max_length,
            use_pos_emb=args.memory_pos_emb,
            subsample_ratio=args.dataset_memory_subsample_ratio).to(args.device)
    elif args.mem_compress_method in {
            "aem_style", "dssp_style", "vq_style", "cycle_progress"}:
        from cleandiffuser.nn_memory.paper_transfer_factory import (
            build_transfer_memory,
        )
        memory = build_transfer_memory(args).to(args.device)
    else:
        raise ValueError(
            f"NCM evaluator does not support memory method "
            f"{args.mem_compress_method!r}")
    report_parameters(diffusion)
    report_parameters(memory)
    x_max = torch.ones(
        (1, args.horizon, args.action_dim), device=args.device)
    x_min = -x_max
    return LongMemDDPM(
        nn_diffusion=diffusion, nn_condition=condition, nn_memory=memory,
        device=args.device, diffusion_steps=args.sample_steps,
        x_max=x_max, x_min=x_min, args_dict=args,
        action_range=normalizer["action"].range.to(args.device),
        action_min=normalizer["action"].min.to(args.device),
        short_cond_dropout=args.short_cond_dropout,
        long_cond_dropout=args.long_cond_dropout,
        optim_params={"lr": args.lr})


def build_pairing_encoder(args):
    from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic

    path = Path(str(args.pairing_checkpoint_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen pairing checkpoint does not exist: {path}")
    encoder = MultiImageObsConditionRobomimic(
        shape_meta=args.shape_meta, emb_dim=256,
        rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
        crop_shape=args.crop_shape, random_crop=args.random_crop,
        use_group_norm=args.use_group_norm, use_seq=args.use_seq,
        keep_horizon_dims=True).to(args.device)
    checkpoint = torch.load(path, map_location=args.device, weights_only=False)
    prefix = "condition."
    encoder.load_state_dict({
        key[len(prefix):]: value
        for key, value in checkpoint["model_ema"].items()
        if key.startswith(prefix)
    }, strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def serializable_summary(value):
    if isinstance(value, dict):
        return {key: serializable_summary(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@hydra.main(
    config_path="../configs/dp/mikasa/chi_transformer_ptp_longmem",
    config_name="shell_game_touch_ncm")
def pipeline(args):
    set_seed(args.seed)
    Logger(pathlib.Path(args.work_dir), args)
    supported = {"ShellGameTouch-v0", "RememberColor3-v0", "RememberColor5-v0"}
    if args.env_name not in supported:
        raise ValueError(f"NCM evaluator does not support {args.env_name}")
    if args.horizon != args.obs_steps + args.action_steps - 1:
        raise ValueError("Invalid diffusion horizon")
    marker_path = Path(args.dataset_path).expanduser() / "NCM_PHYSICAL_HOLD.json"
    if not marker_path.is_file():
        raise ValueError(
            "NCM evaluation normalizer must come from the physical-hold dataset")

    dataset = LongMemMikasaDataset(
        dataset_dir=os.path.expanduser(args.dataset_path),
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        horizon=args.horizon,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        dataset_memory_type=args.dataset_memory_type,
        subsample_ratio=args.dataset_memory_subsample_ratio,
        maximum_memory_length=int(getattr(
            args, "dataset_maximum_memory_length", 16)),
        conversion_n_workers=int(args.conversion_n_workers),
        conversion_max_inflight_tasks=int(args.conversion_max_inflight_tasks),
        load_images=False,
        device=args.device)
    normalizer = dataset.normalizer
    if args.normalizer_dataset_path:
        normalizer_dataset = LongMemMikasaDataset(
            dataset_dir=os.path.expanduser(str(args.normalizer_dataset_path)),
            shape_meta=args.shape_meta,
            n_obs_steps=args.obs_steps,
            horizon=args.horizon,
            pad_before=args.obs_steps - 1,
            pad_after=args.action_steps - 1,
            dataset_memory_type=args.dataset_memory_type,
            subsample_ratio=args.dataset_memory_subsample_ratio,
            maximum_memory_length=int(getattr(
                args, "dataset_maximum_memory_length", 16)),
            conversion_n_workers=int(args.conversion_n_workers),
            conversion_max_inflight_tasks=int(
                args.conversion_max_inflight_tasks),
            load_images=False,
            device=args.device)
        normalizer = normalizer_dataset.normalizer
    agent = build_agent(args, normalizer)
    interventions = tuple(str(mode) for mode in getattr(
        args, "ncm_memory_interventions", INTERVENTIONS))
    invalid_interventions = set(interventions) - set(INTERVENTIONS)
    if invalid_interventions:
        raise ValueError(
            f"Unknown NCM interventions: {sorted(invalid_interventions)}")
    if not interventions:
        raise ValueError("ncm_memory_interventions cannot be empty")
    pairing_encoder = (
        build_pairing_encoder(args)
        if "matched_wrong" in interventions else None)
    model_path = os.path.expanduser(str(args.model_path))
    agent.load(model_path)
    agent.model.eval()
    agent.model_ema.eval()
    envs = make_envs(args)

    results = {}
    try:
        for intervention in interventions:
            print(f"Evaluating intervention={intervention}", flush=True)
            results[intervention] = rollout_mode(
                args, envs, normalizer, agent, pairing_encoder, intervention)
            print({
                key: value for key, value in results[intervention].items()
                if not isinstance(value, np.ndarray)
            }, flush=True)
    finally:
        envs.close()

    summary = {
        "checkpoint": model_path,
        "checkpoint_sha256": file_sha256(model_path),
        "eval_seed_start": int(args.eval_seed_start),
        "diffusion_seed": int(args.diffusion_seed),
        "modes": {
            mode: {
                key: value for key, value in result.items()
                if not isinstance(value, np.ndarray)
            }
            for mode, result in results.items()
        },
    }
    if pairing_encoder is not None:
        summary["pairing_checkpoint_sha256"] = file_sha256(
            Path(str(args.pairing_checkpoint_path)).expanduser())
    if "correct" in results and "matched_wrong" in results:
        correct = results["correct"]
        wrong = results["matched_wrong"]
        correct_choice = correct["episode_branch_choice"]
        wrong_choice = wrong["episode_branch_choice"]
        donor = wrong["episode_donor_branch"]
        correct_success = correct["episode_success"]
        wrong_success = wrong["episode_success"]
        eligible_flip = correct_choice == correct["episode_gt_branch"]
        summary.update({
            "success_gap_correct_minus_matched_wrong": float(np.mean(
                correct_success - wrong_success)),
            "branch_gap_correct_minus_matched_wrong": float(np.mean(
                (correct_choice == correct["episode_gt_branch"]).astype(
                    np.float32)
                - (wrong_choice == wrong["episode_gt_branch"]).astype(
                    np.float32))),
            "wrong_attraction_uplift": float(np.mean(
                (wrong_choice == donor).astype(np.float32)
                - (correct_choice == donor).astype(np.float32))),
            "first_direction_wrong_uplift": float(np.mean(
                (wrong["episode_first_direction"] == donor).astype(
                    np.float32)
                - (correct["episode_first_direction"] == donor).astype(
                    np.float32))),
            "flip_to_wrong_given_correct": float(np.mean(
                wrong_choice[eligible_flip] == donor[eligible_flip]
            )) if np.any(eligible_flip) else float("nan"),
        })
    payload = {"summary": summary, "modes": results}
    output_path = Path(args.result_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, payload)
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as stream:
        json.dump(serializable_summary(summary), stream, indent=2, sort_keys=True)
    print(json.dumps(serializable_summary(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    pipeline()
