"""Train/evaluate four history ablations under one frozen MIKASA protocol."""

import collections
import os
import pathlib
import time
import warnings

import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import Logger, auto_resume_training, prune_checkpoints, set_seed
from cleandiffuser.dataset.dataset_utils import (
    dynamic_batch_collate_fn, loop_dataloader)
from cleandiffuser.dataset.mikasa_dataset_longmem import LongMemMikasaDataset
from cleandiffuser.utils import report_parameters

warnings.filterwarnings("ignore")


METHODS = {"vpwem", "direct", "no_history", "direct_token2"}


def make_dataset(args, load_images=True):
    return LongMemMikasaDataset(
        dataset_dir=os.path.expanduser(args.dataset_path),
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        horizon=args.horizon,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        dataset_memory_type="random_from_full",
        subsample_ratio=args.dataset_memory_subsample_ratio,
        maximum_memory_length=args.dataset_maximum_memory_length,
        preserve_episode_anchor=True,
        history_sampling_seed=args.history_sampling_seed,
        conversion_n_workers=args.conversion_n_workers,
        conversion_max_inflight_tasks=args.conversion_max_inflight_tasks,
        load_images=load_images,
        device=args.device,
    )


def build_agent(args, normalizer):
    from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM
    from cleandiffuser.nn_condition import (
        DirectSpatialHistoryCondition, MultiImageObsConditionRobomimic)
    from cleandiffuser.nn_diffusion.longmem_chitransformerptp import (
        LongMemChiTransformerPTP)

    method = str(args.history_method)
    if method not in METHODS:
        raise ValueError(f"history_method must be one of {sorted(METHODS)}")

    condition_kwargs = dict(
        shape_meta=args.shape_meta, emb_dim=256,
        rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
        crop_shape=args.crop_shape, random_crop=args.random_crop,
        use_group_norm=args.use_group_norm, use_seq=args.use_seq,
        keep_horizon_dims=True)
    if method.startswith("direct"):
        output_tokens = 2 if method == "direct_token2" else 0
        nn_condition = DirectSpatialHistoryCondition(
            **condition_kwargs, pool_size=args.history_pool_size,
            history_frames=args.dataset_maximum_memory_length,
            history_output_tokens=output_tokens,
            history_subsample_ratio=args.dataset_memory_subsample_ratio,
        ).to(args.device)
        memory_tokens = nn_condition.history_token_count
    else:
        nn_condition = MultiImageObsConditionRobomimic(
            **condition_kwargs).to(args.device)
        memory_tokens = 2 if method == "vpwem" else 0

    nn_diffusion = LongMemChiTransformerPTP(
        args.action_dim, 256, args.horizon, args.obs_steps, memory_tokens,
        d_model=256, nhead=4, num_layers=args.diffusion_num_layers,
        p_drop_emb=args.p_drop_emb, p_drop_attn=args.p_drop_attn,
        timestep_emb_type="positional").to(args.device)

    nn_memory = None
    if method == "vpwem":
        from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
        nn_memory = QFormerMemory(
            mem_compress_length=2, emb_dim=256, num_layers=2,
            cache_compress="fifo", cache_max_length=8,
            use_pos_emb="sin_pos",
            subsample_ratio=args.dataset_memory_subsample_ratio,
        ).to(args.device)

    x_max = torch.ones(
        (1, args.horizon, args.action_dim), device=args.device)
    agent = LongMemDDPM(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition,
        nn_memory=nn_memory, device=args.device,
        diffusion_steps=args.sample_steps, x_max=x_max, x_min=-x_max,
        args_dict=args,
        action_range=normalizer["action"].range.to(args.device),
        action_min=normalizer["action"].min.to(args.device),
        short_cond_dropout=args.short_cond_dropout,
        long_cond_dropout=args.long_cond_dropout,
        optim_params={"lr": args.lr})
    return agent


def split_batch(args, dataset, batch):
    condition, memory = {}, {}
    for key, value in batch["obs"].items():
        if key.endswith("_mem"):
            base_key = key[:-4]
            memory[base_key] = dataset.normalizer["obs"][base_key].normalize(
                value.to(args.device))
        elif key not in {"ep_step", "actual_num_frames"}:
            condition[key] = dataset.normalizer["obs"][key].normalize(
                value[:, :args.obs_steps].to(args.device))
    action = dataset.normalizer["action"].normalize(
        batch["action"].to(args.device))
    steps = batch["obs"]["ep_step"].int().to(args.device)
    if str(args.history_method) == "no_history":
        memory = None
        steps = None
    return condition, memory, action, steps


def train(args):
    dataset = make_dataset(args, load_images=True)
    sampler = dataset.get_dataset_sampler()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=False, drop_last=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        collate_fn=dynamic_batch_collate_fn)
    agent = build_agent(args, dataset.normalizer)
    report_parameters(agent.model["diffusion"])
    report_parameters(agent.model["condition"])
    scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    step = auto_resume_training(
        agent, scheduler, args.work_dir, args.gradient_steps)
    logger = Logger(pathlib.Path(args.work_dir), args)
    losses, started = [], time.time()
    for batch in loop_dataloader(loader):
        condition, memory, action, ep_step = split_batch(args, dataset, batch)
        log = agent.update(action, condition, memory, ep_step)
        scheduler.step()
        losses.append(float(log["loss"]))
        if step % args.log_freq == 0:
            logger.log({"step": step, "total_time": time.time() - started,
                        "avg_loss": float(np.mean(losses))}, category="train")
            losses.clear()
        if step > 0 and step % args.save_freq == 0:
            logger.save_agent(agent=agent, identifier=step)
            torch.save(scheduler.state_dict(), os.path.join(
                args.work_dir, "models", f"scheduler_{step}.pt"))
            prune_checkpoints(args.work_dir, keep=args.keep_checkpoints)
        step += 1
        if step > args.gradient_steps:
            break


def make_env(args):
    import gymnasium as gym
    import mikasa_robo_suite  # noqa: F401
    from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info
    from cleandiffuser.env.gymnasium_wrapper import (
        FlattenRGBDObservationWrapper, MultiStepWrapper)

    env = gym.make(
        args.env_name, num_envs=args.num_envs, obs_mode="rgb",
        render_mode="all", control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense", sim_backend="gpu")
    wrappers, timeout = env_info(args.env_name)
    for wrapper, kwargs in wrappers:
        env = wrapper(env, **kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, joints=True)
    env = MultiStepWrapper(
        env, n_obs_steps=args.action_steps,
        n_action_steps=args.inference_action_steps,
        max_episode_steps=timeout)
    return env, int(timeout)


def evaluate(args):
    dataset = make_dataset(args, load_images=False)
    agent = build_agent(args, dataset.normalizer)
    ckpt = int(args.ckpt)
    agent.load(os.path.join(args.work_dir, "models", f"model_{ckpt}.pt"))
    agent.model.eval()
    agent.model_ema.eval()
    env, timeout = make_env(args)
    rewards, successes, steps, infer_times = [], [], [], []
    try:
        for rollout in range(args.eval_episodes // args.num_envs):
            obs, _ = env.reset(
                seed=args.eval_seed_start + rollout * args.num_envs)
            agent.reset_cache()
            # reset_cache clears observation buffers, not Q-Former state.
            # Each new episode batch must also reset both memory modules.
            for model in (agent.model, agent.model_ema):
                if model["memory"] is not None:
                    model["memory"].reset()
            ep_reward = np.zeros(args.num_envs, dtype=np.float32)
            success = np.zeros(args.num_envs, dtype=bool)
            t = 0
            while t < timeout:
                condition = {}
                for key, value in obs.items():
                    tensor = torch.as_tensor(
                        value, dtype=torch.float32, device=args.device)
                    if str(args.history_method) == "no_history":
                        tensor = tensor[:, -args.obs_steps:]
                    condition[key] = dataset.normalizer["obs"][key].normalize(
                        tensor)
                if str(args.history_method) == "no_history":
                    agent.reset_cache()
                prior = torch.zeros(
                    args.num_envs, args.horizon, args.action_dim,
                    device=args.device)
                start_time = time.time()
                with torch.no_grad():
                    normalized_action, _ = agent.sample(
                        prior=prior, n_samples=args.num_envs,
                        sample_steps=args.sample_steps, condition_cfg=condition,
                        w_cfg=1.0, temperature=args.temperature, use_ema=True)
                infer_times.append(time.time() - start_time)
                action_pred = dataset.normalizer["action"].unnormalize(
                    normalized_action).detach().cpu().numpy()
                start = args.obs_steps - 1
                chunk = min(args.inference_action_steps, timeout - t)
                action = action_pred[:, start:start + chunk]
                obs, reward, done, info = env.step(action)
                ep_reward += np.asarray(reward, dtype=np.float32)
                success |= np.asarray([
                    any(info["success"][j]) for j in range(args.num_envs)])
                t += chunk
                if np.all(done):
                    break
            rewards.extend(ep_reward.tolist())
            successes.extend(success.tolist())
            steps.extend([t] * args.num_envs)
            print(f"episodes={len(successes)} success={success.tolist()}",
                  flush=True)
    finally:
        env.close()
    result = {
        "history_method": str(args.history_method),
        "mean_reward": float(np.mean(rewards)),
        "mean_success": float(np.mean(successes)),
        "mean_step": float(np.mean(steps)),
        "avg_infer_time": float(np.mean(infer_times)),
        "eval_seed_start": int(args.eval_seed_start),
        "episode_reward": np.asarray(rewards, dtype=np.float32),
        "episode_success": np.asarray(successes, dtype=np.int8),
        "episode_steps": np.asarray(steps, dtype=np.int32),
    }
    path = os.path.join(args.work_dir, f"result_{ckpt}_paired.npy")
    np.save(path, result)
    print(result, flush=True)


@hydra.main(
    config_path="../configs/dp/mikasa/fair_history",
    config_name="base", version_base=None)
def pipeline(args):
    set_seed(args.seed)
    assert args.horizon == args.obs_steps + args.action_steps - 1
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        raise ValueError("mode must be train or eval")


if __name__ == "__main__":
    pipeline()
