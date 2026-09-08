"""MIKASA 上的 VPWEM 在线 rollout / 评估入口。

与训练入口相比，本文件最值得关注的是 episode 级状态管理：环境每次返回一个
观测 chunk，``LongMemDDPM.sample`` 保留最近 ``obs_steps`` 帧，并把更早的帧
增量写入 ``nn_memory``。episode reset 时必须同时清空 agent 的近期缓存和
训练/EMA 两份 memory 状态。
"""

import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import gym
import pathlib
import time
import collections
import numpy as np
import torch
import pickle
import torch.nn as nn
from utils import set_seed, parse_cfg, Logger
from torch.optim.lr_scheduler import CosineAnnealingLR

from cleandiffuser.dataset.mikasa_dataset_longmem import LongMemMikasaDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
from cleandiffuser.utils.mlp_correlation import batch_mlp_corr
from cleandiffuser.env.gymnasium_wrapper import MultiStepWrapper, FlattenRGBDObservationWrapper
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

import mikasa_robo_suite
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info
import gymnasium as gym

from PIL import Image


def make_async_envs(args):
    
    print(f"Starting to create {args.num_envs} asynchronous Mikasa environments...")

    env_name = args.env_name
    obs_mode = "rgb"
    num_envs = args.num_envs

    env = gym.make(
        env_name, num_envs=num_envs, obs_mode=obs_mode, render_mode="all",
        control_mode="pd_joint_delta_pos", reward_mode="normalized_dense",
        sim_backend="gpu")
    state_wrappers_list, episode_timeout = env_info(env_name)
    print(f"Episode timeout: {episode_timeout}")
    for wrapper_class, wrapper_kwargs in state_wrappers_list:
        env = wrapper_class(env, **wrapper_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, joints=True)
    # Return the observations accumulated across an action chunk. LongMemDDPM
    # moves the older frames into its online cache and passes only the final
    # obs_steps frames to the decoder, so the complete current input remains
    # identical to training while no intervening evidence is dropped.
    env = MultiStepWrapper(env, n_obs_steps=args.action_steps,
                            n_action_steps=args.inference_action_steps,
                            max_episode_steps=args.max_episode_steps)
    return env
    
def inference(args, envs, normalizer, agent, video_path):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []
    predictability_ratio = []
    avg_infer_times = []
    all_all_actions = []
    
    if args.diffusion == "ddpm":
        solver = None
    elif args.diffusion == "ddim":
        solver = "ddim"
    elif args.diffusion == "dpm":   
        solver = "ode_dpmpp_2"
    elif args.diffusion == "edm":
        solver = "euler"
    
    for i in range(args.eval_episodes // args.num_envs): 

        ep_reward = np.zeros(args.num_envs, dtype=np.float32)
        # Fixed, non-overlapping reset seeds make checkpoint and method
        # comparisons episode-paired while preserving distinct trials.
        reset_seed = int(args.seed) + i * int(args.num_envs)
        obs, _ = envs.reset(seed=reset_seed)
        t = 0
        all_actions = []
        all_done = False
        success = [False] * args.num_envs
        previous_action_normalized = None
        
        # 三处状态都要清空：agent 内的 working-memory 列表，以及普通模型和
        # EMA 模型各自的 episodic-memory cache。漏掉任一处都会跨 episode 泄漏。
        agent.reset_cache()
        agent.model["memory"].reset()
        agent.model_ema["memory"].reset()

        while t < args.max_episode_steps:
            
            t0 = time.time()
            obs_dict = {}
            for k in obs.keys():
                obs_seq = obs[k] # .astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                # Convert to torch tensor if not already (normalizer may return numpy or torch)
                if isinstance(obs_seq, np.ndarray):
                    obs_seq = torch.from_numpy(obs_seq).to(args.device)
                else:
                    obs_seq = obs_seq.to(args.device)
                nobs = normalizer['obs'][k].normalize(obs_seq)
                obs_dict[k] = nobs  # (num_envs, obs_steps, obs_dim)
            
            with torch.no_grad():
                condition = obs_dict
                # run sampling (num_envs, horizon, action_dim)
                prior = torch.zeros((args.num_envs, args.horizon, args.action_dim), device=args.device)
                naction, _ = agent.sample(
                    prior=prior, n_samples=args.num_envs,
                    sample_steps=args.sample_steps, solver=solver,
                    condition_cfg=condition,
                    action_context_cfg=previous_action_normalized,
                    w_cfg=1.0, temperature=args.temperature, use_ema=True)
            
            # unnormalize prediction
            naction = naction.detach()  # (num_envs, horizon, action_dim)
            action_pred = normalizer['action'].unnormalize(naction)
            # Convert to numpy and move to CPU
            if isinstance(action_pred, torch.Tensor):
                action_pred = action_pred.detach().cpu().numpy()
            else:
                action_pred = np.array(action_pred)  
            
            # PTP 的前 obs_steps-1 个位置用于对齐过去动作；真正交给环境的是
            # 从 start 开始的未来 action chunk，而不是预测张量的整个 horizon。
            start = args.obs_steps - 1
            end = start + args.inference_action_steps
            previous_action_normalized = naction[:, start:end, :].detach()
            action = action_pred[:, start:end, :]
            all_actions.append(action)

            t1 = time.time()
            avg_infer_times.append(t1 - t0)

            if args.save_video:
                for j in range(args.num_envs):
                    img = obs['rgb'][j][-1, :, :, :]
                    img = (img * 255).astype(np.uint8)
                    img = img.transpose(1, 2, 0)
                    img = Image.fromarray(img)
                    img.save(os.path.join(
                        video_path, f"loop_{i}_env_{j}_step_{t}.jpg"))

            obs, reward, done, info = envs.step(action)
            success_per_env = [any(info['success'][j]) for j in range(args.num_envs)]
            success = [success[j] or success_per_env[j] for j in range(args.num_envs)]
            
            ep_reward += np.asarray(reward, dtype=np.float32)
            t += args.inference_action_steps
            if np.all(done):
                break

        print(f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] reward: {np.around(ep_reward, 2)} success:{success}")
        episode_rewards.extend(ep_reward.tolist())
        episode_steps.extend([t] * args.num_envs)
        episode_success.extend(success)

        all_actions = np.transpose(all_actions, (1,0,2,3))
        B, D1, D2, C = all_actions.shape  # Get dimensions dynamically
        all_actions = all_actions.reshape(B, D1 * D2, C)
        all_all_actions.append(all_actions)
    
    all_all_actions = np.concatenate(all_all_actions, axis=0)
    predictability_ratio.append(batch_mlp_corr(all_all_actions))

    log_dict = {'mean_step': np.nanmean(episode_steps), 
                'mean_reward': np.nanmean(episode_rewards), 
                'mean_success': np.nanmean(episode_success),
                "mlp_corr_pred_actions_full_traj_online_fixed": np.nanmean(predictability_ratio),
                "avg_infer_time": np.mean(avg_infer_times),
                "eval_seed": int(args.seed),
                "episode_id": np.arange(
                    len(episode_success), dtype=np.int64),
                "episode_reward": np.asarray(
                    episode_rewards, dtype=np.float32),
                "episode_success": np.asarray(
                    episode_success, dtype=np.int8),
                "episode_steps": np.asarray(
                    episode_steps, dtype=np.int32),
                }
    
    print(log_dict, flush=True)

    return log_dict


@hydra.main(config_path="../configs/dp/mikasa/chi_transformer_ptp_longmem", config_name="remember_color_5")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    assert args.horizon == args.obs_steps + args.action_steps - 1
    
    # ---------------- Create Environments ----------------
    envs = make_async_envs(args)

    # ---------------- Create Dataset ----------------
   
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = LongMemMikasaDataset(dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
                                    n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
                                    pad_after=args.action_steps-1,
                                    load_images=False, device=args.device)
    print(dataset)
    normalizer = dataset.normalizer
    normalizer_dataset_path = getattr(args, "normalizer_dataset_path", None)
    if normalizer_dataset_path:
        normalizer_dataset = LongMemMikasaDataset(
            os.path.expanduser(str(normalizer_dataset_path)),
            horizon=args.horizon, shape_meta=args.shape_meta,
            n_obs_steps=args.obs_steps, pad_before=args.obs_steps - 1,
            pad_after=args.action_steps - 1, load_images=False,
            device=args.device)
        normalizer = normalizer_dataset.normalizer
        print(
            f"Using frozen-checkpoint normalizer from "
            f"{normalizer_dataset_path}")

    # Training uses the normalizer constructed by LongMemMikasaDataset. Reuse
    # that same normalizer for evaluation instead of requiring a separate
    # normalizer.pkl artifact that the training pipeline never creates.
        
    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
        from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP
        
        nn_condition = MultiImageObsConditionRobomimic(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
        nn_diffusion = LongMemChiTransformerPTP(
            args.action_dim, 256, args.horizon, args.obs_steps, 
            args.mem_compress_length,
            d_model=256, nhead=4, num_layers=args.diffusion_num_layers,
            p_drop_emb=args.p_drop_emb,
            p_drop_attn=args.p_drop_attn,
            memory_read_mode=str(getattr(
                args, "memory_read_mode", "context")),
            memory_gate_bias=float(getattr(
                args, "memory_gate_bias", 0.0)),
            timestep_emb_type="positional").to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")
    
    from cleandiffuser.nn_memory.random_memory import RandomMemory
    from cleandiffuser.nn_memory.kmeans_memory import KMeansMemory
    from cleandiffuser.nn_memory.adjsim_memory import AdjSimMemory
    from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
    from cleandiffuser.nn_memory.per_memory import PERMemory
    if args.mem_compress_method == "random":
        nn_memory = RandomMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "kmeans":
        nn_memory = KMeansMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "adjsim":
        nn_memory = AdjSimMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "qformer":
        nn_memory = QFormerMemory(mem_compress_length=args.mem_compress_length,
                                        emb_dim=args.qformer_emb_dim,
                                        num_layers=args.qformer_num_layers,
                                        cache_compress=args.qformer_cache_compress,
                                        cache_max_length=args.qformer_cache_max_length,
                                        use_pos_emb=args.memory_pos_emb,
                                        subsample_ratio=args.dataset_memory_subsample_ratio,
                                        ).to(args.device)
    elif args.mem_compress_method == "per":
        nn_memory = PERMemory(
            mem_compress_length=args.mem_compress_length,
            evidence_length=int(getattr(args, "per_evidence_length", 0)),
            emb_dim=args.qformer_emb_dim,
            num_layers=args.qformer_num_layers,
            num_heads=int(getattr(args, "qformer_num_heads", 8)),
            bank_size=args.qformer_cache_max_length,
            event_score_mode=str(getattr(args, "per_event_score", "delta")),
            event_threshold=float(getattr(args, "per_event_threshold", 0.05)),
            max_write_interval=int(getattr(args, "per_max_write_interval", 20)),
            merge_similarity=float(getattr(args, "per_merge_similarity", 0.98)),
            score_decay=float(getattr(args, "per_score_decay", 0.995)),
            pos_emb_scale=float(getattr(args, "per_pos_emb_scale", 1.0)),
            subsample_ratio=args.dataset_memory_subsample_ratio,
        ).to(args.device)
    elif args.mem_compress_method in {
            "aem_style", "dssp_style", "vq_style", "cycle_progress"}:
        from cleandiffuser.nn_memory.paper_transfer_factory import (
            build_transfer_memory)
        nn_memory = build_transfer_memory(args).to(args.device)
    else:
        raise ValueError(f"Invalid memory method {args.mem_compress_method}")
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Memory Model =======================")
    report_parameters(nn_memory)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print(f"===================================================================================")

    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
    x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0
    # Get normalizer parameters (already on device)
    action_range = normalizer['action'].range.to(args.device)
    action_min = normalizer['action'].min.to(args.device)
    
    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM
        agent = LongMemDDPM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, nn_memory=nn_memory,
            device=args.device,
            diffusion_steps=args.sample_steps, x_max=x_max, x_min=x_min,
            args_dict=args,
            action_range=action_range,
            action_min=action_min,
            short_cond_dropout=args.short_cond_dropout,
            long_cond_dropout=args.long_cond_dropout,
            optim_params={"lr": args.lr},
            )
    else:
        raise NotImplementedError(f"Diffusion method {args.diffusion} not supported for LongMem")
    
    # Keep the historical default while allowing shorter experiments to select
    # the checkpoint they actually trained (for example, +ckpt=200000).
    ckpt = int(args.get("ckpt", 600000))
    print(ckpt)
    configured_model_path = str(getattr(args, "model_path", ""))
    if configured_model_path.lower() not in {"", "false", "none"}:
        model_path = os.path.expanduser(configured_model_path)
    else:
        model_path = os.path.join(args.work_dir, f"models/model_{ckpt}.pt")
    agent.load(model_path)
    video_path = os.path.join(args.work_dir, f"videos/{ckpt}_{args.inference_action_steps}")
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    agent.model.eval()
    agent.model_ema.eval()

    try:
        result = inference(args, envs, normalizer, agent, video_path)
    finally:
        envs.close()
    # save results
    configured_result_path = str(getattr(args, "result_path", ""))
    if configured_result_path.lower() not in {"", "false", "none"}:
        result_path = os.path.expanduser(configured_result_path)
    else:
        result_path = os.path.join(
            args.work_dir, f"result_{ckpt}_{args.inference_action_steps}.npy")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    np.save(result_path, result)


if __name__ == "__main__":
    pipeline()
