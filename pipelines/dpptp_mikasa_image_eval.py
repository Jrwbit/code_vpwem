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

from cleandiffuser.dataset.mikasa_dataset import MikasaDataset
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


def make_async_envs(args):
    
    print(f"Starting to create {args.num_envs} asynchronous Mikasa environments...")

    env_name = args.env_name
    obs_mode = "rgb"
    num_envs = args.num_envs

    env = gym.make(env_name, num_envs=num_envs, obs_mode=obs_mode, render_mode="all")
    state_wrappers_list, episode_timeout = env_info(env_name)
    print(f"Episode timeout: {episode_timeout}")
    for wrapper_class, wrapper_kwargs in state_wrappers_list:
        env = wrapper_class(env, **wrapper_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, joints=True)
    env = MultiStepWrapper(env, n_obs_steps=args.obs_steps,
                            n_action_steps=args.inference_action_steps,
                            max_episode_steps=episode_timeout)
    env.benchmark_episode_timeout = episode_timeout
    return env
    
def inference(args, envs, normalizer, agent):
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
        eval_seed_start = int(getattr(args, "eval_seed_start", 1))
        # Gymnasium vector environments derive consecutive worker seeds from
        # the reset seed, yielding the paper protocol seeds 1..100.
        obs, _ = envs.reset(seed=eval_seed_start + i * args.num_envs)
        t = 0
        all_actions = []
        all_done = False
        success = [False] * args.num_envs

        episode_timeout = int(getattr(
            envs, "benchmark_episode_timeout", args.max_episode_steps))
        while t < episode_timeout:
            
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
                # Some normalizers keep their own CPU device and may move a
                # CUDA input back to CPU. The condition encoder is on
                # args.device, so enforce the device at this boundary.
                obs_dict[k] = torch.as_tensor(
                    nobs, dtype=torch.float32, device=args.device)
            
            with torch.no_grad():
                condition = obs_dict
                # run sampling (num_envs, horizon, action_dim)
                prior = torch.zeros((args.num_envs, args.horizon, args.action_dim), device=args.device)
                naction, _ = agent.sample(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps, solver=solver, condition_cfg=condition, w_cfg=1.0, temperature=args.temperature, use_ema=True)
            
            # unnormalize prediction
            naction = naction.detach()  # (num_envs, horizon, action_dim)
            action_pred = normalizer['action'].unnormalize(naction)
            # Convert to numpy and move to CPU
            if isinstance(action_pred, torch.Tensor):
                action_pred = action_pred.detach().cpu().numpy()
            else:
                action_pred = np.array(action_pred)

            # get action
            start = args.obs_steps - 1
            chunk_steps = min(
                args.inference_action_steps, episode_timeout - t)
            end = start + chunk_steps
            action = action_pred[:, start:end, :]
            all_actions.append(action)

            t1 = time.time()
            avg_infer_times.append(t1 - t0)

            obs, reward, done, info = envs.step(action)
            success_per_env = [any(info['success'][j]) for j in range(args.num_envs)]
            success = [success[j] or success_per_env[j] for j in range(args.num_envs)]
            
            ep_reward += reward
            t += chunk_steps
            if np.all(done):
                break

        print(f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] reward: {np.around(ep_reward, 2)} success:{success}")
        episode_rewards.extend(ep_reward.tolist())
        episode_steps.extend([t] * args.num_envs)
        episode_success.extend(success)

        all_actions = np.concatenate(all_actions, axis=1)
        all_all_actions.append(all_actions)
    
    all_all_actions = np.concatenate(all_all_actions, axis=0)
    predictability_ratio.append(batch_mlp_corr(all_all_actions))

    log_dict = {'mean_step': np.nanmean(episode_steps), 
                'mean_reward': np.nanmean(episode_rewards), 
                'mean_success': np.nanmean(episode_success),
                "mlp_corr_pred_actions_full_traj_online_fixed": np.nanmean(predictability_ratio),
                "avg_infer_time": np.mean(avg_infer_times),
                }
    
    print(log_dict, flush=True)

    return log_dict


@hydra.main(config_path="../configs/dp/mikasa/chi_transformer_ptp", config_name="shell_game_touch")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    assert args.horizon == args.obs_steps + args.action_steps - 1
    # ---------------- Create Environments ----------------
    envs = make_async_envs(args)

    # modality_mapping = collections.defaultdict(list)
    # for key, attr in args.shape_meta['obs'].items():
    #     modality_mapping[attr.get('type', 'low_dim')].append(key)
    # ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        
    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = MikasaDataset(dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
                                    n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
                                    pad_after=args.action_steps-1,
                                    use_full_episodes=bool(getattr(
                                        args, "dataset_use_full_episodes", False)))
    print(dataset)
    normalizer = dataset.normalizer
    
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     shuffle=True,
    #     pin_memory=True,
    #     persistent_workers=True
    # )
        
    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
        from cleandiffuser.nn_diffusion.chitransformerptp import ChiTransformerPTP
        
        nn_condition = MultiImageObsConditionRobomimic(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
        nn_diffusion = ChiTransformerPTP(
            args.action_dim, 256, args.horizon, args.obs_steps, d_model=256, nhead=4, num_layers=args.diffusion_num_layers,
            timestep_emb_type="positional").to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print(f"===================================================================================")

    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
    x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0
    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.ddpm import DDPM
        agent = DDPM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
            diffusion_steps=args.sample_steps, x_max=x_max, x_min=x_min,
            optim_params={"lr": args.lr})
    elif args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM
        agent = EDM(nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
                    optim_params={"lr": args.lr})
    else:
        raise NotImplementedError
    
    ckpt = int(args.get("ckpt", 600000))
    print(ckpt)
    configured_model_path = str(getattr(args, "model_path", ""))
    if configured_model_path.lower() not in {"", "false", "none"}:
        model_path = os.path.expanduser(configured_model_path)
    else:
        model_path = os.path.join(args.work_dir, f"models/model_{ckpt}.pt")
    agent.load(model_path)
    
    agent.model.eval()
    agent.model_ema.eval()

    result = inference(args, envs, normalizer, agent)
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
