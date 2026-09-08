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
import torch.nn as nn
from utils import set_seed, parse_cfg, Logger, auto_resume_training, prune_checkpoints
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing

# from cleandiffuser.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
# from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
# from cleandiffuser.env.async_vector_env import AsyncVectorEnv
# from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.robomimic_dataset_longmem import LongMemRobomimicImageDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader, dynamic_batch_collate_fn
from cleandiffuser.utils import report_parameters
# import robomimic.utils.train_utils as TrainUtils
# import robomimic.utils.file_utils as FileUtils
# import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import cleandiffuser.dataset.dataset_utils as robomimic_dataset_utils

@hydra.main(config_path="../configs/dp/robomimic_multi_modal/chi_transformer_ptp_longmem", config_name="longsquare_abs_emb")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)
    assert args.horizon == args.obs_steps + args.action_steps - 1

    # ---------------- Create Environment ----------------
    # envs = make_async_envs(args)

    modality_mapping = collections.defaultdict(list)
    for key, attr in args.shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        
    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = LongMemRobomimicImageDataset(dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
                                    n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
                                    pad_after=args.action_steps-1, abs_action=args.abs_action,
                                    dataset_memory_type=args.dataset_memory_type,
                                    subsample_ratio=args.dataset_memory_subsample_ratio,
                                    maximum_memory_length=int(getattr(
                                        args, "dataset_maximum_memory_length", 16)),
                                    preserve_episode_anchor=bool(getattr(
                                        args, "dataset_preserve_episode_anchor", False)),
                                    device=args.device,)
    print(dataset)
    
    # Get dataset sampler for dynamic batching
    train_sampler = dataset.get_dataset_sampler()
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=int(args.get("num_workers", 8)),
        shuffle=train_sampler is None,  # Only shuffle if no custom sampler
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dynamic_batch_collate_fn,
    )
    
    # Move normalizer to GPU after DataLoader creation to allow fork-based multiprocessing
    # This ensures GPU tensors are not created before DataLoader workers are forked
    
    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
        from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP
        
        # nn_condition = MultiImageObsConditionRobomimic(
        #     shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
        #     crop_shape=args.crop_shape, random_crop=args.random_crop, 
        #     use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
        nn_diffusion = LongMemChiTransformerPTP(
            args.action_dim, 256, args.horizon, args.obs_steps, 
            args.mem_compress_length,
            d_model=256, nhead=4, num_layers=args.diffusion_num_layers,
            p_drop_emb=args.p_drop_emb,
            p_drop_attn=args.p_drop_attn,
            timestep_emb_type="positional").to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")
    
    from cleandiffuser.nn_memory.random_memory import RandomMemory
    from cleandiffuser.nn_memory.kmeans_memory import KMeansMemory
    from cleandiffuser.nn_memory.adjsim_memory import AdjSimMemory
    from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
    from cleandiffuser.nn_memory.per_memory import PERMemory
    from cleandiffuser.nn_memory.learned_retention_memory import LearnedRetentionMemory
    from cleandiffuser.nn_memory.recent_memory import RecentMemory
    from cleandiffuser.nn_memory.anchor_recent_memory import AnchorRecentMemory
    if args.mem_compress_method == "random":
        nn_memory = RandomMemory(
            mem_compress_length=args.mem_compress_length,
            retention_seed=int(getattr(args, "retention_seed", args.seed)),
        ).to(args.device)
    elif args.mem_compress_method == "recent":
        nn_memory = RecentMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "anchor_recent":
        nn_memory = AnchorRecentMemory(mem_compress_length=args.mem_compress_length).to(args.device)
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
            evidence_length=int(getattr(args, "per_evidence_length", 1)),
            emb_dim=args.qformer_emb_dim,
            num_layers=args.qformer_num_layers,
            num_heads=int(getattr(args, "qformer_num_heads", 8)),
            bank_size=args.qformer_cache_max_length,
            event_score_mode=str(getattr(args, "per_event_score", "delta")),
            event_threshold=float(getattr(args, "per_event_threshold", 0.05)),
            max_write_interval=int(getattr(
                args, "per_max_write_interval",
                args.dataset_memory_subsample_ratio)),
            merge_similarity=float(getattr(
                args, "per_merge_similarity", 0.98)),
            score_decay=float(getattr(args, "per_score_decay", 0.995)),
            pos_emb_scale=float(getattr(args, "per_pos_emb_scale", 0.1)),
            subsample_ratio=args.dataset_memory_subsample_ratio,
        ).to(args.device)
    elif args.mem_compress_method == "learned_retention":
        nn_memory = LearnedRetentionMemory(
            mem_compress_length=args.mem_compress_length,
            emb_dim=args.qformer_emb_dim,
            num_layers=args.qformer_num_layers,
            num_heads=int(getattr(args, "qformer_num_heads", 8)),
            bank_size=args.qformer_cache_max_length,
            subsample_ratio=args.dataset_memory_subsample_ratio,
            pos_emb_scale=float(getattr(args, "retention_pos_emb_scale", 1.0)),
            retention_head_path=str(getattr(args, "retention_head_path", "")),
        ).to(args.device)
    else:
        raise ValueError(f"Invalid memory method {args.mem_compress_method}")
    
    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM
        x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
        x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0
        # Get normalizer parameters (already on device)
        action_range = dataset.normalizer['action'].range.to(args.device)
        action_min = dataset.normalizer['action'].min.to(args.device)
        agent = LongMemDDPM(
            nn_diffusion=nn_diffusion, nn_condition=None, nn_memory=nn_memory,
            device=args.device,
            diffusion_steps=args.sample_steps, x_max=x_max, x_min=x_min,
            args_dict=args,
            action_range=action_range,
            action_min=action_min,
            short_cond_dropout=args.short_cond_dropout,
            long_cond_dropout=args.long_cond_dropout,
            optim_params={"lr": args.lr})
        
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Memory Model =======================")
    report_parameters(nn_memory)
    print(f"===================================================================================")

    init_model_path = getattr(args, "init_model_path", None)
    if init_model_path:
        init_checkpoint = torch.load(
            os.path.expanduser(str(init_model_path)),
            map_location=args.device, weights_only=False)
        model_msg = agent.model.load_state_dict(
            init_checkpoint["model"], strict=False)
        ema_msg = agent.model_ema.load_state_dict(
            init_checkpoint["model_ema"], strict=False)
        print(f"Initialization model loaded with message: {model_msg}")
        print(f"Initialization EMA loaded with message: {ema_msg}")
    
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    n_gradient_step = auto_resume_training(agent, lr_scheduler, args.work_dir, args.gradient_steps)
    
    if args.mode == "train":
        if n_gradient_step > args.gradient_steps:
            print(f"Training already complete at step {n_gradient_step - 1}.", flush=True)
            logger.finish()
            return
        # ----------------- Training ----------------------
        train_metric_lists = collections.defaultdict(list)
        start_time = time.time()
        
        for batch in loop_dataloader(dataloader):
            # get condition
            nobs = batch['obs']
            # condition = {}
            # for k in nobs.keys():
            condition = nobs['dp_emb'][:, :args.obs_steps, :].to(args.device)
            # Convert actions to torch tensor and normalize (normalizer handles device)
            naction = batch['action']  # Already torch tensor from dataloader
            if isinstance(naction, np.ndarray):
                naction = torch.from_numpy(naction).to(args.device)
            else:
                naction = naction.to(args.device)
            naction = dataset.normalizer['action'].normalize(naction)  # Returns torch tensor on device
            memory = nobs['dp_emb_mem'].to(args.device)
            mem_steps = nobs['ep_step'].to(args.device)

            # update diffusion
            update_log = agent.update(
                naction, condition, memory, mem_steps)
            lr_scheduler.step()
            for key, value in update_log.items():
                if isinstance(value, (int, float, np.number)):
                    train_metric_lists[key].append(float(value))

            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    'step': n_gradient_step,
                    'total_time': time.time() - start_time,
                }
                for key, values in train_metric_lists.items():
                    if values:
                        metrics[f'avg_{key}'] = np.mean(values)
                # Keep the historical metric name for existing dashboards.
                if 'avg_loss' in metrics:
                    metrics['avg_diffusion_loss'] = metrics['avg_loss']
                logger.log(metrics, category='train')
                train_metric_lists.clear()
            
            if n_gradient_step % args.save_freq == 0 and n_gradient_step > 0:
                logger.save_agent(agent=agent, identifier=n_gradient_step)
                torch.save(lr_scheduler.state_dict(), args.work_dir + f'/models/scheduler_{str(n_gradient_step)}.pt')
                prune_checkpoints(args.work_dir, keep=int(args.get("keep_checkpoints", 3)))
            
            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                # finish
                logger.finish()
                break
        
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()
