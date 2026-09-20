import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['MESA_GL_VERSION_OVERRIDE']='4.5'
os.environ['MESA_GLSL_VERSION_OVERRIDE']='450'
os.environ['MESA_GLES_VERSION_OVERRIDE']='4.5'

import gym
import pathlib
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import pickle
from utils import (
    set_seed, parse_cfg, Logger, auto_resume_training, prune_checkpoints)
from cleandiffuser.dataset.multitask import BalancedTaskBatchMux
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing
from collections import defaultdict

# from cleandiffuser.env.robomimic.momart_image_wrapper import MomartImageWrapper
# from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
# from cleandiffuser.env.async_vector_env import AsyncVectorEnv
# from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils.utils import report_parameters, dict_apply 
from cleandiffuser.utils.mlp_correlation import batch_mlp_corr
from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer, ImageNormalizer
import cleandiffuser.utils.robomimic_train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.tensor_utils import to_tensor, to_device
import cleandiffuser.utils.robomimic_dataset_utils as momart_dataset_utils

# torch.autograd.set_detect_anomaly(True)

class DictToObj: 
    def __init__(self, dict_): 
        for k, v in dict_.items(): 
            if isinstance(v, dict): 
                v = DictToObj(v) 
            setattr(self, k, v)


@hydra.main(config_path="../configs/dp/momart/chi_transformer_ptp_longmem", config_name="momart_emb")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)
    assert args.horizon == args.obs_steps + args.action_steps - 1

    modality_mapping = collections.defaultdict(list)
    for key, attr in args.shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        
    # ---------------- Create Dataset ----------------

    task_specs = getattr(args, "task_datasets", None)
    if task_specs:
        task_paths = {
            str(spec.name): os.path.expanduser(str(spec.path))
            for spec in task_specs
        }
    else:
        task_paths = {
            f"{args.env_name}_{args.env_type}": os.path.expanduser(
                str(args.dataset_path))
        }

    def make_dataset(dataset_path):
        dataset_config_dict = {
        "train": {
            "data": dataset_path,
            "num_data_workers": 8,
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": True,
            "hdf5_load_next_obs": False,
            "hdf5_normalize_obs": False,
            "hdf5_filter_key": "train",
            "hdf5_validation_filter_key": "valid",
            "seq_length": args.horizon,
            "pad_seq_length": True,
            "frame_stack": 1,
            "pad_frame_stack": True,
            "dataset_keys": [
                "actions",
                "rewards",
                "dones"
            ],
            "goal_mode": None,
            "cuda": True,
            "batch_size": args.batch_size,
        },
        "experiment": {
            "validate": False
        },
        "memory": {
            "dataset_memory_type": args.dataset_memory_type,
            "dataset_memory_subsample_ratio": args.dataset_memory_subsample_ratio,
        }
        }
        dataset_config = DictToObj(dataset_config_dict)
        dataset, _ = TrainUtils.load_data_for_training(
            dataset_config, obs_keys=args.shape_meta["obs"])

        normalizer = defaultdict(dict)
        action_max = torch.full(
            (args.shape_meta["action"].shape[0],), float('-inf'),
            device=args.device, dtype=torch.float32)
        action_min = torch.full(
            (args.shape_meta["action"].shape[0],), float('inf'),
            device=args.device, dtype=torch.float32)
        for demo in dataset.hdf5_cache.values():
            actions = torch.from_numpy(
                demo['actions'].astype(np.float32)).to(args.device)
            action_max = torch.maximum(actions.max(dim=0).values, action_max)
            action_min = torch.minimum(actions.min(dim=0).values, action_min)
        action_stats = torch.stack([action_min, action_max], dim=0)
        normalizer['action'] = MinMaxNormalizer(
            action_stats, device=args.device)
        dataset.normalizer = normalizer
        return dataset

    datasets = {name: make_dataset(path) for name, path in task_paths.items()}
    num_workers = int(getattr(args, "num_workers", 8))
    dataloaders = {}
    for task_name, task_dataset in datasets.items():
        train_sampler = task_dataset.get_dataset_sampler()
        dataloaders[task_name] = torch.utils.data.DataLoader(
            task_dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=momart_dataset_utils.dynamic_batch_collate_fn,
        )
        print(f"\n============= Training Dataset: {task_name} =============")
        print(task_dataset)
    dataset = next(iter(datasets.values()))
    batch_mux = BalancedTaskBatchMux(dataloaders, seed=args.seed)

    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition.multi_image_condition import MultiImageObsConditionMomart
        from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP
        
        # nn_condition = MultiImageObsConditionMomart(
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
            ema_rate=float(getattr(args, "ema_rate", 0.999)),
            optim_params={
                "lr": args.lr,
                "betas": (
                    float(getattr(args, "adam_beta1", 0.9)),
                    float(getattr(args, "adam_beta2", 0.999)),
                ),
            })
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Memory Model =======================")
    report_parameters(nn_memory)
    print(f"===================================================================================")
    
    
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    n_gradient_step = auto_resume_training(
        agent, lr_scheduler, args.work_dir, args.gradient_steps)

    if args.mode == "train":
        # ----------------- Training ----------------------
        diffusion_loss_list = []
        start_time = time.time()
        while True:
            task_name, batch = batch_mux.next(n_gradient_step)
            task_dataset = datasets[task_name]
            # get condition
            nobs = batch['obs']
            # condition = {}
            # for k in nobs.keys():
            #     con = nobs[k][:, :args.obs_steps, :].numpy().astype(np.float32)#.to(args.device).to(torch.float32)  # (B, To, H, W, C)
            #     # if k.startswith('rgb'):
            #     #     con /= 255.
            #     #     con = np.transpose(con, (0, 1, 4, 2, 3))  # (B, T, C, H, W)
            #     # elif k.startswith('depth'):
            #     #     con = np.transpose(con, (0, 1, 4, 2, 3))  # (B, T, C, H, W)
            #     con = dataset.normalizer['obs'][k].normalize(con)
            #     con = torch.tensor(con, device=args.device, dtype=torch.float32)  # (B, T, C, H, W)
            #     condition[k] = con
            condition = nobs['dp_emb'][:, :args.obs_steps, :].to(args.device)
            # Convert actions to torch tensor and normalize (normalizer handles device)
            naction = batch['actions']  # Already torch tensor from dataloader
            if isinstance(naction, np.ndarray):
                naction = torch.from_numpy(naction).to(args.device)
            else:
                naction = naction.to(args.device)
            naction = task_dataset.normalizer['action'].normalize(naction)
            memory = nobs['dp_emb_mem'].to(args.device)
            # print(memory.shape[1])
            mem_steps = nobs['ep_step'].to(args.device)
            # update diffusion
            diffusion_loss = agent.update(naction, condition, memory, mem_steps)['loss']
            lr_scheduler.step()
            diffusion_loss_list.append(diffusion_loss)

            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    'step': n_gradient_step,
                    'total_time': time.time() - start_time,
                    'avg_diffusion_loss': np.mean(diffusion_loss_list),
                    'task_index': list(datasets).index(task_name),
                }
                logger.log(metrics, category='train')
                diffusion_loss_list = []
            
            if n_gradient_step % args.save_freq == 0 and n_gradient_step > 0:
                logger.save_agent(agent=agent, identifier=n_gradient_step)
                torch.save(lr_scheduler.state_dict(), args.work_dir + f'/models/scheduler_{str(n_gradient_step)}.pt')
                prune_checkpoints(
                    args.work_dir, keep=int(args.get("keep_checkpoints", 3)))
            
            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                # finish
                logger.finish()
                break
        
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()









    
