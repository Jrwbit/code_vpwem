"""MIKASA 上的 VPWEM 训练入口（依赖装配层）。

建议把本文件当作阅读整个项目的起点，而不是算法实现本身。它负责把五个部件
接起来，核心调用链如下：

    Hydra 配置
      -> LongMemMikasaDataset / DataLoader
      -> MultiImageObsConditionRobomimic（图像和关节状态 -> 256 维 token）
      -> QFormerMemory 等记忆模块（可变长历史 -> 固定 Tm 个 token）
      -> LongMemChiTransformerPTP（带噪动作 + 当前观测 + 记忆 -> 预测噪声）
      -> LongMemDDPM.update（loss、反向传播和 EMA）

其中 ``To=obs_steps`` 是不压缩的近期工作记忆长度，``Tm=mem_compress_length``
是情景记忆 token 数，``Ta=horizon`` 是训练时预测的动作序列长度。
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
import torch.nn as nn
from utils import (
    set_seed, parse_cfg, Logger, auto_resume_training, prune_checkpoints)
from torch.optim.lr_scheduler import CosineAnnealingLR

# from cleandiffuser.dataset.robomimic_dataset import RobomimicImageDataset
from cleandiffuser.dataset.mikasa_dataset_longmem import LongMemMikasaDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader, dynamic_batch_collate_fn
from cleandiffuser.dataset.ncm_pairing import partner_positions
from cleandiffuser.utils import report_parameters
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

@hydra.main(config_path="../configs/dp/mikasa/chi_transformer_ptp_longmem", config_name="shell_game_touch")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    assert args.horizon == args.obs_steps + args.action_steps - 1

    # modality_mapping = collections.defaultdict(list)
    # for key, attr in args.shape_meta['obs'].items():
    #     modality_mapping[attr.get('type', 'low_dim')].append(key)
    # ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        
    # ---------------- Create Dataset ----------------
    # Dataset 同时返回两条时间线：
    #   obs[k]      -> 当前 To 帧，作为 working memory / 短条件；
    #   obs[k_mem]  -> 当前窗口之前的稀疏历史，稍后由 nn_memory 压缩。
    dataset_path = os.path.expanduser(args.dataset_path)
    
    dataset = LongMemMikasaDataset(
        dataset_dir=dataset_path,
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        horizon=args.horizon,
        pad_before=args.obs_steps-1,
        pad_after=args.action_steps-1,
        dataset_memory_type=args.dataset_memory_type,
        subsample_ratio=args.dataset_memory_subsample_ratio,
        maximum_memory_length=int(getattr(
            args, "dataset_maximum_memory_length", 16)),
        preserve_episode_anchor=bool(getattr(
            args, "dataset_preserve_episode_anchor", False)),
        include_action_memory=(
            str(args.mem_compress_method) == "aem_style"),
        ncm_pair_manifest_path=getattr(args, "ncm_pair_manifest_path", None),
        ncm_pairing_seed=int(getattr(args, "ncm_pairing_seed", args.seed)),
        ncm_pair_batch_fraction=float(getattr(
            args, "ncm_pair_batch_fraction", 1.0)),
        shellgame_hold_step=getattr(args, "shellgame_hold_step", None),
        shellgame_hold_steps=int(getattr(args, "shellgame_hold_steps", 0)),
        conversion_n_workers=int(getattr(args, "conversion_n_workers", 4)),
        conversion_max_inflight_tasks=int(getattr(
            args, "conversion_max_inflight_tasks", 20)),
        device=args.device,
    )
    normalizer_dataset_path = getattr(
        args, "normalizer_dataset_path", None)
    if normalizer_dataset_path:
        normalizer_dataset = LongMemMikasaDataset(
            dataset_dir=os.path.expanduser(str(normalizer_dataset_path)),
            shape_meta=args.shape_meta,
            n_obs_steps=args.obs_steps,
            horizon=args.horizon,
            pad_before=args.obs_steps - 1,
            pad_after=args.action_steps - 1,
            dataset_memory_type=args.dataset_memory_type,
            subsample_ratio=args.dataset_memory_subsample_ratio,
            maximum_memory_length=int(getattr(
                args, "dataset_maximum_memory_length", 16)),
            conversion_n_workers=int(getattr(
                args, "conversion_n_workers", 4)),
            conversion_max_inflight_tasks=int(getattr(
                args, "conversion_max_inflight_tasks", 20)),
            load_images=False,
            device=args.device,
        )
        dataset.normalizer = normalizer_dataset.normalizer
        print(
            f"Using frozen-checkpoint normalizer from "
            f"{normalizer_dataset_path}")
    print(dataset)
    
    # Get dataset sampler for dynamic batching
    ncm_batch_sampler = dataset.get_ncm_batch_sampler()
    num_workers = int(getattr(args, "num_workers", 8))
    if ncm_batch_sampler is not None:
        manifest_batch_size = int(dataset.ncm_manifest['batches'].shape[1])
        if manifest_batch_size != int(args.batch_size):
            raise ValueError(
                f"NCM manifest batch size {manifest_batch_size} does not "
                f"match configured batch_size {args.batch_size}")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=ncm_batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=dynamic_batch_collate_fn,
        )
    else:
        train_sampler = dataset.get_dataset_sampler()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=dynamic_batch_collate_fn,
        )
        
    # ---------------- Assemble VPWEM ----------------
    # nn_condition、nn_memory、nn_diffusion 分别承担“感知、记忆、动作生成”。
    # 三者由 LongMemDDPM 统一持有和优化，因此记忆压缩器会收到动作去噪损失的梯度。
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
    # 这里是所有记忆压缩方法的统一替换点；保持输出契约
    # [B, T_history, 256] -> [B, Tm, 256]，策略骨干无需随方法改变。
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
    action_range = dataset.normalizer['action'].range.to(args.device)
    action_min = dataset.normalizer['action'].min.to(args.device)
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
            optim_params={"lr": args.lr})
    else:
        raise NotImplementedError(f"Diffusion method {args.diffusion} not supported for LongMem")
    
    init_model_path = getattr(args, "init_model_path", None)
    if init_model_path:
        init_checkpoint = torch.load(
            os.path.expanduser(str(init_model_path)),
            map_location=args.device, weights_only=False)
        init_strict = bool(getattr(args, "init_model_strict", True))
        model_msg = agent.model.load_state_dict(
            init_checkpoint["model"], strict=init_strict)
        ema_msg = agent.model_ema.load_state_dict(
            init_checkpoint["model_ema"], strict=init_strict)
        print(f"Initialization model loaded with message: {model_msg}")
        print(f"Initialization EMA loaded with message: {ema_msg}")

    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    n_gradient_step = auto_resume_training(
        agent, lr_scheduler, args.work_dir, args.gradient_steps)
    
    if args.mode == "train":
        # ----------------- Training ----------------------
        train_metric_lists = collections.defaultdict(list)
        start_time = time.time()
        early_save_steps = {
            int(step) for step in getattr(
                args, "checkpoint_save_early_steps", [])
        }
        for batch in loop_dataloader(dataloader):
            # 把 batch 拆成近期观测 condition 与早期历史 memory。
            # 原始图像版本中二者都是多模态 dict；LongMemDDPM.loss 会让它们
            # 共享同一个视觉编码器，再仅对 memory 分支做定长压缩。
            nobs = batch['obs']
            condition = {}
            memory = {}
            for k in nobs.keys():
                if k.endswith('_mem'):
                    # For memory keys, use the base key (without _mem) to normalize
                    base_key = k[:-4]  # Remove '_mem' suffix
                    con = nobs[k].to(args.device)  # Memory data (full sequence)
                    # Normalize using the base key's normalizer
                    con = dataset.normalizer['obs'][base_key].normalize(con)
                    memory[base_key] = con  # Store in memory dict with base key name
                elif k != 'ep_step' and k != 'actual_num_frames':
                    # For condition keys (non-memory, non-ep_step), normalize normally
                    con = nobs[k][:, :args.obs_steps, :].to(args.device)  # (B, To, ...)
                    con = dataset.normalizer['obs'][k].normalize(con)
                    condition[k] = con
            # Convert actions to torch tensor and normalize (normalizer handles device)
            naction = batch['action']  # Already torch tensor from dataloader
            if isinstance(naction, np.ndarray):
                naction = torch.from_numpy(naction).to(args.device)
            else:
                naction = naction.to(args.device)
            naction = dataset.normalizer['action'].normalize(naction)  # Returns torch tensor on device
            
            mem_steps = nobs['ep_step'].int().to(args.device)

            counterfactual_index = None
            counterfactual_weight = None
            memory_context = None
            if 'memory_action' in batch:
                memory_action = batch['memory_action'].to(args.device)
                memory_action = dataset.normalizer['action'].normalize(
                    memory_action)
                memory_context = {'action_history': memory_action}
            negative_mode = str(getattr(
                args, "counterfactual_negative_mode", "none"))
            if 'ncm' in batch:
                ncm = batch['ncm']
                counterfactual_weight = ncm['alias_weight'].to(args.device)
                if negative_mode in {'matched', 'unmatched_branch'}:
                    partner_key = (
                        'matched_partner' if negative_mode == 'matched'
                        else 'unmatched_partner')
                    counterfactual_index = partner_positions(
                        ncm['sample_index'].to(args.device),
                        ncm[partner_key].to(args.device),
                        valid=counterfactual_weight > 0)

            # 训练主入口：action 加噪、历史压缩、Transformer 去噪、反向传播和
            # EMA 更新都封装在 agent.update 中。阅读算法时可从该调用跳到
            # cleandiffuser/diffusion/longmem_ddpm.py::LongMemDDPM.loss。
            update_log = agent.update(
                naction, condition, memory, mem_steps,
                counterfactual_index=counterfactual_index,
                counterfactual_weight=counterfactual_weight,
                memory_context=memory_context)
            lr_scheduler.step()
            for key, value in update_log.items():
                if value is not None:
                    train_metric_lists[key].append(float(value))

            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    'step': n_gradient_step,
                    'total_time': time.time() - start_time,
                }
                metrics.update({
                    f'avg_{key}': float(np.mean(values))
                    for key, values in train_metric_lists.items()
                    if values
                })
                logger.log(metrics, category='train')
                train_metric_lists.clear()
            
            should_save = n_gradient_step > 0 and (
                n_gradient_step in early_save_steps
                or n_gradient_step % args.save_freq == 0)
            if should_save:
                logger.save_agent(agent=agent, identifier=n_gradient_step)
                torch.save(lr_scheduler.state_dict(), args.work_dir + f'/models/scheduler_{str(n_gradient_step)}.pt')
                prune_checkpoints(
                    args.work_dir, keep=int(args.get("keep_checkpoints", 3)))
            
            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                # finish
                # logger.finish(agent)
                break
        
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()
