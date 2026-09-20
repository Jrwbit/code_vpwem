"""Train the four MOMART history methods under one balanced suite protocol."""

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
from cleandiffuser.dataset.dataset_utils import ImageNormalizer, MinMaxNormalizer
from cleandiffuser.dataset.multitask import BalancedTaskBatchMux
from cleandiffuser.utils import report_parameters
import cleandiffuser.utils.robomimic_dataset_utils as momart_dataset_utils
import cleandiffuser.utils.robomimic_train_utils as TrainUtils
import robomimic.utils.obs_utils as ObsUtils

warnings.filterwarnings("ignore")


METHODS = {"vpwem", "direct", "direct_token2", "no_history"}
VISUAL_KEYS = {"rgb", "rgb_wrist", "depth", "depth_wrist"}


class DictToObj:
    def __init__(self, values):
        for key, value in values.items():
            setattr(self, key, DictToObj(value) if isinstance(value, dict) else value)


def initialize_modalities(args):
    mapping = collections.defaultdict(list)
    for key, attr in args.shape_meta["obs"].items():
        mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(mapping)


def make_dataset_config(args, path):
    return DictToObj({
        "train": {
            "data": path,
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": True,
            "hdf5_load_next_obs": False,
            "hdf5_normalize_obs": False,
            "hdf5_filter_key": "train",
            "seq_length": int(args.horizon),
            "dataset_keys": ["actions", "rewards", "dones"],
            "goal_mode": None,
        },
        "experiment": {"validate": False},
        "memory": {
            "dataset_memory_type": str(args.dataset_memory_type),
            "dataset_memory_subsample_ratio": int(
                args.dataset_memory_subsample_ratio),
            "dataset_maximum_memory_length": int(
                args.dataset_maximum_memory_length),
            "dataset_preserve_episode_anchor": True,
            "history_sampling_seed": int(args.history_sampling_seed),
        },
    })


def build_suite(args):
    datasets = {}
    task_ids = {}
    for spec in args.task_datasets:
        name = str(spec.name)
        dataset, _ = TrainUtils.load_data_for_training(
            make_dataset_config(args, os.path.expanduser(str(spec.path))),
            obs_keys=args.shape_meta["obs"])
        datasets[name] = dataset
        task_ids[name] = int(spec.task_id)

    # A mixed policy must use one coordinate system. In particular, expert and
    # suboptimal files for the same semantic task cannot silently use different
    # action normalizers while sharing a task id.
    proprio_min = np.full(args.action_dim, np.inf, dtype=np.float32)
    proprio_max = np.full(args.action_dim, -np.inf, dtype=np.float32)
    action_min = np.full(args.action_dim, np.inf, dtype=np.float32)
    action_max = np.full(args.action_dim, -np.inf, dtype=np.float32)
    for dataset in datasets.values():
        for demo in dataset.hdf5_cache.values():
            proprio = demo["obs"]["proprio"]
            action = demo["actions"]
            proprio_min = np.minimum(proprio_min, proprio.min(axis=0))
            proprio_max = np.maximum(proprio_max, proprio.max(axis=0))
            action_min = np.minimum(action_min, action.min(axis=0))
            action_max = np.maximum(action_max, action.max(axis=0))

    normalizer = collections.defaultdict(dict)
    for key in VISUAL_KEYS | {"scan"}:
        normalizer["obs"][key] = ImageNormalizer(device=args.device)
    normalizer["obs"]["proprio"] = MinMaxNormalizer(
        np.stack([proprio_min, proprio_max]), device=args.device)
    normalizer["action"] = MinMaxNormalizer(
        np.stack([action_min, action_max]), device=args.device)
    for dataset in datasets.values():
        dataset.normalizer = normalizer

    loaders = {}
    for name, dataset in datasets.items():
        sampler = dataset.get_dataset_sampler()
        loaders[name] = torch.utils.data.DataLoader(
            dataset, sampler=sampler, shuffle=sampler is None,
            batch_size=int(args.physical_batch_size), drop_last=True,
            num_workers=int(args.num_workers), pin_memory=True,
            persistent_workers=int(args.num_workers) > 0,
            collate_fn=momart_dataset_utils.dynamic_batch_collate_fn)
        print(f"dataset={name} task_id={task_ids[name]} {dataset}", flush=True)
    return datasets, task_ids, normalizer, loaders


def build_agent(args, normalizer):
    from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM
    from cleandiffuser.nn_condition import (
        DirectSpatialHistoryConditionMomart,
        MultiImageObsConditionMomart,
        TaskConditionedDirectObsEncoder,
        TaskConditionedObsEncoder,
    )
    from cleandiffuser.nn_diffusion.longmem_chitransformerptp import (
        LongMemChiTransformerPTP)

    method = str(args.history_method)
    if method not in METHODS:
        raise ValueError(f"history_method must be one of {sorted(METHODS)}")
    condition_kwargs = dict(
        shape_meta=args.shape_meta, emb_dim=256,
        rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
        crop_shape=args.crop_shape, random_crop=args.random_crop,
        use_group_norm=args.use_group_norm, use_seq=True,
        keep_horizon_dims=True)
    if method.startswith("direct"):
        raw_condition = DirectSpatialHistoryConditionMomart(
            **condition_kwargs, pool_size=args.history_pool_size,
            history_frames=args.dataset_maximum_memory_length,
            history_output_tokens=2 if method == "direct_token2" else 0,
            history_subsample_ratio=args.dataset_memory_subsample_ratio)
        nn_condition = TaskConditionedDirectObsEncoder(
            raw_condition, args.num_tasks, emb_dim=256).to(args.device)
        memory_tokens = nn_condition.history_token_count
    else:
        raw_condition = MultiImageObsConditionMomart(**condition_kwargs)
        nn_condition = TaskConditionedObsEncoder(
            raw_condition, args.num_tasks, emb_dim=256).to(args.device)
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
            subsample_ratio=args.dataset_memory_subsample_ratio).to(args.device)

    x_max = torch.ones(
        (1, args.horizon, args.action_dim), device=args.device)
    return LongMemDDPM(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition,
        nn_memory=nn_memory, device=args.device,
        diffusion_steps=args.sample_steps, x_max=x_max, x_min=-x_max,
        args_dict=args,
        action_range=normalizer["action"].range.to(args.device),
        action_min=normalizer["action"].min.to(args.device),
        short_cond_dropout=args.short_cond_dropout,
        long_cond_dropout=args.long_cond_dropout,
        ema_rate=args.ema_rate,
        optim_params={"lr": args.lr,
                      "betas": (args.adam_beta1, args.adam_beta2)})


def split_batch(args, dataset, task_id, batch):
    condition, memory = {}, {}
    for key, value in batch["obs"].items():
        if key.endswith("_mem"):
            base_key = key[:-4]
            memory[base_key] = dataset.normalizer["obs"][base_key].normalize(
                value.to(args.device, non_blocking=True))
        elif key not in {"ep_step", "actual_num_frames"}:
            condition[key] = dataset.normalizer["obs"][key].normalize(
                value[:, :args.obs_steps].to(args.device, non_blocking=True))
    batch_size = batch["actions"].shape[0]
    task = torch.full(
        (batch_size,), int(task_id), device=args.device, dtype=torch.long)
    condition["_task_id"] = task
    memory["_task_id"] = task
    action = dataset.normalizer["action"].normalize(
        batch["actions"].to(args.device, non_blocking=True))
    ep_step = batch["obs"]["ep_step"].to(
        args.device, non_blocking=True).long()
    if str(args.history_method) == "no_history":
        memory, ep_step = None, None
    return condition, memory, action, ep_step


def train(args):
    datasets, task_ids, normalizer, loaders = build_suite(args)
    mux = BalancedTaskBatchMux(loaders, seed=args.seed)
    agent = build_agent(args, normalizer)
    report_parameters(agent.model["diffusion"])
    report_parameters(agent.model["condition"])
    if agent.model["memory"] is not None:
        report_parameters(agent.model["memory"])

    effective = int(args.effective_batch_size)
    physical = int(args.physical_batch_size)
    if effective < physical or effective % physical:
        raise ValueError("effective_batch_size must be divisible by physical_batch_size")
    accumulation = effective // physical
    scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    step = auto_resume_training(
        agent, scheduler, args.work_dir, args.gradient_steps)
    logger = Logger(pathlib.Path(args.work_dir), args)
    losses, started = [], time.time()
    print(
        f"START method={args.history_method} physical_batch={physical} "
        f"effective_batch={effective} accumulation={accumulation}",
        flush=True)

    while step <= args.gradient_steps:
        task_name = mux.task_for_step(step)
        micro_losses = []
        for micro_step in range(accumulation):
            selected_name, batch = mux.next(step)
            assert selected_name == task_name
            condition, memory, action, ep_step = split_batch(
                args, datasets[task_name], task_ids[task_name], batch)
            log = agent.update(
                action, condition, memory, ep_step,
                optimizer_step=micro_step == accumulation - 1,
                loss_scale=1.0 / accumulation)
            micro_losses.append(float(log["loss"]))
        scheduler.step()
        losses.append(float(np.mean(micro_losses)))
        if step % args.log_freq == 0:
            logger.log({
                "step": step, "total_time": time.time() - started,
                "avg_diffusion_loss": float(np.mean(losses)),
                "task_index": list(datasets).index(task_name),
                "accumulation_steps": accumulation,
            }, category="train")
            losses.clear()
        if step > 0 and step % args.save_freq == 0:
            logger.save_agent(agent=agent, identifier=step)
            torch.save(scheduler.state_dict(), os.path.join(
                args.work_dir, "models", f"scheduler_{step}.pt"))
            prune_checkpoints(args.work_dir, keep=args.keep_checkpoints)
        step += 1
    logger.finish()


@hydra.main(
    config_path="../configs/dp/momart/fair_history_multitask",
    config_name="base", version_base=None)
def pipeline(args):
    set_seed(args.seed)
    assert args.horizon == args.obs_steps + args.action_steps - 1
    initialize_modalities(args)
    if args.mode != "train":
        raise ValueError("this suite launcher currently supports mode=train")
    train(args)


if __name__ == "__main__":
    pipeline()
