"""Train No-History with box-painted recent main-view RGB observations."""

import collections
import json
import os
import pathlib
import random
import re
import time

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import robomimic.utils.obs_utils as ObsUtils

from cleandiffuser.dataset.dataset_utils import ImageNormalizer, MinMaxNormalizer
from cleandiffuser.dataset.multitask import BalancedTaskBatchMux
from cleandiffuser.utils import report_parameters

from .dataset import BoxedNoHistoryDataset
from momart_no_history_baseline.model import build_agent


VISUAL_KEYS = {"rgb", "rgb_wrist", "depth", "depth_wrist"}


def initialize_modalities(args):
    mapping = collections.defaultdict(list)
    for key, attr in args.shape_meta["obs"].items():
        mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(mapping)


def build_suite(args):
    datasets, task_ids = {}, {}
    for spec in args.task_datasets:
        name = str(spec.name)
        datasets[name] = BoxedNoHistoryDataset(
            zarr_path=os.path.join(str(args.data_root), name+".zarr"),
            bbox_path=os.path.join(str(args.bbox_root), name+".bbox.zarr"),
            obs_keys=args.shape_meta["obs"], seq_length=int(args.horizon),
            obs_steps=int(args.obs_steps),
            subsample_ratio=int(args.dataset_memory_subsample_ratio))
        task_ids[name] = int(spec.task_id)

    proprio_min = np.full(args.action_dim, np.inf, dtype=np.float32)
    proprio_max = np.full(args.action_dim, -np.inf, dtype=np.float32)
    action_min = np.full(args.action_dim, np.inf, dtype=np.float32)
    action_max = np.full(args.action_dim, -np.inf, dtype=np.float32)
    for dataset in datasets.values():
        # TrainingArray is a logical view; slice once instead of letting NumPy
        # fetch every frame separately through its sequence fallback.
        proprio = np.asarray(dataset.replay_buffer["proprio"][:])
        action = np.asarray(dataset.replay_buffer["action"][:])
        proprio_min = np.minimum(proprio_min, proprio.min(axis=0))
        proprio_max = np.maximum(proprio_max, proprio.max(axis=0))
        action_min = np.minimum(action_min, action.min(axis=0))
        action_max = np.maximum(action_max, action.max(axis=0))
    metadata = [d.detector_metadata for d in datasets.values()]
    if any(m != metadata[0] for m in metadata):
        raise ValueError("Datasets have different detector settings")
    normalizer = collections.defaultdict(dict)
    for key in VISUAL_KEYS | {"scan"}:
        normalizer["obs"][key] = ImageNormalizer(device=args.device)
    normalizer["obs"]["proprio"] = MinMaxNormalizer(
        np.stack([proprio_min, proprio_max]), device=args.device)
    normalizer["action"] = MinMaxNormalizer(
        np.stack([action_min, action_max]), device=args.device)

    loaders = {}
    for name, dataset in datasets.items():
        dataset.normalizer = normalizer
        sampler = dataset.get_dataset_sampler()
        loaders[name] = torch.utils.data.DataLoader(
            dataset, sampler=sampler, shuffle=sampler is None,
            batch_size=int(args.physical_batch_size), drop_last=True,
            num_workers=int(args.num_workers), pin_memory=True,
            persistent_workers=int(args.num_workers) > 0,
        )
        print(f"dataset={name} task_id={task_ids[name]} {dataset}", flush=True)
    stats = dict(
        proprio_min=proprio_min, proprio_max=proprio_max,
        action_min=action_min, action_max=action_max,
    )
    return datasets, task_ids, normalizer, loaders, stats


def split_batch(args, dataset, task_id, batch):
    condition = {
        key: dataset.normalizer["obs"][key].normalize(
            value.to(args.device, dtype=torch.float32, non_blocking=True))
        for key, value in batch["obs"].items()
    }
    condition["_task_id"] = torch.full(
        (batch["actions"].shape[0],), int(task_id),
        device=args.device, dtype=torch.long)
    action = dataset.normalizer["action"].normalize(
        batch["actions"].to(args.device, non_blocking=True))
    return condition, action


def latest_checkpoint_pair(work_dir):
    models = pathlib.Path(work_dir) / "models"
    pairs = []
    for model in models.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", model.name)
        if match and model.stat().st_size:
            step = int(match.group(1))
            scheduler = models / f"scheduler_{step}.pt"
            if scheduler.is_file() and scheduler.stat().st_size:
                pairs.append((step, model, scheduler))
    return max(pairs, default=None, key=lambda pair: pair[0])


def train(args):
    if (int(args.effective_batch_size) != int(args.physical_batch_size)
            or int(args.physical_batch_size) not in (64, 256)):
        raise ValueError("This baseline requires a physical and effective batch of 64 or 256")
    if int(args.keep_checkpoints) < int(args.gradient_steps) // int(args.save_freq):
        raise ValueError("keep_checkpoints must retain all requested milestones")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    initialize_modalities(args)
    datasets, task_ids, normalizer, loaders, stats = build_suite(args)
    mux = BalancedTaskBatchMux(loaders, seed=args.seed)
    agent = build_agent(args, normalizer)
    report_parameters(agent.model["diffusion"])
    report_parameters(agent.model["condition"])
    scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)

    work_dir = pathlib.Path(args.work_dir)
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    np.savez(work_dir / "normalizers.npz", **stats)
    OmegaConf.save(args, work_dir / "train_config.yaml")
    detector_metadata = next(iter(datasets.values())).detector_metadata
    (work_dir / "bbox_input.json").write_text(json.dumps(dict(
        detector=detector_metadata, line_color_rgb=[255,0,0], line_width=1,
        input_key="rgb", missing="original_image"), indent=2)+"\n")
    checkpoint = latest_checkpoint_pair(work_dir)
    step = 0
    if checkpoint is not None:
        saved_step, model_path, scheduler_path = checkpoint
        agent.load(str(model_path))
        scheduler.load_state_dict(torch.load(
            scheduler_path, map_location=agent.device, weights_only=False))
        step = saved_step
        if scheduler.last_epoch != saved_step:
            raise ValueError('Checkpoint and scheduler steps disagree')
        if args.get('resume_cosine_horizon') is not None:
            from .resume import extend_cosine_horizon
            old_horizon = scheduler.T_max
            extend_cosine_horizon(scheduler, args.resume_cosine_horizon)
            print(f'cosine horizon {old_horizon} -> {scheduler.T_max}; '
                  f'resumed lr={scheduler.get_last_lr()}', flush=True)
        print(f"resumed from step {saved_step}", flush=True)
    elif args.get('resume_cosine_horizon') is not None:
        raise ValueError('This continuation config requires an existing checkpoint pair')

    initial_step = step
    losses = []
    started = time.time()
    print(f"START method=boxed_no_history physical_batch={args.physical_batch_size} "
          f"effective_batch={args.effective_batch_size}", flush=True)
    while step < args.gradient_steps:
        name, batch = mux.next(step)
        condition, action = split_batch(args, datasets[name], task_ids[name], batch)
        result = agent.update(action, condition, None, None)
        scheduler.step()
        step += 1
        losses.append(float(result["loss"]))
        if step == initial_step + 1 or step % args.log_freq == 0:
            metrics = {
                "step": step, "total_time": time.time() - started,
                "avg_diffusion_loss": float(np.mean(losses)),
                "task_index": list(datasets).index(name),
                "accumulation_steps": 1,
                "physical_batch_size": int(args.physical_batch_size),
                "effective_batch_size": int(args.effective_batch_size),
                "learning_rate": float(agent.optimizer.param_groups[0]['lr']),
            }
            with (work_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(metrics, flush=True)
            losses.clear()
        if step > 0 and step % args.save_freq == 0:
            model_path = models_dir / f"model_{step}.pt"
            agent.save(str(model_path))
            torch.save(scheduler.state_dict(), models_dir / f"scheduler_{step}.pt")
            print(f"saved checkpoint step={step}", flush=True)



@hydra.main(config_path=".", config_name="config_bs256_150k", version_base=None)
def main(args):
    assert args.horizon == args.obs_steps + args.action_steps - 1
    train(args)


if __name__ == "__main__":
    main()
