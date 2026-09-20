"""Balanced four-task MIKASA training for the fair-history models."""

import os
import pathlib
import time

import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import Logger, auto_resume_training, prune_checkpoints, set_seed
from cleandiffuser.dataset.dataset_utils import dynamic_batch_collate_fn
from cleandiffuser.dataset.mikasa_dataset_longmem import LongMemMikasaDataset
from cleandiffuser.dataset.multitask import BalancedTaskBatchMux
from cleandiffuser.utils import report_parameters
from fair_history_mikasa import build_agent, split_batch


def build_datasets_and_loaders(args):
    datasets, loaders = {}, {}
    workers = int(args.num_workers)
    for task_index, spec in enumerate(args.task_datasets):
        name = str(spec.name)
        path = os.path.expanduser(str(spec.path))
        dataset = LongMemMikasaDataset(
            dataset_dir=path,
            shape_meta=args.shape_meta,
            n_obs_steps=args.obs_steps,
            horizon=args.horizon,
            pad_before=args.obs_steps - 1,
            pad_after=args.action_steps - 1,
            dataset_memory_type="random_from_full",
            subsample_ratio=args.dataset_memory_subsample_ratio,
            maximum_memory_length=args.dataset_maximum_memory_length,
            preserve_episode_anchor=True,
            history_sampling_seed=int(args.history_sampling_seed) + task_index,
            conversion_n_workers=args.conversion_n_workers,
            conversion_max_inflight_tasks=args.conversion_max_inflight_tasks,
            load_images=True,
            device=args.device,
        )
        sampler = dataset.get_dataset_sampler()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            collate_fn=dynamic_batch_collate_fn,
        )
        datasets[name] = dataset
        loaders[name] = loader
        print(f"dataset={name} {dataset}", flush=True)
    return datasets, loaders


def train(args):
    datasets, loaders = build_datasets_and_loaders(args)
    task_names = list(datasets)
    mux = BalancedTaskBatchMux(loaders, seed=args.seed)

    # All four tasks share the same action space. Batches stay task-homogeneous
    # so each task's own observation/action normalizer is used correctly.
    agent = build_agent(args, datasets[task_names[0]].normalizer)
    report_parameters(agent.model["diffusion"])
    report_parameters(agent.model["condition"])
    scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    step = auto_resume_training(
        agent, scheduler, args.work_dir, args.gradient_steps)
    logger = Logger(pathlib.Path(args.work_dir), args)
    losses, started = [], time.time()
    print(
        f"START fair_history_mikasa_multitask method={args.history_method} "
        f"physical_batch={args.batch_size} gradient_accumulation=1 "
        f"tasks={task_names}",
        flush=True,
    )

    while step <= args.gradient_steps:
        task_name, batch = mux.next(step)
        dataset = datasets[task_name]
        condition, memory, action, ep_step = split_batch(
            args, dataset, batch)
        log = agent.update(action, condition, memory, ep_step)
        scheduler.step()
        losses.append(float(log["loss"]))
        if step % args.log_freq == 0:
            logger.log({
                "step": step,
                "total_time": time.time() - started,
                "avg_loss": float(np.mean(losses)),
                "task_index": task_names.index(task_name),
                "physical_batch_size": int(args.batch_size),
                "gradient_accumulation_steps": 1,
            }, category="train")
            losses.clear()
        if step > 0 and step % args.save_freq == 0:
            logger.save_agent(agent=agent, identifier=step)
            torch.save(
                scheduler.state_dict(),
                os.path.join(args.work_dir, "models", f"scheduler_{step}.pt"),
            )
            prune_checkpoints(args.work_dir, keep=args.keep_checkpoints)
        step += 1
    logger.finish()


@hydra.main(
    config_path="../configs/dp/mikasa/fair_history",
    config_name="multitask",
    version_base=None,
)
def pipeline(args):
    set_seed(args.seed)
    assert args.horizon == args.obs_steps + args.action_steps - 1
    if str(args.history_method) != "direct_token2":
        raise ValueError("this queue is intended for direct_token2")
    train(args)


if __name__ == "__main__":
    pipeline()
