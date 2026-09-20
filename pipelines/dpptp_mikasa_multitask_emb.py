"""Balanced four-task MIKASA DPPTP on frozen observation embeddings."""

import os
import pathlib
import time

import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import Logger, auto_resume_training, prune_checkpoints, set_seed
from cleandiffuser.dataset.mikasa_dataset import MikasaDataset
from cleandiffuser.dataset.multitask import BalancedTaskBatchMux
from cleandiffuser.utils import report_parameters


def train(args):
    datasets, loaders = {}, {}
    workers = int(args.num_workers)
    for spec in args.task_datasets:
        name, path = str(spec.name), os.path.expanduser(str(spec.path))
        dataset = MikasaDataset(
            path, horizon=args.horizon, shape_meta=args.shape_meta,
            n_obs_steps=args.obs_steps, pad_before=args.obs_steps - 1,
            pad_after=args.action_steps - 1,
            use_full_episodes=bool(args.dataset_use_full_episodes))
        datasets[name] = dataset
        loaders[name] = torch.utils.data.DataLoader(
            dataset, batch_size=int(args.batch_size), shuffle=True,
            drop_last=True, num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0)
        print(f"dataset={name} {dataset}", flush=True)

    from cleandiffuser.diffusion.ddpm import DDPM
    from cleandiffuser.nn_diffusion.chitransformerptp import ChiTransformerPTP
    diffusion = ChiTransformerPTP(
        args.action_dim, 256, args.horizon, args.obs_steps, d_model=256,
        nhead=4, num_layers=args.diffusion_num_layers,
        timestep_emb_type="positional").to(args.device)
    report_parameters(diffusion)
    limit = torch.ones((1, args.horizon, args.action_dim), device=args.device)
    agent = DDPM(
        nn_diffusion=diffusion, nn_condition=None, device=args.device,
        diffusion_steps=args.sample_steps, x_max=limit, x_min=-limit,
        ema_rate=args.ema_rate,
        optim_params={"lr": args.lr,
                      "betas": (args.adam_beta1, args.adam_beta2)})
    scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    step = auto_resume_training(
        agent, scheduler, args.work_dir, args.gradient_steps)
    logger = Logger(pathlib.Path(args.work_dir), args)
    mux = BalancedTaskBatchMux(loaders, seed=args.seed)
    losses, started = [], time.time()
    print(f"START embedding_dpptp batch={args.batch_size}", flush=True)
    while step <= args.gradient_steps:
        task_name, batch = mux.next(step)
        condition = batch["obs"]["dp_emb"][:, :args.obs_steps].to(
            args.device, non_blocking=True)
        action = batch["action"].to(args.device, non_blocking=True)
        log = agent.update(action, condition)
        scheduler.step()
        losses.append(float(log["loss"]))
        if step % args.log_freq == 0:
            logger.log({
                "step": step, "total_time": time.time() - started,
                "avg_diffusion_loss": float(np.mean(losses)),
                "task_index": list(datasets).index(task_name),
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
    config_path="../configs/dp/mikasa/chi_transformer_ptp",
    config_name="mikasa_multitask_emb", version_base=None)
def pipeline(args):
    set_seed(args.seed)
    assert args.horizon == args.obs_steps + args.action_steps - 1
    train(args)


if __name__ == "__main__":
    pipeline()
