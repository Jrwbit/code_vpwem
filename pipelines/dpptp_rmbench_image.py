import collections
import os
import pathlib
import signal
import time
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import Logger, set_seed
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.dataset.rmbench_dataset import RMBenchImageDataset
from cleandiffuser.utils import report_parameters


_STOP_REQUESTED = False


def _request_stop(signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Received signal {signum}; saving an interrupt checkpoint after the current step.", flush=True)


def _move_normalizer_to_device(normalizer, device):
    if isinstance(normalizer, dict):
        for value in normalizer.values():
            _move_normalizer_to_device(value, device)
        return normalizer
    for attr in ("min", "max", "range"):
        value = getattr(normalizer, attr, None)
        if isinstance(value, torch.Tensor):
            setattr(normalizer, attr, value.to(device))
    if hasattr(normalizer, "device"):
        normalizer.device = device
    return normalizer


def _move_normalizer_to_cpu(normalizer):
    if isinstance(normalizer, dict):
        for value in normalizer.values():
            _move_normalizer_to_cpu(value)
        return normalizer
    for attr in ("min", "max", "range"):
        value = getattr(normalizer, attr, None)
        if isinstance(value, torch.Tensor):
            setattr(normalizer, attr, value.detach().cpu())
    if hasattr(normalizer, "device"):
        normalizer.device = "cpu"
    return normalizer


def _none_if_unset(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return value


def _as_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return float(value)


def _set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        param_group["initial_lr"] = lr


def _checkpoint_step(path):
    stem = pathlib.Path(path).stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _latest_checkpoint(models_dir):
    models_dir = pathlib.Path(models_dir)
    checkpoints = []
    for model_path in models_dir.glob("model_*.pt"):
        step = _checkpoint_step(model_path)
        if step is not None:
            checkpoints.append((step, model_path))
    if not checkpoints:
        return None, None, None
    step, model_path = max(checkpoints, key=lambda item: item[0])
    scheduler_path = models_dir / f"scheduler_{step}.pt"
    return step, model_path, scheduler_path if scheduler_path.exists() else None


def _resolve_resume_checkpoint(args):
    resume_model_path = _none_if_unset(getattr(args, "resume_model_path", None))
    resume_scheduler_path = _none_if_unset(getattr(args, "resume_scheduler_path", None))
    resume_step = _none_if_unset(getattr(args, "resume_step", None))
    models_dir = pathlib.Path(args.work_dir) / "models"

    if resume_model_path is not None:
        model_path = pathlib.Path(os.path.expanduser(str(resume_model_path)))
        step = int(resume_step) if resume_step is not None else _checkpoint_step(model_path)
        scheduler_path = (
            pathlib.Path(os.path.expanduser(str(resume_scheduler_path)))
            if resume_scheduler_path is not None
            else models_dir / f"scheduler_{step}.pt"
            if step is not None
            else None
        )
        return step, model_path, scheduler_path if scheduler_path and scheduler_path.exists() else None

    if resume_step is not None:
        step = int(resume_step)
        model_path = models_dir / f"model_{step}.pt"
        scheduler_path = models_dir / f"scheduler_{step}.pt"
        return step, model_path, scheduler_path if scheduler_path.exists() else None

    if bool(getattr(args, "resume", False)):
        return _latest_checkpoint(models_dir)

    return None, None, None


def _save_checkpoint(logger, agent, lr_scheduler, normalizer, step, identifier=None):
    identifier = step if identifier is None else identifier
    logger.save_agent(agent=agent, identifier=identifier)
    torch.save(lr_scheduler.state_dict(), pathlib.Path(logger._model_dir) / f"scheduler_{identifier}.pt")
    torch.save(
        _move_normalizer_to_cpu(deepcopy(normalizer)),
        pathlib.Path(logger._model_dir) / "rmbench_swap_blocks_normalizer.pt",
    )


def build_agent(args):
    if args.nn != "chi_transformer":
        raise ValueError(f"Invalid nn type {args.nn}")

    from cleandiffuser.diffusion.ddpm import DDPM
    from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
    from cleandiffuser.nn_diffusion.chitransformerptp import ChiTransformerPTP

    nn_condition = MultiImageObsConditionRobomimic(
        shape_meta=args.shape_meta,
        emb_dim=256,
        rgb_model_name=args.rgb_model,
        resize_shape=args.resize_shape,
        crop_shape=args.crop_shape,
        random_crop=args.random_crop,
        use_group_norm=args.use_group_norm,
        use_seq=args.use_seq,
        keep_horizon_dims=True,
    ).to(args.device)
    nn_diffusion = ChiTransformerPTP(
        args.action_dim,
        256,
        args.horizon,
        args.obs_steps,
        d_model=256,
        nhead=4,
        num_layers=4,
        timestep_emb_type="positional",
    ).to(args.device)

    report_parameters(nn_diffusion)
    report_parameters(nn_condition)

    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device)
    x_min = -x_max
    if args.diffusion != "ddpm":
        raise NotImplementedError(f"Diffusion method {args.diffusion} not supported for RMBench DP")
    return DDPM(
        nn_diffusion=nn_diffusion,
        nn_condition=nn_condition,
        device=args.device,
        diffusion_steps=args.sample_steps,
        x_max=x_max,
        x_min=x_min,
        optim_params={"lr": args.lr},
    )


@hydra.main(config_path="../configs/dp/rmbench/chi_transformer_ptp", config_name="swap_blocks")
def pipeline(args):
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)
    assert args.horizon == args.obs_steps + args.action_steps - 1

    dataset = RMBenchImageDataset(
        dataset_dir=os.path.expanduser(args.dataset_path),
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        horizon=args.horizon,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        device="cpu",
    )
    print(dataset)

    num_workers = int(getattr(args, "num_workers", 0))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    _move_normalizer_to_device(dataset.normalizer, args.device)

    agent = build_agent(args)
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    n_gradient_step = 0
    resume_step, resume_model_path, resume_scheduler_path = _resolve_resume_checkpoint(args)
    reset_scheduler_on_resume = bool(getattr(args, "reset_scheduler_on_resume", False))
    if resume_model_path is not None:
        if not resume_model_path.exists():
            raise FileNotFoundError(f"Resume model checkpoint not found: {resume_model_path}")
        print(f"Resuming from model checkpoint: {resume_model_path}", flush=True)
        agent.load(str(resume_model_path))
        if reset_scheduler_on_resume:
            _set_optimizer_lr(agent.optimizer, float(args.lr))
            lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
            print(f"Reset scheduler on resume; optimizer lr set to {float(args.lr):.8e}", flush=True)
        elif resume_scheduler_path is not None:
            lr_scheduler.load_state_dict(torch.load(resume_scheduler_path, map_location=args.device, weights_only=False))
            print(f"Resumed scheduler checkpoint: {resume_scheduler_path}", flush=True)
        n_gradient_step = int(resume_step) + 1 if resume_step is not None else 0

    if args.mode != "train":
        raise ValueError("Illegal mode")

    loss_log = collections.defaultdict(list)
    start_time = time.time()
    save_freq = int(getattr(args, "save_freq", 0))
    save_on_exit = bool(getattr(args, "save_on_exit", True))
    try:
        for batch in loop_dataloader(dataloader):
            nobs = batch["obs"]
            condition = {}
            for key, value in nobs.items():
                condition[key] = dataset.normalizer["obs"][key].normalize(value.to(args.device, non_blocking=True))
            naction = dataset.normalizer["action"].normalize(batch["action"].to(args.device, non_blocking=True))

            update_log = agent.update(naction, condition)
            lr_scheduler.step()
            for key, value in update_log.items():
                scalar = _as_float(value)
                if scalar is not None:
                    loss_log[key].append(scalar)

            if n_gradient_step % args.log_freq == 0:
                log_dict = {
                    "step": n_gradient_step,
                    "total_time": time.time() - start_time,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if "loss" in loss_log:
                    log_dict["avg_diffusion_loss"] = np.mean(loss_log["loss"])
                    log_dict["last_diffusion_loss"] = loss_log["loss"][-1]
                logger.log(log_dict, category="train")
                loss_log.clear()

            if save_freq > 0 and n_gradient_step % save_freq == 0 and n_gradient_step > 0:
                _save_checkpoint(logger, agent, lr_scheduler, dataset.normalizer, n_gradient_step)

            if _STOP_REQUESTED:
                break

            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                break
    finally:
        if save_on_exit and (n_gradient_step > 0 or resume_model_path is not None):
            _save_checkpoint(logger, agent, lr_scheduler, dataset.normalizer, n_gradient_step)


if __name__ == "__main__":
    pipeline()
