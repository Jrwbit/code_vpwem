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
from cleandiffuser.dataset.dataset_utils import dynamic_batch_collate_fn, loop_dataloader
from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer
from cleandiffuser.dataset.rmbench_dataset import LongMemRMBenchImageDataset
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


def _lowdim_obs_keys(args):
    return [
        key
        for key, attr in args.shape_meta["obs"].items()
        if attr.get("type", "low_dim") == "low_dim"
    ]


def _lowdim_obs_dim(args):
    return sum(int(args.shape_meta["obs"][key]["shape"][0]) for key in _lowdim_obs_keys(args))


def _concat_lowdim_obs(nobs, normalizer, args, suffix=""):
    chunks = []
    for key in _lowdim_obs_keys(args):
        value = nobs[f"{key}{suffix}"].to(args.device, non_blocking=True)
        chunks.append(normalizer["obs"][key].normalize(value))
    if not chunks:
        raise ValueError("longmemdpptp_rmbench_image_emb requires at least one low-dim obs key")
    return torch.cat(chunks, dim=-1)


def _uses_current_delta(args):
    return str(getattr(args, "action_representation", "absolute")) == "current_delta"


def _current_delta_actions(batch, args):
    actions = batch["action"].to(args.device, non_blocking=True)
    if not _uses_current_delta(args):
        return actions
    current = batch["obs"]["agent_pos"][:, int(args.obs_steps) - 1].to(
        args.device, non_blocking=True
    )
    return actions - current[:, None, :]


def _fit_current_delta_normalizer(dataset):
    chunks = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        current = sample["obs"]["agent_pos"][dataset.n_obs_steps - 1]
        chunks.append(sample["action"] - current[None, :])
    values = torch.cat(chunks, dim=0).numpy()
    return MinMaxNormalizer(values, device="cpu")


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


def build_agent(args, normalizer):
    if args.nn != "chi_transformer":
        raise ValueError(f"Invalid nn type {args.nn}")

    from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP

    obs_dim = _lowdim_obs_dim(args)
    nn_diffusion = LongMemChiTransformerPTP(
        args.action_dim,
        obs_dim,
        args.horizon,
        args.obs_steps,
        args.mem_compress_length,
        d_model=256,
        nhead=4,
        num_layers=int(getattr(args, "diffusion_num_layers", 4)),
        p_drop_emb=args.p_drop_emb,
        p_drop_attn=args.p_drop_attn,
        timestep_emb_type="positional",
    ).to(args.device)

    from cleandiffuser.nn_memory.adjsim_memory import AdjSimMemory
    from cleandiffuser.nn_memory.kmeans_memory import KMeansMemory
    from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
    from cleandiffuser.nn_memory.random_memory import RandomMemory

    if args.mem_compress_method == "random":
        nn_memory = RandomMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "kmeans":
        nn_memory = KMeansMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "adjsim":
        nn_memory = AdjSimMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "qformer":
        nn_memory = QFormerMemory(
            mem_compress_length=args.mem_compress_length,
            emb_dim=args.qformer_emb_dim,
            num_layers=args.qformer_num_layers,
            num_heads=int(getattr(args, "qformer_num_heads", 8)),
            cache_compress=args.qformer_cache_compress,
            cache_max_length=args.qformer_cache_max_length,
            use_pos_emb=args.memory_pos_emb,
        ).to(args.device)
    else:
        raise ValueError(f"Invalid memory method {args.mem_compress_method}")

    report_parameters(nn_diffusion)
    report_parameters(nn_memory)

    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device)
    x_min = -x_max
    action_range = normalizer["action"].range.to(args.device)
    action_min = normalizer["action"].min.to(args.device)

    if args.diffusion != "ddpm":
        raise NotImplementedError(f"Diffusion method {args.diffusion} not supported for RMBench VPWEM")

    from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM

    return LongMemDDPM(
        nn_diffusion=nn_diffusion,
        nn_condition=None,
        nn_memory=nn_memory,
        device=args.device,
        diffusion_steps=args.sample_steps,
        x_max=x_max,
        x_min=x_min,
        args_dict=args,
        action_range=action_range,
        action_min=action_min,
        short_cond_dropout=args.short_cond_dropout,
        long_cond_dropout=args.long_cond_dropout,
        optim_params={"lr": args.lr},
    )


@hydra.main(config_path="../configs/dp/rmbench/chi_transformer_ptp_longmem", config_name="swap_blocks_emb")
def pipeline(args):
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)
    assert args.horizon == args.obs_steps + args.action_steps - 1

    dataset = LongMemRMBenchImageDataset(
        dataset_dir=os.path.expanduser(args.dataset_path),
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        horizon=args.horizon,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        dataset_memory_type=args.dataset_memory_type,
        subsample_ratio=args.dataset_memory_subsample_ratio,
        maximum_memory_length=args.qformer_cache_max_length,
        load_lowdim_to_memory=bool(getattr(args, "load_lowdim_to_memory", True)),
        device="cpu",
    )
    print(dataset)

    if _uses_current_delta(args):
        dataset.normalizer["action"] = _fit_current_delta_normalizer(dataset)
        print(
            "Using current-relative action targets; "
            f"delta min={dataset.normalizer['action'].min}, "
            f"delta max={dataset.normalizer['action'].max}",
            flush=True,
        )

    train_sampler = dataset.get_dataset_sampler()
    num_workers = int(getattr(args, "num_workers", 0))
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
    _move_normalizer_to_device(dataset.normalizer, args.device)

    agent = build_agent(args, dataset.normalizer)
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
            scheduler_t_max = int(getattr(args, "resume_scheduler_t_max", args.gradient_steps))
            lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=scheduler_t_max, last_epoch=-1)
            print(
                "Reset scheduler on resume; "
                f"optimizer lr set to {float(args.lr):.8e}; "
                f"scheduler T_max={scheduler_t_max}",
                flush=True,
            )
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
            condition = _concat_lowdim_obs(nobs, dataset.normalizer, args)[:, : args.obs_steps]
            memory = _concat_lowdim_obs(nobs, dataset.normalizer, args, suffix="_mem")
            naction = dataset.normalizer["action"].normalize(_current_delta_actions(batch, args))
            mem_steps = nobs["ep_step"].int().to(args.device, non_blocking=True)

            update_log = agent.update(naction, condition, memory, mem_steps)
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
                for key, values in loss_log.items():
                    if key == "loss":
                        continue
                    log_dict[f"avg_{key}"] = np.mean(values)
                    log_dict[f"last_{key}"] = values[-1]
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
