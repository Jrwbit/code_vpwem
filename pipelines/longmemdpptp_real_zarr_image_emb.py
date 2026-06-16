import pathlib
import sys
import time
import warnings

import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from cleandiffuser.dataset.dataset_utils import dynamic_batch_collate_fn, loop_dataloader
from cleandiffuser.dataset.real_zarr_image_dataset import LongMemRealZarrImageDataset
from cleandiffuser.utils import report_parameters
from utils import Logger, set_seed


@hydra.main(
    config_path="../configs/dp/real_zarr/chi_transformer_ptp_longmem",
    config_name="block_2cam_emb",
)
def pipeline(args):
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)
    assert args.horizon == args.obs_steps + args.action_steps - 1

    dataset = LongMemRealZarrImageDataset(
        args.dataset_path,
        horizon=args.horizon,
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        dataset_memory_type=args.dataset_memory_type,
        subsample_ratio=args.dataset_memory_subsample_ratio,
        maximum_memory_length=args.maximum_memory_length,
        device=args.device,
    )
    print(dataset)

    train_sampler = dataset.get_dataset_sampler()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=dynamic_batch_collate_fn,
    )

    if args.nn == "chi_transformer":
        from cleandiffuser.nn_diffusion.longmem_chitransformerptp import (
            LongMemChiTransformerPTP,
        )

        nn_diffusion = LongMemChiTransformerPTP(
            args.action_dim,
            256,
            args.horizon,
            args.obs_steps,
            args.mem_compress_length,
            d_model=256,
            nhead=4,
            num_layers=4,
            p_drop_emb=args.p_drop_emb,
            p_drop_attn=args.p_drop_attn,
            timestep_emb_type="positional",
        ).to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")

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
            cache_compress=args.qformer_cache_compress,
            cache_max_length=args.qformer_cache_max_length,
            use_pos_emb=args.memory_pos_emb,
        ).to(args.device)
    else:
        raise ValueError(f"Invalid memory method {args.mem_compress_method}")

    if args.diffusion != "ddpm":
        raise NotImplementedError(f"Unsupported diffusion: {args.diffusion}")

    from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM

    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device)
    x_min = -x_max
    action_range = dataset.normalizer["action"].range.to(args.device)
    action_min = dataset.normalizer["action"].min.to(args.device)
    agent = LongMemDDPM(
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

    print("======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print("===================================================================================")
    print("======================= Parameter Report of Memory Model =======================")
    report_parameters(nn_memory)
    print("===================================================================================")

    if args.mode != "train":
        raise ValueError("Illegal mode")

    n_gradient_step = 0
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    diffusion_loss_list = []
    save_loss_list = []
    saved_checkpoints = []
    start_time = time.time()

    for batch in loop_dataloader(dataloader):
        nobs = batch["obs"]
        condition = nobs["dp_emb"][:, : args.obs_steps, :].to(args.device)
        naction = batch["action"].to(args.device)
        naction = dataset.normalizer["action"].normalize(naction)
        memory = nobs["dp_emb_mem"].to(args.device)
        mem_steps = nobs["ep_step"].to(args.device)

        diffusion_loss = agent.update(naction, condition, memory, mem_steps)["loss"]
        lr_scheduler.step()
        diffusion_loss_list.append(diffusion_loss)
        save_loss_list.append(diffusion_loss)

        if n_gradient_step % args.log_freq == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "avg_diffusion_loss": np.mean(diffusion_loss_list),
            }
            logger.log(metrics, category="train")
            diffusion_loss_list = []

        if n_gradient_step % args.save_freq == 0 and n_gradient_step > 0:
            save_loss = float(np.mean(save_loss_list)) if save_loss_list else np.inf
            logger.save_agent(agent=agent, identifier=n_gradient_step)
            model_path = pathlib.Path(args.work_dir) / "models" / f"model_{n_gradient_step}.pt"
            scheduler_path = pathlib.Path(args.work_dir) / "models" / f"scheduler_{n_gradient_step}.pt"
            torch.save(lr_scheduler.state_dict(), scheduler_path)
            saved_checkpoints.append(
                {
                    "step": n_gradient_step,
                    "loss": save_loss,
                    "model_path": model_path,
                    "scheduler_path": scheduler_path,
                }
            )
            save_loss_list = []

            save_top_k = int(getattr(args, "save_top_k", 0) or 0)
            if save_top_k > 0 and len(saved_checkpoints) > save_top_k:
                saved_checkpoints = sorted(saved_checkpoints, key=lambda item: item["loss"])
                stale = saved_checkpoints[save_top_k:]
                saved_checkpoints = saved_checkpoints[:save_top_k]
                for item in stale:
                    for path_key in ("model_path", "scheduler_path"):
                        path = item[path_key]
                        if path.exists():
                            path.unlink()
                    print(
                        f"removed checkpoint step={item['step']} "
                        f"loss={item['loss']:.6f} outside top-{save_top_k}"
                    )

        n_gradient_step += 1
        if n_gradient_step > args.gradient_steps:
            logger.finish()
            break


if __name__ == "__main__":
    pipeline()
