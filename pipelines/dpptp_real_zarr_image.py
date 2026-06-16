import collections
import math
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

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.dataset.real_zarr_image_dataset import RealZarrImageDataset
from cleandiffuser.utils import report_parameters
from utils import Logger, set_seed

import robomimic.utils.obs_utils as ObsUtils


@hydra.main(
    config_path="../configs/dp/real_zarr/chi_transformer_ptp",
    config_name="block_2cam",
)
def pipeline(args):
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)
    assert args.horizon == args.obs_steps + args.action_steps - 1

    modality_mapping = collections.defaultdict(list)
    for key, attr in args.shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    dataset = RealZarrImageDataset(
        args.dataset_path,
        horizon=args.horizon,
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
    )
    print(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    if args.nn == "chi_transformer":
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
    else:
        raise ValueError(f"Invalid nn type {args.nn}")

    print("======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print("===================================================================================")
    print("======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print("===================================================================================")

    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device)
    x_min = -x_max
    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.ddpm import DDPM

        agent = DDPM(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            device=args.device,
            diffusion_steps=args.sample_steps,
            x_max=x_max,
            x_min=x_min,
            optim_params={"lr": args.lr},
        )
    else:
        raise NotImplementedError(f"Unsupported diffusion: {args.diffusion}")

    n_gradient_step = 0
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)

    if args.mode != "train":
        raise ValueError("Illegal mode")

    diffusion_loss_list = []
    save_loss_list = []
    saved_checkpoints = []
    start_time = time.time()
    for batch in loop_dataloader(dataloader):
        nobs = batch["obs"]
        condition = {
            key: value[:, : args.obs_steps, :].to(args.device)
            for key, value in nobs.items()
        }
        naction = batch["action"].to(args.device)

        diffusion_loss = agent.update(naction, condition)["loss"]
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
            save_loss = float(np.mean(save_loss_list)) if save_loss_list else math.inf
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
