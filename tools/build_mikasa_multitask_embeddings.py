"""Build four-task MIKASA embeddings from one shared DPPTP EMA encoder."""

import argparse
import os
import pathlib

import numpy as np
import torch


TASKS = (
    "RememberColor3-v0",
    "ShellGameTouch-v0",
    "RememberColor5-v0",
    "InterceptGrabMedium-v0",
)


def shape_meta():
    return {
        "action": {"shape": [8]},
        "obs": {
            "rgb": {"shape": [3, 128, 128], "type": "rgb"},
            "rgb_wrist": {"shape": [3, 128, 128], "type": "rgb"},
            "joints": {"shape": [25], "type": "low_dim"},
        },
    }


def build_encoder(checkpoint, device):
    from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
    encoder = MultiImageObsConditionRobomimic(
        shape_meta=shape_meta(), emb_dim=256, rgb_model_name="resnet18",
        resize_shape=None, crop_shape=[100, 100], random_crop=True,
        use_group_norm=True, use_seq=True,
        keep_horizon_dims=True).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    condition_state = {
        key[len("condition."):]: value
        for key, value in state["model_ema"].items()
        if key.startswith("condition.")
    }
    if not condition_state:
        raise KeyError("checkpoint has no EMA condition encoder parameters")
    encoder.load_state_dict(condition_state, strict=True)
    encoder.eval()
    return encoder


def joint_range(dataset_dir):
    lower = np.full(25, np.inf, dtype=np.float32)
    upper = np.full(25, -np.inf, dtype=np.float32)
    for index in range(1000):
        path = dataset_dir / f"train_data_{index}.npz"
        if not path.exists():
            break
        with np.load(path) as demo:
            success = np.flatnonzero(demo["success"] == 1)
            length = int(success[0] + 1) if len(success) else 0
            if length:
                joints = demo["joints"][:length].astype(np.float32)
                lower = np.minimum(lower, joints.min(axis=0))
                upper = np.maximum(upper, joints.max(axis=0))
    if not np.isfinite(lower).all():
        raise ValueError(f"no successful episodes in {dataset_dir}")
    span = upper - lower
    span[span == 0] = 1
    return lower, span


def encode_task(encoder, dataset_dir, device, overwrite=False):
    lower, span = joint_range(dataset_dir)
    written = skipped = 0
    for index in range(1000):
        source = dataset_dir / f"train_data_{index}.npz"
        if not source.exists():
            break
        target = dataset_dir / f"emb_{index}.npz"
        if target.exists() and not overwrite:
            skipped += 1
            continue
        with np.load(source) as demo:
            images = demo["rgb"].astype(np.float32) / 255.0
            joints = demo["joints"].astype(np.float32)
        obs = {
            "rgb": torch.from_numpy(
                np.moveaxis(images[..., :3], -1, 1)[None] * 2.0 - 1.0
            ).to(device),
            "rgb_wrist": torch.from_numpy(
                np.moveaxis(images[..., 3:], -1, 1)[None] * 2.0 - 1.0
            ).to(device),
            "joints": torch.from_numpy(
                (((joints - lower) / span) * 2.0 - 1.0)[None]
            ).to(device),
        }
        with torch.inference_mode():
            embedding = encoder(obs).float().cpu().numpy()
        temporary = target.with_suffix(".tmp.npz")
        np.savez(temporary, dp_emb=embedding)
        os.replace(temporary, target)
        written += 1
        if written % 50 == 0:
            print(
                f"{dataset_dir.name}: written={written} skipped={skipped}",
                flush=True)
    print(
        f"COMPLETE {dataset_dir.name}: written={written} skipped={skipped}",
        flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="/home/jrw/vpwem_data/mikasa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(100)
    encoder = build_encoder(args.checkpoint, args.device)
    for task in TASKS:
        encode_task(
            encoder, pathlib.Path(args.data_root) / task, args.device,
            overwrite=args.overwrite)


if __name__ == "__main__":
    main()
