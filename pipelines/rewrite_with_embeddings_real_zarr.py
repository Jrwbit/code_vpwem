import collections
import pathlib
import sys
import warnings

import hydra
import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from cleandiffuser.dataset.real_zarr_image_dataset import (
    RealZarrImageDataset,
    resolve_replay_buffer_path,
)
from utils import set_seed

import robomimic.utils.obs_utils as ObsUtils


class RealZarrTimestepDataset(Dataset):
    def __init__(self, replay_buffer, shape_meta, normalizer):
        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.normalizer = normalizer

        self.rgb_keys = []
        self.lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type '{obs_type}' for key '{key}'")

    def __len__(self):
        return self.replay_buffer["action"].shape[0]

    def __getitem__(self, idx):
        obs = {}
        for key in self.rgb_keys:
            value = self.replay_buffer[key][idx]
            value = np.moveaxis(value, -1, 0).astype(np.float32) / 255.0
            value = value[None]
            obs[key] = self.normalizer["obs"][key].normalize(value)

        for key in self.lowdim_keys:
            value = self.replay_buffer[key][idx].astype(np.float32)
            value = value[None]
            obs[key] = self.normalizer["obs"][key].normalize(value)
        return obs


def _create_agent(args):
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

    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device)
    x_min = -x_max
    if args.diffusion != "ddpm":
        raise NotImplementedError(f"Unsupported diffusion: {args.diffusion}")
    return DDPM(
        nn_diffusion=nn_diffusion,
        nn_condition=nn_condition,
        device=args.device,
        diffusion_steps=args.sample_steps,
        x_max=x_max,
        x_min=x_min,
        optim_params={"lr": args.lr},
    )


@hydra.main(
    config_path="../configs/dp/real_zarr/chi_transformer_ptp",
    config_name="block_2cam",
)
def main(args):
    set_seed(args.seed)

    modality_mapping = collections.defaultdict(list)
    for key, attr in args.shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    if not hasattr(args, "model_path"):
        raise ValueError("Missing model_path. Pass model_path=/path/to/model_*.pt")

    dataset = RealZarrImageDataset(
        args.dataset_path,
        horizon=args.horizon,
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
    )

    agent = _create_agent(args)
    model_path = pathlib.Path(str(args.model_path)).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    agent.load(str(model_path))
    agent.model_ema.condition.eval()

    zarr_path = resolve_replay_buffer_path(args.dataset_path)
    root = zarr.open(str(zarr_path), mode="r+")
    data_group = root["data"]
    replay_buffer = dataset.replay_buffer

    embedding_key = str(getattr(args, "embedding_key", "dp_emb"))
    if embedding_key in data_group:
        print(f"Deleting existing /data/{embedding_key}")
        del data_group[embedding_key]

    batch_size = int(getattr(args, "embedding_batch_size", args.batch_size))
    timestep_dataset = RealZarrTimestepDataset(replay_buffer, args.shape_meta, dataset.normalizer)
    dataloader = DataLoader(
        timestep_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=True,
    )

    emb_arr = None
    write_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Writing {embedding_key}"):
            obs = {key: value.to(args.device) for key, value in batch.items()}
            embeddings = agent.model_ema.condition(obs).detach().cpu().numpy()
            if embeddings.ndim == 3 and embeddings.shape[1] == 1:
                embeddings = embeddings[:, 0]
            embeddings = embeddings.astype(np.float32)

            if emb_arr is None:
                emb_arr = data_group.create_dataset(
                    embedding_key,
                    shape=(len(timestep_dataset), embeddings.shape[-1]),
                    chunks=(min(4096, len(timestep_dataset)), embeddings.shape[-1]),
                    dtype=np.float32,
                    compressor=None,
                    overwrite=True,
                )

            next_idx = write_idx + embeddings.shape[0]
            emb_arr[write_idx:next_idx] = embeddings
            write_idx = next_idx

    print(f"Done. Wrote /data/{embedding_key} with shape {emb_arr.shape} to {zarr_path}")


if __name__ == "__main__":
    main()
