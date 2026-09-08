import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from cleandiffuser.dataset.rmbench_dataset import RMBenchImageDataset
from utils import set_seed


class RMBenchTimestepDataset(Dataset):
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
            elif obs_type == "low_dim" and not key.endswith("emb"):
                self.lowdim_keys.append(key)

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


@hydra.main(config_path="../configs/dp/rmbench/chi_transformer_ptp", config_name="swap_blocks")
def main(args):
    from omegaconf import OmegaConf

    OmegaConf.set_struct(args, False)
    set_seed(args.seed)
    if not hasattr(args, "model_path") or not args.model_path:
        raise ValueError("Missing model_path. Pass model_path=/path/to/context/model_*.pt")

    dataset_path = pathlib.Path(os.path.expanduser(str(args.dataset_path)))
    if not dataset_path.is_dir() or dataset_path.suffix != ".zarr":
        raise ValueError("rewrite_rmbench_with_embeddings.py requires dataset_path to be a writable replay .zarr")

    dataset = RMBenchImageDataset(
        dataset_dir=dataset_path,
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        horizon=args.horizon,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        device="cpu",
    )

    from pipelines.dpptp_rmbench_image import build_agent

    agent = build_agent(args)
    model_path = pathlib.Path(os.path.expanduser(str(args.model_path)))
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    agent.load(str(model_path))
    agent.model_ema["condition"].eval()

    root = zarr.open(str(dataset_path), mode="r+")
    data_group = root["data"]
    embedding_key = str(getattr(args, "embedding_key", "dp_emb"))
    if embedding_key in data_group:
        print(f"Deleting existing /data/{embedding_key}", flush=True)
        del data_group[embedding_key]

    batch_size = int(getattr(args, "embedding_batch_size", args.batch_size))
    timestep_dataset = RMBenchTimestepDataset(dataset.replay_buffer, args.shape_meta, dataset.normalizer)
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
            obs = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
            embeddings = agent.model_ema["condition"](obs).detach().cpu().numpy()
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

    print(f"Done. Wrote /data/{embedding_key} with shape {emb_arr.shape} to {dataset_path}", flush=True)


if __name__ == "__main__":
    main()
