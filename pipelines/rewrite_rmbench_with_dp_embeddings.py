import argparse
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RMBenchDPTimestepDataset(Dataset):
    def __init__(self, data_group):
        self.data_group = data_group
        self.rgb_key = "head_cam"
        self.lowdim_key = "agent_pos"
        if self.rgb_key not in data_group:
            raise KeyError(f"Missing /data/{self.rgb_key} in replay zarr")
        if self.lowdim_key not in data_group:
            raise KeyError(f"Missing /data/{self.lowdim_key} in replay zarr")

    def __len__(self):
        return self.data_group["action"].shape[0]

    def __getitem__(self, idx):
        rgb = self.data_group[self.rgb_key][idx]
        rgb = np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0
        return {
            "head_cam": rgb[None],
            "agent_pos": self.data_group[self.lowdim_key][idx].astype(np.float32)[None],
        }


def _resolve_rmbench_root(path):
    if path is not None:
        return pathlib.Path(os.path.expanduser(path)).resolve()
    return pathlib.Path(__file__).resolve().parents[2] / "rmbench"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write RMBench DP obs-encoder embeddings into a VPWEM replay zarr."
    )
    parser.add_argument("--dataset_path", required=True, help="Writable VPWEM replay .zarr path")
    parser.add_argument("--dp_ckpt", required=True, help="RMBench DP .ckpt path")
    parser.add_argument("--embedding_key", default="dp_emb")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rmbench_root", default="/home/wjr/code/rmbench")
    return parser.parse_args()


def main():
    args = parse_args()
    rmbench_root = _resolve_rmbench_root(args.rmbench_root)
    if str(rmbench_root) not in sys.path:
        sys.path.insert(0, str(rmbench_root))

    from policy.VPWEM.dp_embedding import DPEmbeddingEncoder

    dataset_path = pathlib.Path(os.path.expanduser(args.dataset_path))
    if not dataset_path.is_dir() or dataset_path.suffix != ".zarr":
        raise ValueError(f"dataset_path must be a writable replay .zarr directory: {dataset_path}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device={args.device}, but CUDA is not available")

    root = zarr.open(str(dataset_path), mode="r+")
    data_group = root["data"]
    dataset = RMBenchDPTimestepDataset(data_group)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    encoder = DPEmbeddingEncoder(
        ckpt_file=args.dp_ckpt,
        device=args.device,
        rmbench_root=rmbench_root,
    )

    embedding_key = str(args.embedding_key)
    if embedding_key in data_group:
        print(f"Deleting existing /data/{embedding_key}", flush=True)
        del data_group[embedding_key]

    emb_arr = data_group.create_dataset(
        embedding_key,
        shape=(len(dataset), encoder.embedding_dim),
        chunks=(min(4096, len(dataset)), encoder.embedding_dim),
        dtype=np.float32,
        compressor=None,
        overwrite=True,
    )

    write_idx = 0
    for batch in tqdm(dataloader, desc=f"Writing {embedding_key} from DP encoder"):
        with torch.no_grad():
            embeddings = encoder.encode_tensor_batch(batch)[:, 0].detach().cpu().numpy().astype(np.float32)
        next_idx = write_idx + embeddings.shape[0]
        emb_arr[write_idx:next_idx] = embeddings
        write_idx = next_idx

    print(
        f"Done. Wrote /data/{embedding_key} shape={emb_arr.shape} "
        f"using DP ckpt={args.dp_ckpt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
