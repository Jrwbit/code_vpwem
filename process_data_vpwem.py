import argparse
from pathlib import Path

from omegaconf import OmegaConf

from cleandiffuser.dataset.rmbench_dataset import convert_rmbench_to_replay_zarr


def main():
    parser = argparse.ArgumentParser(description="Preprocess RMBench HDF5 episodes to VPWEM replay zarr.")
    parser.add_argument("--dataset_dir", required=True, help="RMBench dataset directory containing data/episode*.hdf5.")
    parser.add_argument("--config_path", required=True, help="VPWEM YAML config with shape_meta.")
    parser.add_argument("--output_path", required=True, help="Output replay buffer zarr directory.")
    parser.add_argument("--action_source", default="joint_action/vector")
    parser.add_argument("--chunk_length", type=int, default=100)
    parser.add_argument("--no_overwrite", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    output_path = convert_rmbench_to_replay_zarr(
        dataset_dir=Path(args.dataset_dir).expanduser(),
        output_path=Path(args.output_path).expanduser(),
        shape_meta=OmegaConf.to_container(cfg.shape_meta, resolve=True),
        action_source=args.action_source,
        chunk_length=args.chunk_length,
        overwrite=not args.no_overwrite,
    )
    print(f"Saved RMBench replay zarr to {output_path}")


if __name__ == "__main__":
    main()
