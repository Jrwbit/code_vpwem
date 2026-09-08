import argparse

from cleandiffuser.dataset.rmbench_dataset import convert_rmbench_to_replay_zarr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a VPWEM RMBench replay zarr with DP-style obs_t -> qpos_{t+1} actions."
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--chunk_length", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    shape_meta = {
        "action": {"shape": [14]},
        "obs": {
            "head_cam": {
                "shape": [3, 240, 320],
                "type": "rgb",
                "source": "observation/head_camera/rgb",
            },
            "agent_pos": {
                "shape": [14],
                "type": "low_dim",
                "source": "joint_action/vector",
            },
        },
    }
    output_path = convert_rmbench_to_replay_zarr(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        shape_meta=shape_meta,
        chunk_length=args.chunk_length,
        shift_action=True,
    )
    print(f"Wrote DP-aligned RMBench replay zarr: {output_path}", flush=True)


if __name__ == "__main__":
    main()
