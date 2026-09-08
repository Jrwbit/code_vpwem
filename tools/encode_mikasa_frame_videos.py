#!/usr/bin/env python3
"""Encode per-environment MIKASA rollout frames into numbered MP4 files."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio


FRAME_RE = re.compile(r"loop_(\d+)_env_(\d+)_step_(\d+)\.jpg$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Keep at most this many chronological frames per episode.",
    )
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in args.frames_dir.glob("loop_*_env_*_step_*.jpg"):
        match = FRAME_RE.match(path.name)
        if match:
            loop_id, env_id, step = map(int, match.groups())
            grouped[(loop_id, env_id)].append((step, path))

    if not grouped:
        raise RuntimeError(f"No rollout frames found in {args.frames_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for (loop_id, env_id), frames in sorted(grouped.items()):
        episode_id = loop_id * args.num_envs + env_id
        frames.sort(key=lambda item: item[0])
        if args.max_frames is not None:
            frames = frames[: args.max_frames]
        output_path = args.output_dir / f"episode_{episode_id:03d}.mp4"
        with imageio.get_writer(
            output_path,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for _, frame_path in frames:
                writer.append_data(imageio.imread(frame_path))
        manifest.append(
            {
                "episode_id": episode_id,
                "loop_id": loop_id,
                "env_id": env_id,
                "frame_count": len(frames),
                "first_step": frames[0][0],
                "last_step": frames[-1][0],
                "video": output_path.name,
            }
        )

    manifest_path = args.output_dir / "video_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Encoded {len(manifest)} videos into {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
