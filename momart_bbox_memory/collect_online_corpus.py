"""Collect causal RGB streams with separate privileged evaluation masks.

Fit/development use disjoint training episodes. Final test uses validation
episodes excluding the previous 20261020 audit. No policy model is loaded.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def split_episodes(train, valid, previous, seed=20261021):
    train, valid = sorted(set(train)), sorted(set(valid))
    if set(train) & set(valid):
        raise ValueError("Train/validation episodes overlap")
    rng = np.random.default_rng(seed)
    if len(train) < 5 or len(set(valid)-set(previous)) < 3:
        raise ValueError("Insufficient disjoint episodes")
    chosen = list(rng.choice(train, 5, replace=False))
    test = list(rng.choice(sorted(set(valid)-set(previous)), 3, replace=False))
    return dict(fit=chosen[:3], dev=chosen[3:], test=test)


def collect(args):
    import cv2
    import h5py
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    from .annotate_sim_gt import _segmentation_mask, _target_class_id
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    previous = json.loads(Path(args.previous).read_text())["episodes"]
    started = time.perf_counter()
    decode = lambda names: [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    with h5py.File(args.source, "r") as f:
        splits = split_episodes(decode(f["mask/train"][:]), decode(f["mask/valid"][:]), previous, args.seed)
        meta = json.loads(f["data"].attrs["env_args"])
        class_id = _target_class_id(meta)
        env = EnvGibsonMOMART(env_name=meta["env_name"], ig_config=meta["env_kwargs"]["ig_config"],
            postprocess_visual_obs=False, render=False, render_offscreen=True, use_image_obs=True)
        env.env.hard_reset = False
        renderer = env.env.simulator.renderer
        manifest = dict(source=args.source, seed=args.seed, episodes=splits, previous_test=previous,
                        image_hw=[90,120], stride=1, segments=5, policy_used=False,
                        protocol="causal_live_action_segments", streams=[], complete=False)
        for split, demos in splits.items():
            folder = out / split
            folder.mkdir(exist_ok=True)
            for demo in demos:
                ep = f["data"][demo]
                samples = 16 if split == "test" else 12
                starts = np.linspace(0, max(0, len(ep["actions"])-samples), 5, dtype=int)
                for segment, start in enumerate(starts):
                    env.env.reset_to(np.asarray(ep["states"][start]), exclude=[])
                    images, masks = [], []
                    for offset in range(min(samples, len(ep["actions"])-int(start))):
                        rendered = renderer.render_robot_cameras(modes=["rgb", "seg"])
                        rgb = cv2.resize((np.asarray(rendered[0])[...,:3]*255).astype(np.uint8),
                                         (120,90), interpolation=cv2.INTER_CUBIC)
                        if offset == 0:
                            sensor = cv2.resize(np.asarray(env.get_observation()["rgb"]),
                                                (120,90), interpolation=cv2.INTER_CUBIC)
                            mae = float(np.abs(sensor.astype(float)-rgb.astype(float)).mean())
                            if mae > 1:
                                raise RuntimeError(f"Online RGB mismatch: {mae}")
                        images.append(rgb)
                        masks.append(_segmentation_mask(rendered[1], class_id))
                        _, _, done, _ = env.step(np.asarray(ep["actions"][int(start)+offset]))
                        if done:
                            break
                    name = f"{demo}_{segment}.npz"
                    np.savez_compressed(folder/name, rgb=np.stack(images), mask=np.stack(masks))
                    manifest["streams"].append(dict(split=split, demo=demo, segment=segment,
                        start=int(start), frames=len(images), file=f"{split}/{name}", sensor_mae=mae))
                print(json.dumps(dict(split=split, demo=demo, streams=len(manifest["streams"]),
                                      elapsed=time.perf_counter()-started)), flush=True)
        manifest["complete"] = True
        manifest["elapsed_seconds"] = time.perf_counter()-started
        (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps(dict(complete=True, output=str(out), elapsed=manifest["elapsed_seconds"])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--previous", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261021)
    collect(p.parse_args())


if __name__ == "__main__": main()
