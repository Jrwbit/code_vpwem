"""Detector-only evaluation on live MoMaRT action segments, without a policy.

Held-out HDF5 states initialize segments; subsequent frames come from physics
steps using recorded actions. RGB and class segmentation are rendered together.
Only RGB is passed to the detector. Results describe these sampled segments,
not closed-loop policy success or accuracy on the original recorded RGB.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .detector import BowlDetector, QualityGatedDetector
from .interaction import mask_to_box


def overlap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
    return float(intersection / max(float(union), 1e-12))


def summarize(rows, prefix="gated", min_pixels=1):
    """Missing detections stay in the visible-frame hit-rate denominator."""
    visible = [r for r in rows if r["visibility_pixels"] >= min_pixels]
    absent = [r for r in rows if r["visibility_pixels"] == 0]
    predicted = [r for r in rows if r[prefix + "_valid"]]
    paired = [r for r in visible if r[prefix + "_valid"]]
    hits = sum(r[prefix + "_valid"] and r[prefix + "_iou"] >= .5 for r in visible)
    missing = sum(not r[prefix + "_valid"] for r in visible)
    fp = sum(r[prefix + "_valid"] for r in absent)
    # Runs must never span independent initialization segments.
    longest = run = 0
    previous = None
    for r in rows:
        segment = (r["demo"], r["segment"])
        if segment != previous:
            run = 0
        run = run + 1 if r["visibility_pixels"] >= min_pixels and not r[prefix + "_valid"] else 0
        longest = max(longest, run)
        previous = segment
    ratio = lambda a, b: a / b if b else None
    return dict(frames=len(rows), visible_frames=len(visible), absent_frames=len(absent),
                predicted_frames=len(predicted), iou50_hits=int(hits),
                hit_rate=ratio(hits, len(visible)), missing_rate=ratio(missing, len(visible)),
                absent_false_positive_rate=ratio(fp, len(absent)),
                localization_precision=ratio(hits, len(predicted)) if min_pixels == 1 else None,
                paired_mean_iou=float(np.mean([r[prefix + "_iou"] for r in paired])) if paired else None,
                longest_visible_missing_samples=longest)


def evaluate(args):
    import cv2
    import h5py
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    from .annotate_sim_gt import _segmentation_mask, _target_class_id

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    started = time.perf_counter()
    with h5py.File(args.source, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        class_id = _target_class_id(env_args)
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["mask/valid"][:]]
        train = {n.decode() if isinstance(n, bytes) else str(n) for n in f["mask/train"][:]}
        if set(names) & train or not names:
            raise ValueError("Expected disjoint, non-empty held-out episodes")
        rng = np.random.default_rng(args.seed)
        names = list(rng.choice(sorted(names), size=min(args.episodes, len(names)), replace=False))
        env = EnvGibsonMOMART(env_name=env_args["env_name"],
            ig_config=env_args["env_kwargs"]["ig_config"], postprocess_visual_obs=False,
            render=False, render_offscreen=True, use_image_obs=True)
        env.env.hard_reset = False
        renderer = env.env.simulator.renderer
        detector = BowlDetector(device=args.device, local_files_only=True)
        gated = QualityGatedDetector(detector, threshold=args.threshold)
        rows = []
        alignment_maes = []
        with (out / "frames.jsonl").open("w") as stream:
            for demo in names:
                episode = f["data"][demo]
                n = len(episode["actions"])
                horizon = args.samples * args.stride
                starts = np.linspace(0, max(0, n-horizon), args.segments, dtype=int)
                for segment, start in enumerate(starts):
                    env.env.reset_to(np.asarray(episode["states"][start]), exclude=[])
                    images, masks, indices = [], [], []
                    for offset in range(min(horizon, n-int(start))):
                        if offset % args.stride == 0:
                            # Adjacent outputs are RGB, seg for the same main camera.
                            rendered = renderer.render_robot_cameras(modes=["rgb", "seg"])
                            rgb = (np.asarray(rendered[0])[..., :3] * 255).astype(np.uint8)
                            rgb = cv2.resize(rgb, (120, 90), interpolation=cv2.INTER_CUBIC)
                            if not images:
                                # Check the actual online sensor path, not recorded RGB.
                                sensor = np.asarray(env.get_observation()["rgb"])
                                sensor = cv2.resize(sensor, (120, 90), interpolation=cv2.INTER_CUBIC)
                                mae = float(np.abs(sensor.astype(float)-rgb.astype(float)).mean())
                                if mae > 1.0:
                                    raise RuntimeError(f"Live sensor/camera mismatch: MAE={mae}")
                                alignment_maes.append(mae)
                            mask = _segmentation_mask(rendered[1], class_id)
                            images.append(rgb); masks.append(mask); indices.append(int(start)+offset)
                        _, _, done, _ = env.step(np.asarray(episode["actions"][int(start)+offset]))
                        if done:
                            break
                    raw = detector(images)
                    accepted = gated.filter(raw)
                    for j, (image, mask, d, g, index) in enumerate(zip(images, masks, raw, accepted, indices)):
                        gt = mask_to_box(mask, minimum_pixels=1)
                        row = dict(demo=demo, segment=segment, action_index=index,
                                   visibility_pixels=int(mask.sum()), gt_box=gt,
                                   raw_valid=d.valid, gated_valid=g.valid, score=d.score,
                                   ambiguous=d.ambiguous, raw_box=d.box.tolist(), gated_box=g.box.tolist(),
                                   raw_iou=overlap(d.box, gt) if d.valid and gt is not None else 0.,
                                   gated_iou=overlap(g.box, gt) if g.valid and gt is not None else 0.)
                        rows.append(row); stream.write(json.dumps(row)+"\n")
                        # Keep all small images/masks for reproducible visual auditing.
                        cv2.imwrite(str(out / f"{demo}_{segment}_{j}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(out / f"{demo}_{segment}_{j}_mask.png"), mask.astype(np.uint8)*255)
                    stream.flush()
                    print(json.dumps(dict(demo=demo, segment=segment, frames=len(rows),
                                          elapsed_seconds=time.perf_counter()-started)), flush=True)
        result = dict(source=str(Path(args.source).resolve()), seed=args.seed, episodes=names,
                      protocol="heldout_state_initialized_live_action_segments", policy_used=False,
                      image_hw=[90,120], samples_per_segment=args.samples, stride=args.stride,
                      live_sensor_alignment_max_mae=max(alignment_maes),
                      detector=gated.metadata, raw=summarize(rows, "raw"), gated=summarize(rows),
                      gated_min3pixels=summarize(rows, min_pixels=3),
                      elapsed_seconds=time.perf_counter()-started)
        (out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--segments", type=int, default=5)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--threshold", type=float, default=.35)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if min(args.episodes, args.segments, args.samples, args.stride) < 1:
        p.error("sampling counts must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
