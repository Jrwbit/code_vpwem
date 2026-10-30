"""Generate accurate offline visible-object boxes from BHR iGibson replay.

The simulator state and segmentation are annotation-only privileged inputs.
They never enter the policy or online detector.  The resulting box is the
tight box of the bowl pixels visible in the same 90x120 RGB view.  Occluded or
out-of-view bowls remain invalid; no previous box is carried forward.

This module is intended for the BHR environment, where ``gibson2`` and the
MoMaRT robomimic wrapper are installed.  It is intentionally importable
without those packages so the cache contract can still be unit-tested here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import zarr

from .interaction import mask_to_box


VERSION = "simulator_visible_mask_v1"


def _env_args(root):
    attrs = root.attrs.get("source_data_attrs", None)
    if attrs is None:
        attrs = root.attrs.get("env_args", {})
    if isinstance(attrs, str):
        attrs = json.loads(attrs)
    raw = attrs.get("env_args", attrs) if isinstance(attrs, dict) else attrs
    if isinstance(raw, str):
        raw = json.loads(raw)
    return raw


def _target_class_id(env_args):
    config = env_args["env_kwargs"]["ig_config"]
    objects = config.get("objects", [])
    if len(objects) != 1 or "class_id" not in objects[0]:
        raise ValueError("Expected one target object with class_id")
    return int(objects[0]["class_id"])


def _goal_metadata(env_args):
    goal = env_args.get("env_kwargs", {}).get("ig_config", {}).get("goal", {})
    return dict(goal_location=goal.get("location"), goal_surface=goal.get("surface"))


def _segmentation_mask(segmentation, class_id, output_hw=(90, 120)):
    seg = np.asarray(segmentation)
    if seg.ndim != 3 or seg.shape[-1] < 1:
        raise ValueError("Unexpected iGibson segmentation shape")
    # iGibson's mesh renderer returns class IDs in the red channel divided by
    # 255, even though the wrapper's generic segmentation path is disabled.
    ids = np.rint(seg[..., 0] * 255).astype(np.int32)
    mask = ids == int(class_id)
    h, w = output_hw
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def annotate_dataset(source, destination, gpu=None, start_frame=0, stop_frame=None,
                     batch_commit=100, state_source=None):
    """Replay one Zarr dataset and write a resumable simulator-GT sidecar."""

    source, destination = Path(source), Path(destination)
    root = zarr.open_group(str(source), mode="r")
    rgb = root["data/rgb"]
    states = root["data/states"] if "data/states" in root else None
    ends = np.asarray(root["meta/episode_ends"])
    if states is not None and len(rgb) != len(states):
        raise ValueError("RGB and state lengths differ")
    h5 = None
    h5_states = None
    if states is None:
        if state_source is None:
            state_source = root.attrs.get("source_path")
        if not state_source:
            raise ValueError("Zarr has no states; pass --state-source HDF5")
        import h5py
        h5 = h5py.File(str(state_source), "r", swmr=True)
        state_source_names = root.attrs.get("demo_names", [])
        if isinstance(state_source_names, str):
            state_source_names = json.loads(state_source_names)
        state_source_names = list(state_source_names)
        if len(state_source_names) != len(ends):
            masks = root.attrs.get("source_masks", {})
            if isinstance(masks, str):
                masks = json.loads(masks)
            train = masks.get("train", []) if isinstance(masks, dict) else []
            if len(train) == len(ends):
                state_source_names = [f"demo_{int(i)}" for i in train]
        if not state_source_names:
            state_source_names = [f"demo_{i}" for i in range(len(ends))]
        if len(state_source_names) != len(ends):
            raise ValueError("HDF5 demo names and Zarr episode ends differ")
        expected_names = []
        previous = 0
        for demo, end in zip(state_source_names, ends):
            length = int(end) - previous
            if len(h5["data"][demo]["states"]) != length:
                raise ValueError(f"State length mismatch for {demo}: expected {length}")
            expected_names.append(demo)
            previous = int(end)
        state_source_names = expected_names
    env_args = _env_args(root)
    class_id = _target_class_id(env_args)
    signature = dict(source=str(source.resolve()), shape=list(rgb.shape),
                     episode_ends_sha256=hashlib.sha256(ends.tobytes()).hexdigest(),
                     annotation_standard=VERSION, class_id=class_id,
                     box_format="normalized_xyxy", annotation_source="simulator_segmentation",
                     online_input="rgb_only", privileged_offline=True,
                     target_resolution=[90, 120], **_goal_metadata(env_args))
    destination.parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open_group(str(destination), mode="a")
    if "signature" in dst.attrs and dst.attrs["signature"] != signature:
        raise ValueError("Existing simulator-GT sidecar has a different signature")
    if "signature" not in dst.attrs:
        count = len(rgb)
        chunk = min(1024, max(1, count))
        for name, shape, dtype in [
            ("boxes", (count, 4), "f4"), ("scores", (count,), "f4"),
            ("valid", (count,), "bool"), ("ambiguous", (count,), "bool"),
            ("candidates", (count,), "i4"), ("visibility_pixels", (count,), "i4"),
        ]:
            dst.create_dataset(name, shape=shape, dtype=dtype, chunks=(chunk,) + shape[1:])
        dst.attrs.update(signature=signature, completed_frames=0, complete=False,
                         annotation_standard=VERSION, target_class_id=class_id,
                         **_goal_metadata(env_args))
    completed = int(dst.attrs.get("completed_frames", 0))
    if completed < 0 or completed > len(rgb):
        raise ValueError("Invalid resume cursor")
    first = max(completed, int(start_frame))
    last = len(rgb) if stop_frame is None else min(len(rgb), int(stop_frame))
    if last < first:
        raise ValueError("stop_frame precedes resume cursor")

    # Importing the simulator is deliberately delayed: local unit tests do
    # not need BHR's iGibson installation.
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    env = EnvGibsonMOMART(
        env_name=env_args["env_name"],
        ig_config=env_args["env_kwargs"]["ig_config"],
        postprocess_visual_obs=False, render=False, render_offscreen=True,
        use_image_obs=True, image_height=480, image_width=640,
    )
    renderer = env.env.simulator.renderer
    started = time.perf_counter()
    current_demo = -1
    current_states = None
    try:
        # Fail before writing labels if the simulator's first robot camera is
        # not the same view as the RGB array used by the policy.
        probe_frame = first
        if states is not None:
            probe_state = np.asarray(states[probe_frame])
        else:
            probe_demo = int(np.searchsorted(ends, probe_frame, side="right"))
            probe_start = 0 if probe_demo == 0 else int(ends[probe_demo - 1])
            probe_state = np.asarray(h5["data"][state_source_names[probe_demo]]["states"][probe_frame - probe_start])
        env.env.reset_to(probe_state, exclude=[])
        rendered_rgb = renderer.render_robot_cameras(modes=["rgb"])
        if not isinstance(rendered_rgb, (list, tuple)) or not rendered_rgb:
            raise RuntimeError("iGibson returned no robot camera RGB")
        probe_rgb = np.asarray(rendered_rgb[0])
        if probe_rgb.ndim == 3 and probe_rgb.shape[-1] > 3:
            probe_rgb = probe_rgb[..., :3]
        if np.issubdtype(probe_rgb.dtype, np.floating) and float(probe_rgb.max()) <= 1.5:
            probe_rgb = probe_rgb * 255.0
        probe = cv2.resize(probe_rgb, (120, 90), interpolation=cv2.INTER_CUBIC)
        mae = float(np.abs(probe.astype(np.float32) - np.asarray(rgb[probe_frame]).astype(np.float32)).mean())
        if mae > 12.0:
            raise RuntimeError(f"Simulator main-camera alignment check failed: MAE={mae:.3f}")
        dst.attrs["rgb_alignment_mae"] = mae
        for frame in range(first, last):
            if states is not None:
                state = np.asarray(states[frame])
            else:
                demo = int(np.searchsorted(ends, frame, side="right"))
                demo_start = 0 if demo == 0 else int(ends[demo - 1])
                if demo != current_demo:
                    current_states = h5["data"][state_source_names[demo]]["states"]
                    current_demo = demo
                state = np.asarray(current_states[frame - demo_start])
            env.env.reset_to(state, exclude=[])
            rendered = renderer.render_robot_cameras(modes=["seg"])
            if not isinstance(rendered, (list, tuple)) or not rendered:
                raise RuntimeError("iGibson returned no robot camera segmentation")
            mask = _segmentation_mask(rendered[0], class_id)
            box = mask_to_box(mask, minimum_pixels=1)
            if box is None:
                dst["boxes"][frame] = np.zeros(4, np.float32)
                dst["valid"][frame] = False
                dst["visibility_pixels"][frame] = 0
            else:
                dst["boxes"][frame] = box
                dst["valid"][frame] = True
                dst["visibility_pixels"][frame] = int(mask.sum())
            dst["scores"][frame] = 1.0 if box is not None else 0.0
            dst["ambiguous"][frame] = False
            dst["candidates"][frame] = 1 if box is not None else 0
            if (frame + 1) % max(1, int(batch_commit)) == 0 or frame + 1 == last:
                dst.attrs["completed_frames"] = frame + 1
                print(json.dumps(dict(frame=frame + 1, total=last,
                    valid=int(np.asarray(dst["valid"][first:frame + 1]).sum()),
                    elapsed_seconds=time.perf_counter() - started)), flush=True)
    finally:
        # The wrapper does not expose close(); process teardown releases the
        # EGL context.  Delete references before a worker exits.
        del renderer, env
        if h5 is not None:
            h5.close()
    if last == len(rgb):
        dst.attrs["complete"] = True
    return dict(source=str(source), destination=str(destination),
                frames=last-first, completed_frames=int(dst.attrs["completed_frames"]),
                complete=bool(dst.attrs.get("complete", False)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--batch-commit", type=int, default=100)
    parser.add_argument("--state-source", help="HDF5 source when Zarr has no states")
    args = parser.parse_args()
    print(json.dumps(annotate_dataset(args.source, args.destination,
        start_frame=args.start_frame, stop_frame=args.stop_frame,
        batch_commit=args.batch_commit, state_source=args.state_source), indent=2))


if __name__ == "__main__":
    main()
