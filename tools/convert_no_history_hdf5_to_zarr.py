#!/usr/bin/env python3
"""Convert MoMaRT or Robomimic raw-image HDF5 data to persistent Zarr."""

import argparse
import fcntl
import json
import os
import pathlib
import shutil

import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from omegaconf import OmegaConf


FORMAT_VERSION = 1
COMPRESSOR = Blosc(
    cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("momart", "robomimic"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def complete_cache(path, kind, source):
    if not path.is_dir():
        return False
    try:
        root = zarr.open_group(str(path), mode="r")
        return (
            root.attrs.get("vpwem_cache_complete") is True
            and root.attrs.get("vpwem_cache_format_version") == FORMAT_VERSION
            and root.attrs.get("vpwem_cache_kind") == kind
            and root.attrs.get("source_size") == source.stat().st_size
            and root.attrs.get("source_mtime_ns") == source.stat().st_mtime_ns
        )
    except Exception:
        return False


def prepare_paths(source, target, kind, force):
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.suffix != ".zarr":
        raise ValueError(f"target must end in .zarr: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if complete_cache(target, kind, source) and not force:
        # Upgrade existing caches without rewriting any observation arrays.
        copy_env_metadata(source, zarr.open_group(str(target), mode="a"))
        print(f"SKIP complete cache: {target}", flush=True)
        return source, target, None
    partial = target.with_name(target.name + ".partial")
    for candidate in (partial, target if force else None):
        if candidate is not None and candidate.exists():
            if candidate.parent != target.parent:
                raise RuntimeError(f"refusing unsafe cleanup: {candidate}")
            shutil.rmtree(candidate)
    if target.exists():
        raise FileExistsError(
            f"incomplete target exists; rerun with --force: {target}"
        )
    return source, target, partial


def copy_env_metadata(source, root):
    with h5py.File(source, "r") as handle:
        env_args = handle["data"].attrs.get("env_args")
        if env_args is not None:
            root.attrs["env_args"] = json.dumps(json.loads(env_args))


def create_array(group, name, shape, dtype):
    frame_bytes = int(np.prod(shape[1:]) * np.dtype(dtype).itemsize)
    chunk_length = max(1, min(64, (2 * 1024 * 1024) // max(frame_bytes, 1)))
    chunks = (min(chunk_length, shape[0]),) + tuple(shape[1:])
    return group.empty(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=COMPRESSOR,
        overwrite=True,
    )


def selected_demos(handle):
    if "mask/train" in handle:
        return [item.decode("utf-8") for item in handle["mask/train"][:]]
    return sorted(
        handle["data"].keys(), key=lambda item: int(item.split("_")[-1])
    )


def convert_momart(source, partial, config_path):
    config = OmegaConf.load(config_path)
    obs_keys = tuple(config.shape_meta.obs.keys())
    with h5py.File(source, "r", swmr=True, libver="latest") as handle:
        demos = selected_demos(handle)
        lengths = [int(handle[f"data/{demo}"].attrs["num_samples"]) for demo in demos]
        episode_ends = np.cumsum(lengths, dtype=np.int64)
        n_steps = int(episode_ends[-1])
        root = zarr.open_group(str(partial), mode="w")
        data = root.create_group("data")
        meta = root.create_group("meta")
        meta.array(
            "episode_ends", episode_ends, compressor=None, overwrite=True
        )
        arrays = {}
        first = handle[f"data/{demos[0]}"]
        for key in obs_keys:
            item = first[f"obs/{key}"]
            arrays[key] = create_array(
                data, key, (n_steps,) + item.shape[1:], item.dtype
            )
        actions = first["actions"]
        arrays["action"] = create_array(
            data, "action", (n_steps,) + actions.shape[1:], np.float32
        )
        offset = 0
        for index, (demo, length) in enumerate(zip(demos, lengths), 1):
            group = handle[f"data/{demo}"]
            end = offset + length
            for key in obs_keys:
                arrays[key][offset:end] = group[f"obs/{key}"][:]
            arrays["action"][offset:end] = group["actions"][:].astype(
                np.float32
            )
            offset = end
            print(
                f"momart {source.name}: demo {index}/{len(demos)} "
                f"steps={offset}/{n_steps}",
                flush=True,
            )
        root.attrs["demo_names"] = json.dumps(demos)
        return root, n_steps, len(demos)


def convert_robomimic(source, partial, config_path, workers, max_inflight):
    from cleandiffuser.dataset.robomimic_dataset import (
        _convert_robomimic_to_replay,
    )
    from cleandiffuser.dataset.dataset_utils import RotationTransformer

    config = OmegaConf.load(config_path)
    store = zarr.DirectoryStore(str(partial))
    replay = _convert_robomimic_to_replay(
        store=store,
        shape_meta=config.shape_meta,
        dataset_path=str(source),
        abs_action=bool(config.abs_action),
        rotation_transformer=RotationTransformer(
            from_rep="axis_angle", to_rep="rotation_6d"
        ),
        n_workers=workers,
        max_inflight_tasks=max_inflight,
    )
    root = zarr.open_group(store=store, mode="a")
    return root, replay.n_steps, replay.n_episodes


def main():
    args = parse_args()
    source, target, partial = prepare_paths(
        pathlib.Path(args.source), pathlib.Path(args.target), args.kind, args.force
    )
    if partial is None:
        return
    lock_path = target.with_name(target.name + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.kind == "momart":
            root, n_steps, n_episodes = convert_momart(
                source, partial, args.config
            )
        else:
            root, n_steps, n_episodes = convert_robomimic(
                source,
                partial,
                args.config,
                args.workers,
                args.max_inflight,
            )
        copy_env_metadata(source, root)
        root.attrs.update(
            {
                "vpwem_cache_complete": True,
                "vpwem_cache_format_version": FORMAT_VERSION,
                "vpwem_cache_kind": args.kind,
                "source_path": str(source),
                "source_size": source.stat().st_size,
                "source_mtime_ns": source.stat().st_mtime_ns,
                "n_steps": int(n_steps),
                "n_episodes": int(n_episodes),
            }
        )
        del root
        os.replace(partial, target)
        print(
            f"COMPLETE kind={args.kind} target={target} "
            f"steps={n_steps} episodes={n_episodes}",
            flush=True,
        )


if __name__ == "__main__":
    main()
