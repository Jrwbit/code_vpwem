"""Read online environment metadata without loading training observations."""

import json
from pathlib import Path

import zarr


def get_env_metadata(dataset_path):
    path = Path(dataset_path).expanduser()
    if path.suffix == ".zarr":
        root = zarr.open_group(str(path), mode="r")
        if root.attrs.get("vpwem_cache_complete") is not True:
            raise ValueError(f"incomplete Zarr cache: {path}")
        if "env_args" not in root.attrs:
            raise ValueError(f"Zarr cache lacks env_args; rerun the converter: {path}")
        return json.loads(root.attrs["env_args"])

    # Keep old evaluation commands and checkpoint configurations usable.
    import h5py
    with h5py.File(path, "r") as handle:
        return json.loads(handle["data"].attrs["env_args"])
