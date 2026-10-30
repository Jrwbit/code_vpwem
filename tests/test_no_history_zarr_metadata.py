import json

import h5py
import numpy as np
import pytest
import zarr

from momart_no_history_baseline.metadata import get_env_metadata
from tools.convert_no_history_hdf5_to_zarr import (
    FORMAT_VERSION, copy_env_metadata, prepare_paths,
)


def test_existing_cache_metadata_upgrade_preserves_arrays(tmp_path, monkeypatch):
    source = tmp_path / "task.hdf5"
    target = tmp_path / "task.zarr"
    metadata = {"env_name": "test", "type": 3, "env_kwargs": {"seed": 100}}
    with h5py.File(source, "w") as handle:
        handle.create_group("data").attrs["env_args"] = json.dumps(metadata)
    root = zarr.open_group(str(target), mode="w")
    root.array("action", np.arange(12).reshape(3, 4))
    root.attrs.update({
        "vpwem_cache_complete": True,
        "vpwem_cache_format_version": FORMAT_VERSION,
        "vpwem_cache_kind": "momart",
        "source_size": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
    })
    assert prepare_paths(source, target, "momart", False)[2] is None
    assert get_env_metadata(source) == metadata
    np.testing.assert_array_equal(root["action"][:], np.arange(12).reshape(3, 4))

    def forbid_hdf5(*args, **kwargs):
        raise AssertionError("Zarr evaluation must not open HDF5")

    monkeypatch.setattr(h5py, "File", forbid_hdf5)
    assert get_env_metadata(target) == metadata


def test_fresh_cache_metadata_and_missing_metadata(tmp_path):
    source = tmp_path / "task.hdf5"
    target = tmp_path / "task.zarr"
    with h5py.File(source, "w") as handle:
        handle.create_group("data").attrs["env_args"] = '{"type": 3}'
    root = zarr.open_group(str(target), mode="w")
    with pytest.raises(ValueError, match="incomplete"):
        get_env_metadata(target)
    root.attrs["vpwem_cache_complete"] = True
    with pytest.raises(ValueError, match="lacks env_args"):
        get_env_metadata(target)
    copy_env_metadata(source, root)
    assert get_env_metadata(target) == {"type": 3}
