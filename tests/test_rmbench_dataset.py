import io
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from cleandiffuser.dataset.rmbench_dataset import (
    LongMemRMBenchImageDataset,
    convert_rmbench_to_replay_zarr,
)


def encode_png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return np.bytes_(buffer.getvalue())


class RMBenchDatasetTest(unittest.TestCase):
    def test_loads_rmbench_episode_and_returns_longmem_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            path = data_dir / "episode0.hdf5"
            rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
            with h5py.File(path, "w") as f:
                obs = f.create_group("observation").create_group("head_camera")
                obs.create_dataset("rgb", data=np.array([encode_png(rgb + i) for i in range(6)]))
                joint = f.create_group("joint_action")
                joint.create_dataset("vector", data=np.arange(6 * 14, dtype=np.float32).reshape(6, 14))

            shape_meta = {
                "action": {"shape": [14]},
                "obs": {
                    "head_cam": {
                        "shape": [3, 4, 5],
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

            dataset = LongMemRMBenchImageDataset(
                tmpdir,
                shape_meta=shape_meta,
                horizon=4,
                n_obs_steps=2,
                pad_before=1,
                pad_after=2,
                dataset_memory_type="random_from_full",
                subsample_ratio=2,
                maximum_memory_length=3,
            )

            sample = dataset[0]

            self.assertEqual(sample["action"].shape, torch.Size([4, 14]))
            self.assertEqual(sample["obs"]["head_cam"].shape, torch.Size([2, 3, 4, 5]))
            self.assertEqual(sample["obs"]["head_cam_mem"].shape[-3:], torch.Size([3, 4, 5]))
            self.assertEqual(sample["obs"]["agent_pos"].shape, torch.Size([2, 14]))
            self.assertIn("ep_step", sample["obs"])
            self.assertIn("actual_num_frames", sample["obs"])

    def test_loads_preprocessed_rmbench_zarr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "raw" / "data"
            data_dir.mkdir(parents=True)
            path = data_dir / "episode0.hdf5"
            rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
            with h5py.File(path, "w") as f:
                obs = f.create_group("observation").create_group("head_camera")
                obs.create_dataset("rgb", data=np.array([encode_png(rgb + i) for i in range(6)]))
                joint = f.create_group("joint_action")
                joint.create_dataset("vector", data=np.arange(6 * 14, dtype=np.float32).reshape(6, 14))

            shape_meta = {
                "action": {"shape": [14]},
                "obs": {
                    "head_cam": {
                        "shape": [3, 4, 5],
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

            zarr_path = Path(tmpdir) / "rmbench.zarr"
            convert_rmbench_to_replay_zarr(
                dataset_dir=Path(tmpdir) / "raw",
                output_path=zarr_path,
                shape_meta=shape_meta,
            )

            dataset = LongMemRMBenchImageDataset(
                zarr_path,
                shape_meta=shape_meta,
                horizon=4,
                n_obs_steps=2,
                pad_before=1,
                pad_after=2,
                dataset_memory_type="random_from_full",
                subsample_ratio=2,
                maximum_memory_length=3,
            )
            sample = dataset[0]

            self.assertEqual(sample["action"].shape, torch.Size([4, 14]))
            self.assertEqual(sample["obs"]["head_cam"].shape, torch.Size([2, 3, 4, 5]))
            self.assertEqual(sample["obs"]["head_cam_mem"].shape[-3:], torch.Size([3, 4, 5]))
            self.assertEqual(sample["obs"]["agent_pos"].shape, torch.Size([2, 14]))


    def test_shifted_replay_zarr_matches_dp_action_timing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "raw" / "data"
            data_dir.mkdir(parents=True)
            path = data_dir / "episode0.hdf5"
            rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
            qpos = np.arange(6 * 14, dtype=np.float32).reshape(6, 14)
            with h5py.File(path, "w") as f:
                obs = f.create_group("observation").create_group("head_camera")
                obs.create_dataset("rgb", data=np.array([encode_png(rgb + i) for i in range(6)]))
                joint = f.create_group("joint_action")
                joint.create_dataset("vector", data=qpos)

            shape_meta = {
                "action": {"shape": [14]},
                "obs": {
                    "head_cam": {
                        "shape": [3, 4, 5],
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

            zarr_path = Path(tmpdir) / "rmbench_dpaligned.zarr"
            convert_rmbench_to_replay_zarr(
                dataset_dir=Path(tmpdir) / "raw",
                output_path=zarr_path,
                shape_meta=shape_meta,
                shift_action=True,
            )

            dataset = LongMemRMBenchImageDataset(
                zarr_path,
                shape_meta=shape_meta,
                horizon=4,
                n_obs_steps=2,
                pad_before=1,
                pad_after=2,
                dataset_memory_type="random_from_full",
                subsample_ratio=2,
                maximum_memory_length=3,
            )
            replay = dataset.replay_buffer
            np.testing.assert_allclose(replay["agent_pos"][:], qpos[:-1])
            np.testing.assert_allclose(replay["action"][:], qpos[1:])
            np.testing.assert_allclose(replay["action"][:-1], replay["agent_pos"][1:])


if __name__ == "__main__":
    unittest.main()
