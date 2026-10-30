"""Training selection must exclude validation and preserve demo frame identity."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import zarr
from .cache import training_replay, Hdf5ImageArray, TrainingArray
from cleandiffuser.dataset.dataset_utils import SequenceSampler


class CacheTests(unittest.TestCase):
    def test_reordered_train_excludes_valid_and_pads_inside_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = zarr.open_group(str(Path(tmp)/'cache.zarr'), mode='w')
            root.attrs.update(vpwem_cache_complete=True, dataset_kind='momart',
                              source_demos=['demo_0','demo_1','demo_10'],
                              source_masks={'train':[2,0], 'valid':[1]})
            root.create_dataset('meta/episode_ends', data=np.array([2,5,7]))
            root.create_dataset('data/action', data=np.arange(7)[:,None])
            root.create_dataset('data/rgb', data=np.arange(7)[:,None,None,None])
            replay, names, mapping = training_replay(root)
            self.assertEqual(names, ['demo_10','demo_0'])
            np.testing.assert_array_equal(mapping, [5,6,0,1])
            np.testing.assert_array_equal(replay['action'][:].ravel(), [5,6,0,1])
            np.testing.assert_array_equal(replay['action'][1:3].ravel(), [6,0])
            np.testing.assert_array_equal(replay['action'][-1], [1])
            sampler = SequenceSampler(replay, 3, pad_after=2)
            np.testing.assert_array_equal(sampler.sample_sequence(1)['action'].ravel(), [6,6,6])
            np.testing.assert_array_equal(sampler.sample_sequence(2)['action'].ravel(), [0,1,1])
            root.attrs['source_masks'] = {'train':[2,1], 'valid':[1]}
            with self.assertRaisesRegex(ValueError, 'overlap'):
                training_replay(root)
            root.attrs['source_masks'] = {'valid':[1]}
            with self.assertRaisesRegex(ValueError, 'train mask'):
                training_replay(root)

    def test_original_hdf5_pixels_survive_reordering_and_pickle(self):
        import h5py
        import pickle
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'source.hdf5')
            pixels = np.arange(24, dtype=np.uint8).reshape(4,2,1,3)
            with h5py.File(path, 'w') as handle:
                handle.create_dataset('data/demo_0/obs/rgb', data=pixels[:2])
                handle.create_dataset('data/demo_10/obs/rgb', data=pixels[2:])
            raw = Hdf5ImageArray(path, 'rgb', ['demo_0','demo_10'],
                                 np.array([2,4]), pixels.shape, pixels.dtype)
            view = TrainingArray(raw, np.array([2,3,0,1]))
            np.testing.assert_array_equal(view[:], pixels[[2,3,0,1]])
            np.testing.assert_array_equal(view[1:3], pixels[[3,0]])
            restored = pickle.loads(pickle.dumps(view))
            np.testing.assert_array_equal(restored[-1], pixels[1])
            raw._handle.close()
            restored.array._handle.close()

    def test_original_train_cache_preserves_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = zarr.open_group(tmp, mode='w')
            root.attrs.update(vpwem_cache_complete=True, vpwem_cache_kind='momart',
                              demo_names='["demo_0","demo_10"]')
            root.create_dataset('meta/episode_ends', data=np.array([2,4]))
            root.create_dataset('data/action', data=np.arange(4)[:,None])
            replay, names, mapping = training_replay(root)
            np.testing.assert_array_equal(mapping, np.arange(4))
            np.testing.assert_array_equal(replay['action'][:], root['data/action'][:])
            self.assertEqual(names, ['demo_0','demo_10'])

if __name__ == '__main__':
    unittest.main()
