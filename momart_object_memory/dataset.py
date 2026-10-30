"""Same baseline labels/modalities with shared perception and causal snapshots."""
from pathlib import Path
import numpy as np
import zarr
from .cache import replay_signature, stream_contract, select_mode
from .model import OBJECT_KEYS


class ObjectDataset:
    def __init__(self, base, args, name):
        self.base = base
        self.mode = args.object_memory.mode
        self.root = zarr.open_group(str(Path(args.object_memory.cache_root)/(name+'.objects.zarr')), mode='r')
        expected = replay_signature(base.replay_buffer, base.source_frame_map,
                                    list(base._demo_id_to_start_indices))
        signature = self.root.attrs.get('signature', {})
        if (self.root.attrs.get('complete') is not True
                or self.root.attrs.get('completed_frames') != expected['frames']
                or any(signature.get(k) != v for k, v in expected.items())
                or signature.get('source') != str((Path(args.data_root)/(name+'.zarr')).resolve())
                or signature.get('stream') != stream_contract(args)
                or signature.get('perception', {}).get('classes') != list(args.object_memory.classes)):
            raise ValueError('Incomplete or incompatible object cache: '+name)
        if signature['perception']['feature_dim'] != int(args.object_memory.feature_dim):
            raise ValueError('Object feature dimension mismatch')
        self.perception_metadata = signature['perception']
        # The archived baseline's main-image detector is independent of object ROI perception.
        self.detector_metadata = base.detector_metadata
        self.normalizer = base.normalizer

    def __len__(self):
        return len(self.base)

    def get_dataset_sampler(self):
        return self.base.get_dataset_sampler()

    def __getitem__(self, index):
        sample = self.base[index]
        start, end, _, _ = self.base.sampler.indices[index]
        frames = np.minimum(np.arange(start, start+self.base.obs_steps), end-1)
        anchor = int(frames[-1])
        # Keep the baseline's already-painted observations byte-for-byte.
        snapshot = select_mode({key: self.root[key][anchor] for key in OBJECT_KEYS}, self.mode)
        sample['objects'] = snapshot
        return sample
