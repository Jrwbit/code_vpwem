"""Read new target sidecars without modifying any legacy red-bowl cache."""
import hashlib
from pathlib import Path
import numpy as np
import zarr
from momart_no_history_baseline.dataset import ZarrNoHistorySequenceDataset
from .inputs import replace_target_inputs

STANDARD = 'causal_selected_target_v1'


def source_signature(dataset, path):
    return dict(source=str(Path(path).resolve()),
        frames=len(dataset.replay_buffer['rgb']),
        episode_ends_sha256=hashlib.sha256(np.asarray(dataset.replay_buffer.episode_ends).tobytes()).hexdigest(),
        source_frame_map_sha256=hashlib.sha256(np.asarray(dataset.source_frame_map, dtype=np.int64).tobytes()).hexdigest(),
        demo_names=list(dataset._demo_id_to_start_indices))


class TargetNoHistoryDataset(ZarrNoHistorySequenceDataset):
    def __init__(self, zarr_path, obs_keys, seq_length, obs_steps, subsample_ratio,
                 bbox_path, mode, target_metadata):
        super().__init__(zarr_path, obs_keys, seq_length, obs_steps, subsample_ratio)
        if mode not in ('box', 'box_coordinates_roi'):
            raise ValueError('Unknown target input mode')
        if target_metadata.get('annotation_standard') != STANDARD:
            raise ValueError('Expected new causal target protocol')
        cache = zarr.open_group(str(bbox_path), mode='r')
        if (cache.attrs.get('complete') is not True
                or cache.attrs.get('completed_frames') != len(self)
                or cache.attrs.get('target_metadata') != target_metadata
                or cache.attrs.get('signature') != source_signature(self, zarr_path)):
            raise ValueError('Incomplete, stale or mismatched target cache')
        if Path(zarr_path).stem not in target_metadata['supported_sources']:
            raise ValueError('Source outside frozen selector vocabulary')
        for key in ('bundle_manifest_sha256', 'sensor_protocol_sha256'):
            value = target_metadata.get(key, '')
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Missing frozen protocol hash: ' + key)
        self.boxes = np.asarray(cache['boxes'], dtype=np.float32)
        self.valid = np.asarray(cache['valid'], dtype=bool)
        if self.boxes.shape != (len(self),4) or self.valid.shape != (len(self),):
            raise ValueError('Incorrect target array shapes')
        chosen = self.boxes[self.valid]
        if (not np.isfinite(chosen).all() or (chosen < 0).any() or (chosen > 1).any()
                or (chosen[:,2:] <= chosen[:,:2]).any()):
            raise ValueError('Invalid selected target boxes')
        self.mode = mode
        self.detector_metadata = dict(target_metadata, input_mode=mode)

    def _target_sample(self, sample, index):
        result = super()._prepare_sample(sample)
        start, end, _, _ = self.sampler.indices[index]
        frames = np.minimum(np.arange(start,start+self.obs_steps),end-1)
        batched = {k:v[None] for k,v in result['obs'].items()}
        result['obs'] = {k:v[0] for k,v in replace_target_inputs(
            batched, self.boxes[frames][None], self.valid[frames][None], self.mode).items()}
        return result

    def __getitem__(self, index):
        return self._target_sample(self.sampler.sample_sequence(index), index)

    def __getitems__(self, indices):
        from momart_no_history_baseline.batch_read import read_sequence_batch
        return [self._target_sample(sample,index) for sample,index in
                zip(read_sequence_batch(self,indices),indices)]
