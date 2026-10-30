"""Generate causal anchor snapshots using exactly the deployment update stream."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import zarr
from omegaconf import OmegaConf
from momart_no_history_baseline.cache import training_replay
from .state import ObjectBank, ObservationStream
from .perception import FrozenPerception, file_sha256


def replay_signature(replay, frame_map, names):
    return dict(episode_ends_sha256=hashlib.sha256(np.asarray(replay.episode_ends).tobytes()).hexdigest(),
                source_frame_map_sha256=hashlib.sha256(np.asarray(frame_map).tobytes()).hexdigest(),
                demo_names=list(names), frames=len(replay['rgb']), rgb_shape=list(replay['rgb'].shape))


def stream_contract(config):
    m = config.object_memory
    return dict(version='causal_phase_prefix_v1', obs_steps=int(config.obs_steps),
                action_steps=int(config.action_steps), period_seconds=float(m.period_seconds),
                slots=int(m.slots), history=int(m.history), feature_dim=int(m.feature_dim),
                match_iou=float(m.match_iou), training_phases='all', deployment_phase=0,
                reset='sensor_step_once', overflow='error',
                implementation_sha256={name:file_sha256(Path(__file__).with_name(name))
                                       for name in ('cache.py','state.py')})


def write_cache(config, source, target, perceive):
    if int(config.obs_steps) != 2:
        raise ValueError('Only two-observation PTP supported')
    target = Path(target)
    if target.exists():
        raise FileExistsError('Refusing to overwrite object cache: '+str(target))
    source = Path(source)
    root = zarr.open_group(str(source), mode='r')
    replay, names, frame_map = training_replay(root)
    rgb = replay['rgb']; n = len(rgb); m = config.object_memory
    out = zarr.open_group(str(target), mode='w')
    signature = replay_signature(replay, frame_map, names)
    signature.update(source=str(source.resolve()), perception=perceive.metadata,
                     stream=stream_contract(config))
    out.attrs.update(complete=False, signature=signature)
    bank = ObjectBank(int(m.slots), int(m.history), int(m.feature_dim), float(m.match_iou))
    example = bank.snapshot('a3')
    arrays = {key: out.create_dataset(key, shape=(n,)+v.shape, dtype=v.dtype,
                                      chunks=(1,)+v.shape) for key, v in example.items()}
    bowl_boxes = out.create_dataset('bowl_boxes', shape=(n, 4), dtype='f4', chunks=(256, 4))
    bowl_valid = out.create_dataset('bowl_valid', shape=(n,), dtype='bool', chunks=(256,))
    start = 0
    for end in replay.episode_ends:
        end = int(end); length = end-start
        # Perception is stateless and frozen: safe to reuse outputs across phases.
        # Only this episode is resident, and only cadence-visible frames enter bank.
        observations = []
        for index in range(start, end):
            seen = perceive(np.asarray(rgb[index]))
            observations.append(seen)
            bowls = [o for o in seen if o.class_id == 0]
            if bowls:
                bowl_boxes[index] = max(bowls, key=lambda o: o.score).box
                bowl_valid[index] = True
        for phase in range(int(config.action_steps)):
            stream = ObservationStream(ObjectBank(int(m.slots), int(m.history),
                                                 int(m.feature_dim), float(m.match_iou)),
                                       lambda frame: observations[int(frame)])
            stream.observe([0, 0], [0, 0], [0., 0.])
            queries = ([0] if phase == 0 else []) + list(range(phase or int(config.action_steps),
                                                                 length, int(config.action_steps)))
            for query in queries:
                frames = [max(0, query-1), query]
                stream.observe(frames, frames, [f*float(m.period_seconds) for f in frames])
                snapshot = stream.bank.snapshot('a3')
                for key, value in snapshot.items():
                    arrays[key][start+query] = value
        start = end
        print(json.dumps(dict(source=source.name, completed_frames=end, total=n)), flush=True)
    out.attrs.update(complete=True, completed_frames=n)


def select_mode(snapshot, mode):
    data = {k: np.array(v, copy=True) for k, v in snapshot.items()}
    if mode not in ('a1', 'a2', 'a3', 'a3_latest_control'):
        raise ValueError('Unknown memory mode')
    if mode != 'a3':
        for key in ('features', 'ages'):
            data[key][:, :-1] = 0
        data['event_valid'][:, :-1] = False
    if mode == 'a1':
        data['valid'] &= data['visible']
        for key in data:
            if key != 'valid':
                data[key][~data['valid']] = 0
    if mode == 'a3_latest_control':
        for key in ('features', 'ages', 'event_valid'):
            data[key][:] = data[key][:, -1:]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    perceive = FrozenPerception(cfg.object_memory, args.device)
    for spec in cfg.task_datasets:
        source = Path(cfg.data_root)/(spec.name+'.zarr')
        target = Path(cfg.object_memory.cache_root)/(spec.name+'.objects.zarr')
        target.parent.mkdir(parents=True, exist_ok=True)
        write_cache(cfg, source, target, perceive)


if __name__ == '__main__':
    main()
