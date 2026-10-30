"""Generate causal anchor snapshots using exactly the deployment update stream."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import zarr
from omegaconf import OmegaConf
from momart_no_history_baseline.cache import training_replay
from .state import make_bank, ObservationStream
from .association import association_contract
from .perception import FrozenPerception, file_sha256, validate_perception_metadata


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
                reset='sensor_step_once', overflow=str(m.get('overflow','error')),
                association=association_contract(m.get('association', None)),
                implementation_sha256={name:file_sha256(Path(__file__).with_name(name))
                                       for name in ('cache.py','state.py','association.py')})


def write_cache(config, source, target, perceive, observation_reader=None, resume=False):
    if int(config.obs_steps) != 2:
        raise ValueError('Only two-observation PTP supported')
    validate_perception_metadata(config.object_memory, perceive.metadata)
    target = Path(target)
    if target.exists() and not resume:
        raise FileExistsError('Refusing to overwrite object cache: '+str(target))
    source = Path(source)
    root = zarr.open_group(str(source), mode='r')
    replay, names, frame_map = training_replay(root)
    rgb = replay['rgb']; n = len(rgb); m = config.object_memory
    signature = replay_signature(replay, frame_map, names)
    signature.update(source=str(source.resolve()), perception=perceive.metadata,
                     stream=stream_contract(config))
    if target.exists():
        out = zarr.open_group(str(target), mode='a')
        if out.attrs.get('signature') != signature:
            raise ValueError('Resume cache signature differs')
        completed = int(out.attrs.get('completed_frames', 0))
        if out.attrs.get('complete') is True:
            if completed != n: raise ValueError('Invalid completed object cache')
            return
        if completed not in [0] + list(map(int, replay.episode_ends)):
            raise ValueError('Resume point is not an episode boundary')
    else:
        out = zarr.open_group(str(target), mode='w')
        out.attrs.update(complete=False, completed_frames=0, signature=signature)
        completed = 0
    bank = make_bank(config)
    example = bank.snapshot('a3')
    arrays = {key: out.require_dataset(key, shape=(n,)+v.shape, dtype=v.dtype,
                                       chunks=(min(128,n),)+v.shape) for key, v in example.items()}
    bowl_boxes = out.require_dataset('bowl_boxes', shape=(n, 4), dtype='f4', chunks=(min(256,n), 4))
    bowl_valid = out.require_dataset('bowl_valid', shape=(n,), dtype='bool', chunks=(min(256,n),))
    start = 0
    for end in replay.episode_ends:
        end = int(end); length = end-start
        if end <= completed:
            start = end
            continue
        episode = {key: np.zeros((length,)+v.shape, v.dtype) for key,v in example.items()}
        episode_boxes = np.zeros((length,4), np.float32)
        episode_valid = np.zeros(length,bool)
        # Perception is stateless and frozen: safe to reuse outputs across phases.
        # Only this episode is resident, and only cadence-visible frames enter bank.
        observations = []
        for index in range(start, end):
            seen = observation_reader(index) if observation_reader is not None else perceive(np.asarray(rgb[index]))
            observations.append(seen)
            bowls = [o for o in seen if o.class_id == 0]
            if bowls:
                episode_boxes[index-start] = max(bowls, key=lambda o: o.score).box
                episode_valid[index-start] = True
        for phase in range(int(config.action_steps)):
            stream = ObservationStream(make_bank(config),
                                       lambda frame: observations[int(frame)])
            stream.observe([0, 0], [0, 0], [0., 0.])
            queries = ([0] if phase == 0 else []) + list(range(phase or int(config.action_steps),
                                                                 length, int(config.action_steps)))
            for query in queries:
                frames = [max(0, query-1), query]
                stream.observe(frames, frames, [f*float(m.period_seconds) for f in frames])
                snapshot = stream.bank.snapshot('a3')
                for key, value in snapshot.items():
                    episode[key][query] = value
        for key,value in episode.items():
            arrays[key][start:end] = value
        bowl_boxes[start:end] = episode_boxes
        bowl_valid[start:end] = episode_valid
        out.attrs["completed_frames"] = end
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
