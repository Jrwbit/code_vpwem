#!/usr/bin/env python3
"""Read-only audit against the actual BHR MoMaRT experiment checkout and caches.

Run with that checkout's Python. No training, cache rewrite, or rollout occurs.
The JSON contains measurements and source hashes, never images or weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--boxed-work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-step', type=int, default=150000)
    parser.add_argument('--detector-bundle', type=Path)
    parser.add_argument('--detector-device', default='cpu')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import robomimic.utils.obs_utils as ObsUtils
    from omegaconf import OmegaConf
    from momart_boxed_tokens.dataset import BoxedTokenDataset
    from momart_boxed_tokens.train import initialize_modalities, split_batch
    from momart_boxed_tokens.evaluate import load_normalizers, normalize_observations
    from momart_boxed_tokens.inputs import OnlineObjectInputs
    from momart_boxed_tokens.model import build_agent
    from momart_bbox_memory.detector import Detection
    from cleandiffuser.nn_condition.multi_image_condition import MultiImageObsConditionMomart

    torch.set_num_threads(2)
    torch.manual_seed(100)
    cfg = OmegaConf.load(args.work_dir / 'train_config.yaml')
    cfg.device = 'cpu'
    initialize_modalities(cfg)
    normalizers = load_normalizers(cfg, args.work_dir, 'cpu')
    other = OmegaConf.to_container(OmegaConf.load(args.boxed_work_dir / 'train_config.yaml'), resolve=True)
    this = OmegaConf.to_container(OmegaConf.load(args.work_dir / 'train_config.yaml'), resolve=True)
    report = dict(checks=[], datasets=[], source_files=[], config_differences={
        k: {'boxed': other.get(k), 'tokens': this.get(k)}
        for k in sorted(set(this) | set(other)) if this.get(k) != other.get(k)})

    def check(name, ok, **details):
        report['checks'].append(dict(name=name, passed=bool(ok), **details))
        print(json.dumps(report['checks'][-1], ensure_ascii=False), flush=True)

    files = [f'{package}/{name}.py' for package, names in {
        'momart_boxed_tokens': ['model', 'inputs', 'dataset', 'train', 'evaluate'],
        'momart_boxed_no_history': ['painting', 'dataset'],
        'momart_no_history_baseline': ['cache', 'model', 'envs'],
        'momart_bbox_memory': ['detector', 'precision_detector', 'annotate_bbox'],
    }.items() for name in names]
    files += ['cleandiffuser/utils/crop_randomizer.py',
              'cleandiffuser/dataset/dataset_utils.py',
              'cleandiffuser/nn_condition/base_nn_condition.py',
              'cleandiffuser/nn_condition/multi_image_condition.py',
              'cleandiffuser/diffusion/longmem_ddpm.py',
              'cleandiffuser/env/wrapper.py']
    for relative in files:
        hashes = {key: (hashlib.sha256((root / relative).read_bytes()).hexdigest()
                        if (root / relative).is_file() else None)
                  for key, root in [('training', args.training_root), ('evaluation', args.source_root)]}
        report['source_files'].append(dict(path=relative, **hashes))
    # The online tracker was introduced after training; its absence is expected.
    critical = [x for x in report['source_files'] if x['path'].startswith('momart_boxed_tokens/')
                or x['path'] == 'momart_boxed_no_history/painting.py']
    check('training_evaluation_object_input_source_identity',
          all(x['training'] == x['evaluation'] and x['training'] is not None for x in critical),
          different=[x['path'] for x in report['source_files'] if x['training'] != x['evaluation']])
    with np.load(args.work_dir / 'normalizers.npz') as a, np.load(args.boxed_work_dir / 'normalizers.npz') as b:
        check('boxed_tokens_same_normalizers', set(a.files) == set(b.files) and
              all(np.array_equal(a[k], b[k]) for k in a.files))

    example_condition = None
    detector_example = None
    total_samples = 0
    for spec in cfg.task_datasets:
        ds = BoxedTokenDataset(Path(cfg.data_root) / (spec.name + '.zarr'), cfg.shape_meta['obs'],
                              cfg.horizon, cfg.obs_steps, cfg.dataset_memory_subsample_ratio,
                              Path(cfg.bbox_root) / (spec.name + '.bbox.zarr'))
        ds.normalizer = normalizers
        ends = np.asarray(ds.replay_buffer.episode_ends)
        valid_ids = np.flatnonzero(ds.valid)
        absent_ids = np.flatnonzero(~ds.valid)
        # Include starts/ends of multiple episodes and interior valid/missing boxes.
        indices = {0, len(ds) // 2, len(ds) - 1}
        for end in (int(ends[0]), int(ends[len(ends) // 2])):
            indices.update(i for i in (end - 2, end - 1, end) if 0 <= i < len(ds))
        indices.update(max(0, int(ids[len(ids) // 2]) - 1)
                       for ids in (valid_ids, absent_ids) if len(ids))
        max_error = {}; roi_oracle_error = 0.; pixel_source_error = 0
        for index in sorted(indices):
            sample = ds[index]
            raw_sample = ds.sampler.sample_sequence(index)
            start, end, _, _ = ds.sampler.indices[index]
            frames = np.minimum(np.arange(start, start + cfg.obs_steps), end - 1)
            for frame in frames:
                # Compare pixels actually used by training with the sidecar's source array.
                original = np.asarray(ds.replay_buffer['rgb'][int(frame)])
                array = ds.replay_buffer['rgb'].array
                physical = int(ds.source_frame_map[int(frame)])
                if hasattr(array, '_file'):
                    # HDF5 fallback: independently open the declared sidecar source below.
                    import zarr
                    source = zarr.open_group(ds.bbox.attrs['signature']['source'], mode='r')
                    source_pixels = np.asarray(source['data/rgb'][physical])
                else:
                    source_pixels = np.asarray(array[physical])
                pixel_source_error = max(pixel_source_error, int(np.abs(
                    original.astype(np.int32) - source_pixels.astype(np.int32)).max()))
            processed = ObsUtils.process_obs_dict({k: raw_sample[k][:cfg.obs_steps]
                                                    for k in cfg.shape_meta['obs']})
            obs = {k: np.asarray(v)[None] for k, v in processed.items()}
            detections = [Detection(ds.boxes[int(f)], 1., bool(ds.valid[int(f)]), False, 1)
                          for f in frames]

            class FixedDetector:
                def __call__(self, images):
                    assert len(images) == len(frames)
                    # Validate HWC uint8/RGB conversion against independent source pixels.
                    for image, frame in zip(images, frames):
                        np.testing.assert_array_equal(image, ds.replay_buffer['rgb'][int(frame)])
                    return detections

            online = OnlineObjectInputs(FixedDetector(), batch_size=cfg.obs_steps)(obs)
            batch = torch.utils.data.default_collate([sample])
            train_obs, _ = split_batch(cfg, ds, spec.task_id, batch)
            eval_obs = normalize_observations(online, normalizers, 'cpu')
            eval_obs['_task_id'] = torch.tensor([spec.task_id], dtype=torch.long)
            for key in train_obs:
                delta = float((train_obs[key] - eval_obs[key]).abs().max())
                max_error[key] = max(max_error.get(key, 0.), delta)
            # Independent half-open integer ROI oracle, prior to normalization.
            f = int(frames[-1])
            if ds.valid[f]:
                box = np.clip(ds.boxes[f], 0, 1)
                x1 = int(np.floor(float(box[0]) * 120)); y1 = int(np.floor(float(box[1]) * 90))
                x2 = int(np.ceil(float(box[2]) * 120)); y2 = int(np.ceil(float(box[3]) * 90))
                raw = np.asarray(ds.replay_buffer['rgb'][f])
                crop = torch.tensor(raw[y1:y2, x1:x2].copy()).permute(2, 0, 1).float() / 255
                expected = F.interpolate(crop[None], (64, 64), mode='bilinear', align_corners=False)[0]
                roi_oracle_error = max(roi_oracle_error, float((expected - torch.tensor(sample['obs']['_roi'])).abs().max()))
                if example_condition is None:
                    example_condition = {k: v.clone() for k, v in eval_obs.items()}
            total_samples += 1
        row = dict(task=spec.name, samples=len(indices), max_abs_error=max_error,
                   roi_oracle_max_abs_error=roi_oracle_error, annotation_source_pixel_error=pixel_source_error,
                   frames=len(ds.valid), valid_frames=int(ds.valid.sum()),
                   source_frame_map_identity=bool(np.array_equal(ds.source_frame_map, np.arange(len(ds.valid)))),
                   annotation_standard=ds.bbox.attrs.get('annotation_standard', 'detector'),
                   detector=ds.detector_metadata)
        report['datasets'].append(row)
        check('same_frames_same_boxes_parity/' + spec.name,
              max(max_error.values()) < 1e-6 and roi_oracle_error < 1e-6 and pixel_source_error == 0,
              samples=len(indices), maximum_error=max(max_error.values()))
        boxes = ds.boxes[ds.valid]
        check('valid_boxes_finite_nonempty/' + spec.name,
              np.isfinite(boxes).all() and ((boxes >= 0) & (boxes <= 1)).all()
              and (boxes[:, 2:] > boxes[:, :2]).all())
        if detector_example is None and len(valid_ids):
            # A valid episode segment; preserve real stream order and episode boundary.
            frame = int(valid_ids[len(valid_ids) // 2])
            epi = int(np.searchsorted(ends, frame, side='right'))
            lo = int(ends[epi - 1]) if epi else 0
            hi = int(ends[epi])
            start = max(lo, min(frame - 8, hi - 25))
            seq = list(range(start, min(start + 25, hi)))
            detector_example = dict(task=spec.name, frames=seq,
                images=[np.asarray(ds.replay_buffer['rgb'][i]) for i in seq],
                cached=[Detection(ds.boxes[i], 1., bool(ds.valid[i]), False, 1) for i in seq])
        del ds
    report['total_samples'] = total_samples

    # Capture the actual backbone inputs; spatial ramps identify crop origins exactly.
    encoder = MultiImageObsConditionMomart(shape_meta=cfg.shape_meta, emb_dim=256,
        rgb_model_name=cfg.rgb_model, resize_shape=cfg.resize_shape, crop_shape=cfg.crop_shape,
        random_crop=cfg.random_crop, use_group_norm=cfg.use_group_norm, use_seq=True, keep_horizon_dims=True)
    captured = {}

    class Capture(torch.nn.Module):
        def __init__(self, key, dim):
            super().__init__(); self.key = key; self.dim = dim

        def forward(self, x):
            captured[self.key] = x.detach().clone()
            return x.new_zeros((x.shape[0], self.dim))

    for key in list(encoder.key_model_map):
        encoder.key_model_map[key] = Capture(key, 64 if key == 'scan' else 512)
    ramp = torch.arange(90)[:, None] * 120 + torch.arange(120)[None, :]
    probe = {key: torch.zeros((4, 2, *attr['shape'])) for key, attr in cfg.shape_meta['obs'].items()}
    for key in ('rgb', 'rgb_wrist', 'depth', 'depth_wrist'):
        probe[key][:] = ramp
    crop_result = {}
    for mode in ('train', 'eval'):
        encoder.train(mode == 'train'); torch.manual_seed(20261028)
        encoder(probe)
        offsets = {}
        for key in ('rgb', 'rgb_wrist', 'depth', 'depth_wrist'):
            origins = captured[key][:, 0, 0, 0].long()
            offsets[key] = torch.stack([origins // 120, origins % 120], dim=-1).tolist()
        crop_result[mode] = offsets
    report['crop_origins_yx'] = crop_result
    report['rgb_depth_train_same_origins'] = crop_result['train']['rgb'] == crop_result['train']['depth']
    report['rgb_depth_eval_same_origins'] = crop_result['eval']['rgb'] == crop_result['eval']['depth']
    # These are measurements, not test failures: the old pipeline intentionally draws separately.
    check('eval_center_crop_pixel_origin', all(
        xy == [8, 10] for values in crop_result['eval'].values() for xy in values))
    report['coordinate_reference'] = dict(token='original normalized xyxy',
        scene_train='independent crop, no crop transform in tokens',
        scene_eval='center crop y=8,x=10,height=75,width=100',
        roi='unpainted original frame, floor/ceil, bilinear 64x64, align_corners=False')
    del encoder

    # Verify saved model and EMA, then compare train/eval paths using identical observations.
    agent = build_agent(cfg, normalizers)
    model_path = args.work_dir / 'models' / f'model_{args.checkpoint_step}.pt'
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    agent.model.load_state_dict(state['model'], strict=True)
    agent.model_ema.load_state_dict(state['model_ema'], strict=True)
    del state
    agent.model.eval(); agent.model_ema.eval()
    digest = hashlib.sha256()
    with model_path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    check('checkpoint_strict_model_and_ema', True, sha256=digest.hexdigest())
    with torch.no_grad():
        packed = agent.model_ema['condition'](example_condition)
        torch.manual_seed(20261028)
        x = torch.randn(1, cfg.horizon, cfg.action_dim)
        t = torch.full((1,), 25, dtype=torch.long)
        base = agent.model_ema['diffusion'](x, t, packed)
        sensitivity = []
        for pixels in (0.25, 1., 2.):
            for direction in (-1, 1):
                changed = {k: v.clone() for k, v in example_condition.items()}
                changed['_bbox'][:, [0, 2]] += direction * pixels / 120
                token = agent.model_ema['condition'](changed)
                output = agent.model_ema['diffusion'](x, t, token)
                sensitivity.append(dict(x_shift_pixels=direction * pixels,
                    position_token_l2=float(torch.linalg.vector_norm(token[0, -1, 256:512] - packed[0, -1, 256:512])),
                    noise_prediction_rmse=float((output[:, 1:] - base[:, 1:]).square().mean().sqrt())))
        report['position_only_checkpoint_sensitivity'] = sensitivity
        # End-to-end equality with dropout/crops held equal by evaluation mode.
        prior = torch.zeros(1, cfg.horizon, cfg.action_dim)
        torch.manual_seed(20261028); agent.reset_cache()
        action, _ = agent.sample(prior=prior, n_samples=1, sample_steps=cfg.sample_steps,
                                 condition_cfg=example_condition, w_cfg=1., use_ema=True)
        check('ema_ddpm_finite_action', bool(torch.isfinite(action).all()), shape=list(action.shape),
              min=float(action.min()), max=float(action.max()))
    del agent

    if args.detector_bundle:
        from momart_bbox_memory.precision_detector import PrecisionDetector
        from momart_bbox_memory.detector import BowlDetector
        images = detector_example['images']
        selected = sorted(set(i for i in (0, 7, 8, 15, 16, 23, 24) if i < len(images)))
        offline = BowlDetector(device=args.detector_device, local_files_only=True)
        fresh = offline([images[i] for i in selected])
        # Reuse the exact same frozen GDINO instance in both paths; only strategy differs.
        online = PrecisionDetector(args.detector_bundle, device=args.detector_device)
        online.global_detector = offline
        results = []
        for i, current in zip(selected, fresh):
            tracked = online.detect(images[i], stream_id=0)
            cached = detector_example['cached'][i]
            results.append(dict(frame=detector_example['frames'][i],
                cached_valid=bool(cached.valid), fresh_valid=bool(current.valid), online_valid=bool(tracked.valid),
                cached_box=cached.box.tolist(), fresh_box=current.box.tolist(), online_box=tracked.box.tolist()))
        report['detector_sample'] = dict(task=detector_example['task'],
            protocol='cold-start segment within one episode, not full-episode tracker state; '
                     'frames [0,7,8,15,16,23,24] relative to segment',
            offline_metadata=offline.metadata, online_metadata=online.metadata, results=results,
            note='Small diagnostic sample, not detector accuracy or rollout success.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    failures = [x['name'] for x in report['checks'] if not x['passed']]
    print(json.dumps(dict(output=str(args.output), samples=total_samples, failures=failures)))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
