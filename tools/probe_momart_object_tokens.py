"""CPU-only, read-only interventions on the saved MoMaRT 150k object policy.

Run on BHR with the experiment environment; writes only diagnostic JSON.
These training-frame probes measure dependence, not task success or causality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from momart_boxed_tokens.dataset import BoxedTokenDataset
    from momart_boxed_tokens.train import initialize_modalities, split_batch
    from momart_boxed_tokens.evaluate import load_normalizers
    from momart_boxed_tokens.inputs import object_inputs
    from momart_boxed_tokens.model import build_agent

    torch.set_num_threads(2)
    torch.manual_seed(100)
    cfg = OmegaConf.load(args.work_dir / "train_config.yaml")
    cfg.device = "cpu"
    initialize_modalities(cfg)
    normalizers = load_normalizers(cfg, args.work_dir, "cpu")
    observations, targets, metadata, shifted_rois = [], [], [], []
    for spec in cfg.task_datasets:
        ds = BoxedTokenDataset(Path(cfg.data_root) / (spec.name + ".zarr"), cfg.shape_meta["obs"],
                              cfg.horizon, cfg.obs_steps, cfg.dataset_memory_subsample_ratio,
                              Path(cfg.bbox_root) / (spec.name + ".bbox.zarr"))
        ds.normalizer = normalizers
        # Pick the dataset sample nearest its midpoint with a valid newest frame.
        candidates = sorted(range(len(ds)), key=lambda i: abs(i - len(ds)//2))
        for index in candidates:
            start, end, _, _ = ds.sampler.indices[index]
            frame = min(int(start) + int(cfg.obs_steps) - 1, int(end) - 1)
            if ds.valid[frame]:
                break
        else:
            raise ValueError(f"No valid sample for {spec.name}")
        obs, action = split_batch(cfg, ds, spec.task_id,
                                  torch.utils.data.default_collate([ds[index]]))
        raw = np.asarray(ds.replay_buffer["rgb"][frame]).transpose(2, 0, 1).astype(np.float32)/255
        moved = obs["_bbox"][0].numpy().copy()
        moved[[0, 2]] += 1/120
        moved = np.clip(moved, 0, 1)
        _, roi, _ = object_inputs(raw, moved, True)
        shifted_rois.append(torch.from_numpy(roi)[None]*2-1)
        observations.append(obs)
        targets.append(action)
        metadata.append({"task": str(spec.name), "sample_index": index, "frame": frame,
                         "bbox": obs["_bbox"][0].tolist(), "shifted_bbox": moved.tolist()})
        print("Loaded " + str(spec.name), flush=True)
        del ds
    obs = {k: torch.cat([o[k] for o in observations]) for k in observations[0]}
    target = torch.cat(targets)
    agent = build_agent(cfg, normalizers)
    checkpoint = args.work_dir / "models/model_150000.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent.model_ema.load_state_dict(state["model_ema"], strict=True)
    del state
    agent.model_ema.eval()
    diffusion = agent.model_ema["diffusion"]
    original_mask = diffusion.memory_mask.clone()
    scenarios = {"normal": ((), False, False), "no_position": ((1,), False, False),
                 "no_roi": ((2,), False, False), "no_objects": ((1, 2), False, False),
                 "position_shift_1px": ((), True, False),
                 "roi_shift_1px": ((), False, True), "both_shift_1px": ((), True, True)}
    report = {"samples": metadata, "training_frames_not_heldout": True,
              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "diffusion_steps": agent.diffusion_steps, "results": {},
              "note": "Fixed scene pixels, fixed noises, EMA eval mode. ROI/position interventions isolate branches; not online success rates."}
    baseline_action = None
    with torch.no_grad():
        for name, (disabled, move_position, move_roi) in scenarios.items():
            started = time.time()
            diffusion.memory_mask.copy_(original_mask)
            for column in disabled:
                diffusion.memory_mask[:, column] = float("-inf")
            changed = {k: v.clone() for k, v in obs.items()}
            if move_position:
                changed["_bbox"] = torch.tensor([m["shifted_bbox"] for m in metadata])
            if move_roi:
                changed["_roi"] = torch.cat(shifted_rois)
            packed = agent.model_ema["condition"](changed)
            losses = []
            for timestep in (5, 25, 45):
                assert timestep < agent.diffusion_steps
                torch.manual_seed(20261029 + timestep)
                eps = torch.randn_like(target)
                t = torch.full((len(metadata),), timestep, dtype=torch.long)
                xt, _, _ = agent.add_noise(target, t=t, eps=eps)
                prediction = diffusion(xt, t, packed)
                losses.append((prediction[:, 1:]-eps[:, 1:]).square().mean((1, 2)))
            torch.manual_seed(20261029)
            agent.reset_cache()
            action, _ = agent.sample(prior=torch.zeros_like(target), n_samples=len(metadata),
                                     sample_steps=int(cfg.sample_steps), condition_cfg=changed,
                                     w_cfg=1., use_ema=True)
            assert torch.isfinite(action).all()
            if baseline_action is None:
                baseline_action = action.clone()
            delta = action[:, 1:] - baseline_action[:, 1:]
            row = {"noise_mse_mean": float(torch.stack(losses).mean()),
                   "noise_mse_per_sample": torch.stack(losses).mean(0).tolist(),
                   "normalized_action_rmse": float(delta.square().mean().sqrt()),
                   "normalized_action_rmse_per_sample": delta.square().mean((1, 2)).sqrt().tolist(),
                   "normalized_action_max_abs_delta": float(delta.abs().max()),
                   "seconds": time.time()-started}
            report["results"][name] = row
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
            print(name + ": " + json.dumps(row), flush=True)
        diffusion.memory_mask.copy_(original_mask)


if __name__ == "__main__":
    main()
