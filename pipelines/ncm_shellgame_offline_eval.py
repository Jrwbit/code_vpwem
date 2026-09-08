import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch

from cleandiffuser.dataset.dataset_utils import dynamic_batch_collate_fn
from cleandiffuser.dataset.mikasa_dataset_longmem import LongMemMikasaDataset
from cleandiffuser.dataset.ncm_pairing import (
    NCMPairedBatchSampler,
    partner_positions,
)
from ncm_shellgame_eval import build_agent, file_sha256
from utils import set_seed


def prepare_batch(args, dataset, batch):
    condition, memory = {}, {}
    observations = batch["obs"]
    for key, value in observations.items():
        if key.endswith("_mem"):
            base_key = key[:-4]
            value = value.to(args.device)
            memory[base_key] = dataset.normalizer["obs"][base_key].normalize(
                value)
        elif key not in {"ep_step", "actual_num_frames"}:
            value = value[:, :args.obs_steps].to(args.device)
            condition[key] = dataset.normalizer["obs"][key].normalize(value)
    action = dataset.normalizer["action"].normalize(
        batch["action"].to(args.device))
    steps = observations["ep_step"].int().to(args.device)
    ncm = batch["ncm"]
    weight = ncm["alias_weight"].to(args.device)
    partner = partner_positions(
        ncm["sample_index"].to(args.device),
        ncm["matched_partner"].to(args.device),
        valid=weight > 0)
    return action, condition, memory, steps, partner, weight


@hydra.main(
    config_path="../configs/dp/mikasa/chi_transformer_ptp_longmem",
    config_name="shell_game_touch_ncm")
def pipeline(args):
    set_seed(args.seed)
    if str(args.counterfactual_negative_mode) != "matched":
        raise ValueError("Offline NCM gap evaluation requires mode=matched")
    if float(args.counterfactual_memory_coeff) <= 0:
        raise ValueError("counterfactual_memory_coeff must be positive")

    dataset = LongMemMikasaDataset(
        dataset_dir=os.path.expanduser(args.dataset_path),
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        horizon=args.horizon,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        dataset_memory_type=args.dataset_memory_type,
        subsample_ratio=args.dataset_memory_subsample_ratio,
        maximum_memory_length=int(getattr(
            args, "dataset_maximum_memory_length", 16)),
        preserve_episode_anchor=bool(getattr(
            args, "dataset_preserve_episode_anchor", False)),
        ncm_pair_manifest_path=args.ncm_pair_manifest_path,
        ncm_pairing_seed=int(args.ncm_pairing_seed),
        conversion_n_workers=int(args.conversion_n_workers),
        conversion_max_inflight_tasks=int(args.conversion_max_inflight_tasks),
        device=args.device)
    if args.normalizer_dataset_path:
        normalizer_dataset = LongMemMikasaDataset(
            dataset_dir=os.path.expanduser(str(args.normalizer_dataset_path)),
            shape_meta=args.shape_meta,
            n_obs_steps=args.obs_steps,
            horizon=args.horizon,
            pad_before=args.obs_steps - 1,
            pad_after=args.action_steps - 1,
            dataset_memory_type=args.dataset_memory_type,
            subsample_ratio=args.dataset_memory_subsample_ratio,
            maximum_memory_length=int(getattr(
                args, "dataset_maximum_memory_length", 16)),
            conversion_n_workers=int(args.conversion_n_workers),
            conversion_max_inflight_tasks=int(
                args.conversion_max_inflight_tasks),
            load_images=False,
            device=args.device)
        dataset.normalizer = normalizer_dataset.normalizer
    sampler = NCMPairedBatchSampler(
        dataset.ncm_manifest["batches"], shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=dynamic_batch_collate_fn)
    agent = build_agent(args, dataset.normalizer)
    model_path = os.path.expanduser(str(args.model_path))
    agent.load(model_path)
    agent.model.eval()
    agent.model_ema.eval()
    agent.nn_condition_drop.eval()
    agent.nn_memory_drop.eval()

    metric_values = {
        "correct_action_loss": [],
        "wrong_action_loss": [],
        "delta_cf": [],
        "r_cf": [],
        "margin_violation": [],
    }
    mc_samples = int(args.offline_mc_samples)
    max_batches = int(args.offline_max_batches)
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            if max_batches > 0 and batch_id >= max_batches:
                break
            prepared = prepare_batch(args, dataset, batch)
            action, condition, memory, steps, partner, weight = prepared
            for draw in range(mc_samples):
                # Every checkpoint sees the same add_noise t/epsilon table.
                seed = int(args.offline_seed) + batch_id * mc_samples + draw
                set_seed(seed)
                _, _, _, correct = agent.loss(
                    action, condition, memory, steps,
                    counterfactual_index=partner,
                    counterfactual_weight=weight,
                    use_ema=True)
                losses = agent._last_memory_losses
                metric_values["correct_action_loss"].append(float(correct))
                metric_values["wrong_action_loss"].append(float(
                    losses["wrong_memory_action_loss"]))
                metric_values["delta_cf"].append(float(
                    losses["counterfactual_gap"]))
                metric_values["r_cf"].append(float(
                    losses["counterfactual_ratio"]))
                metric_values["margin_violation"].append(float(
                    losses["counterfactual_margin_violation"]))

    result = {
        "checkpoint": model_path,
        "checkpoint_sha256": file_sha256(model_path),
        "manifest_sha256": str(dataset.ncm_manifest["sha256"].item()),
        "offline_seed": int(args.offline_seed),
        "mc_samples": mc_samples,
        "batches": len(metric_values["delta_cf"]) // mc_samples,
    }
    for key, values in metric_values.items():
        result[key] = float(np.mean(values))
        result[f"{key}_se"] = float(
            np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    output = Path(args.offline_result_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    pipeline()
