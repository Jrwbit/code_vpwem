#!/usr/bin/env python3
"""Build leave-one-out action-influence labels and fit a retention head.

This is a development-screen teacher, not an online oracle policy. It uses
demonstration actions and a frozen VPWEM checkpoint to measure how much removing
each history slot raises denoising loss.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cleandiffuser.dataset.dataset_utils import dynamic_batch_collate_fn
from cleandiffuser.dataset.robomimic_dataset_longmem import LongMemRobomimicImageDataset
from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP
from cleandiffuser.nn_memory.learned_retention_memory import RetentionHead, retention_features
from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
from cleandiffuser.utils import cosine_beta_schedule


def prefixed_state(state, prefix):
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def per_sample_loss(prediction, target):
    return (prediction - target).pow(2).flatten(1).mean(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    shape_meta = {
        "action": {"shape": [20]},
        "obs": {"dp_emb": {"shape": [256], "type": "low_dim"}},
    }
    dataset = LongMemRobomimicImageDataset(
        args.dataset, shape_meta=shape_meta, n_obs_steps=2, horizon=9,
        pad_before=1, pad_after=7, abs_action=True,
        dataset_memory_type="random_from_full", subsample_ratio=20,
        maximum_memory_length=16, device=device)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        drop_last=True, pin_memory=True, persistent_workers=True,
        collate_fn=dynamic_batch_collate_fn)

    diffusion = LongMemChiTransformerPTP(
        20, 256, 9, 2, 2, d_model=256, nhead=4, num_layers=4,
        p_drop_emb=0.0, p_drop_attn=0.3,
        timestep_emb_type="positional").to(device).eval()
    # The teacher sees all sampled history (up to 16) before producing the
    # same two policy memory tokens.
    teacher = QFormerMemory(
        mem_compress_length=2, emb_dim=256, num_layers=2, num_heads=8,
        cache_compress="none", cache_max_length=16,
        use_pos_emb="sin_pos", subsample_ratio=20).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model_ema"]
    diffusion.load_state_dict(prefixed_state(state, "diffusion."), strict=True)
    teacher.qformer.load_state_dict(prefixed_state(state, "memory.qformer."), strict=True)
    for parameter in list(diffusion.parameters()) + list(teacher.parameters()):
        parameter.requires_grad_(False)

    beta = torch.tensor(
        cosine_beta_schedule(T=50), device=device, dtype=torch.float32)
    bar_alpha = torch.cumprod(1.0 - beta, dim=0)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    feature_chunks, label_chunks, sequence_records = [], [], []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            obs = batch["obs"]
            condition = obs["dp_emb"][:, :2].to(device)
            memory = obs["dp_emb_mem"].to(device)
            steps = obs["ep_step"].to(device).long()
            lengths = obs["actual_num_frames"].to(device).long().flatten()
            action = batch["action"].to(device)
            action = dataset.normalizer["action"].normalize(action)
            batch_size, history, _ = memory.shape
            t = torch.randint(0, 50, (batch_size,), generator=generator, device=device)
            noise = torch.randn(action.shape, generator=generator, device=device)
            alpha = bar_alpha[t].view(batch_size, 1, 1)
            noisy = action * alpha.sqrt() + noise * (1.0 - alpha).sqrt()

            full_tokens = teacher(memory, steps)
            full_loss = per_sample_loss(
                diffusion(noisy, t, condition, full_tokens), noise)
            labels = torch.zeros(batch_size, history, device=device)
            for slot in range(history):
                if history == 1:
                    continue
                keep = torch.arange(history, device=device) != slot
                dropped_tokens = teacher(memory[:, keep], steps[:, keep])
                dropped_loss = per_sample_loss(
                    diffusion(noisy, t, condition, dropped_tokens), noise)
                labels[:, slot] = (dropped_loss - full_loss).clamp_min(0)

            positions = torch.arange(history, device=device)[None]
            valid = positions >= (history - lengths[:, None])
            features = retention_features(memory, steps)
            feature_chunks.append(features[valid].cpu())
            label_chunks.append(labels[valid].cpu())
            for row in range(batch_size):
                start = history - int(lengths[row])
                sequence_records.append({
                    "features": features[row, start:].cpu(),
                    "labels": labels[row, start:].cpu(),
                })
            if (batch_index + 1) % 8 == 0:
                print(f"oracle batches {batch_index + 1}/{args.max_batches}", flush=True)

    features = torch.cat(feature_chunks)
    labels = torch.cat(label_chunks)
    split = int(0.8 * len(features))
    permutation = torch.randperm(len(features), generator=torch.Generator().manual_seed(args.seed))
    train_idx, valid_idx = permutation[:split], permutation[split:]
    mean, std = labels[train_idx].mean(), labels[train_idx].std().clamp_min(1e-8)
    targets = (labels - mean) / std
    head = RetentionHead(256).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
    for epoch in range(30):
        order = train_idx[torch.randperm(len(train_idx))]
        head.train()
        for start in range(0, len(order), 1024):
            idx = order[start:start + 1024]
            prediction = head(features[idx].to(device))
            loss = F.smooth_l1_loss(prediction, targets[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == 29:
            head.eval()
            with torch.no_grad():
                pred = head(features[valid_idx].to(device)).cpu()
                val_loss = F.smooth_l1_loss(pred, targets[valid_idx]).item()
                corr = torch.corrcoef(torch.stack([pred, targets[valid_idx]]))[0, 1].item()
            print(f"epoch={epoch} val_loss={val_loss:.6f} corr={corr:.4f}", flush=True)

    # Offline top-k retained-importance upper-bound diagnostic.
    head.eval()
    totals = {name: 0.0 for name in ("oracle", "learned", "delta", "random")}
    denominator = 0.0
    rng = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for record in sequence_records:
            seq_features, seq_labels = record["features"], record["labels"]
            count = min(8, len(seq_labels))
            denominator += float(seq_labels.sum())
            learned_scores = head(seq_features.to(device)).cpu()
            current = seq_features[:, :256]
            previous = torch.cat([current[:1], current[:-1]])
            delta_scores = 1.0 - F.cosine_similarity(current, previous, dim=-1)
            random_scores = torch.rand(len(seq_labels), generator=rng)
            for name, scores in (
                    ("oracle", seq_labels), ("learned", learned_scores),
                    ("delta", delta_scores), ("random", random_scores)):
                chosen = scores.topk(count).indices
                totals[name] += float(seq_labels[chosen].sum())
    retained = {
        name: value / max(denominator, 1e-12) for name, value in totals.items()}
    result = {
        "samples": int(len(features)),
        "sequences": int(len(sequence_records)),
        "label_mean": float(mean),
        "label_std": float(std),
        "retained_oracle_importance_fraction": retained,
    }
    torch.save({
        "retention_head": head.cpu().state_dict(),
        "target_mean": mean,
        "target_std": std,
        "metadata": result,
    }, output / "retention_head.pt")
    torch.save({"features": features, "labels": labels}, output / "oracle_labels.pt")
    (output / "ORACLE_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
