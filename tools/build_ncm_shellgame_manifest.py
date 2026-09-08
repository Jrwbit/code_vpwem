#!/usr/bin/env python3
"""Build and audit clean MIKASA NCM alias pairs.

The released demonstrations start their branch-specific expert action at step
5.  With ``obs_steps=2`` there is no clean two-frame post-cue current window.
The primary protocol therefore uses a physically collected branch-neutral hold
through step 5, marked by ``NCM_PHYSICAL_HOLD.json``.  An opt-in one-step
offline hold remains available for pipeline smoke tests only.

For ShellGameTouch, oracle branch labels can be reconstructed from the public
collector protocol.  RememberColor uses the sidecar labels written by the
physical neutral-prefix collector.  Labels are stored only in the manifest and
are never added to policy observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from cleandiffuser.dataset.ncm_pairing import NCM_MANIFEST_VERSION
from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic


SHAPE_META = {
    "action": {"shape": [8]},
    "obs": {
        "rgb": {"shape": [3, 128, 128], "type": "rgb"},
        "rgb_wrist": {"shape": [3, 128, 128], "type": "rgb"},
        "joints": {"shape": [25], "type": "low_dim"},
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--normalizer-dataset", default=None,
        help="Dataset whose joint min/max was used by the frozen checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--pairing-seed", type=int, default=100)
    parser.add_argument("--num-branches", type=int)
    parser.add_argument("--decision-step", type=int, default=6)
    parser.add_argument("--hold-step", type=int, default=5)
    parser.add_argument("--hold-steps", type=int, default=0)
    parser.add_argument("--branch-labels", default=None)
    parser.add_argument("--audit-pairs", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def insert_hold(array: np.ndarray, key: str, step: int, count: int):
    if count == 0:
        return array
    if key in ("rgb", "joints"):
        held = np.repeat(array[step:step + 1], count, axis=0)
        return np.concatenate([array[:step + 1], held, array[step + 1:]], axis=0)
    neutral = np.zeros_like(array[step:step + 1])
    if key == "action" and neutral.shape[-1] > 0 and step > 0:
        neutral[..., -1] = array[step - 1:step, -1]
    neutral = np.repeat(neutral, count, axis=0)
    return np.concatenate([array[:step], neutral, array[step:]], axis=0)


def scan_episodes(
    dataset: Path, hold_step: int, hold_steps: int, episode_timeout: int,
):
    paths = sorted(
        dataset.glob("train_data_*.npz"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    lengths, final_tcp_y = [], []
    joints_min = np.full(25, np.inf, dtype=np.float64)
    joints_max = np.full(25, -np.inf, dtype=np.float64)
    for expected_id, path in enumerate(paths):
        raw_id = int(path.stem.rsplit("_", 1)[1])
        if raw_id != expected_id:
            raise ValueError(
                f"Episode files must be contiguous; expected {expected_id}, got {raw_id}")
        with np.load(path) as episode:
            success = insert_hold(
                episode["success"], "success", hold_step, hold_steps)
            hits = np.flatnonzero(success == 1)
            length = min(
                int(hits[0]) + 1 + 8, int(episode_timeout), len(success)
            ) if len(hits) else 0
            lengths.append(length)
            if length:
                joints = insert_hold(
                    episode["joints"], "joints", hold_step, hold_steps)[:length]
                joints_min = np.minimum(joints_min, joints.min(axis=0))
                joints_max = np.maximum(joints_max, joints.max(axis=0))
                final_tcp_y.append(float(joints[-1, 1]))
            else:
                final_tcp_y.append(float("nan"))
    if not paths:
        raise ValueError(f"No train_data_*.npz files in {dataset}")
    return paths, np.asarray(lengths, dtype=np.int64), np.asarray(
        final_tcp_y, dtype=np.float32), joints_min, joints_max


def reconstruct_branches(num_episodes: int, output: Path) -> np.ndarray:
    if num_episodes % 250:
        raise ValueError(
            "Public collector reconstruction requires a multiple of 250 episodes")
    labels = []
    for batch_seed in range(num_episodes // 250):
        # ManiSkill's BatchedRNG uses the reset seed for slot zero and draws
        # the remaining per-slot seeds from that RandomState. TableSceneBuilder
        # consumes slot-zero robot noise before ShellGame samples the branch.
        main_rng = np.random.RandomState(batch_seed)
        episode_seeds = np.concatenate([
            np.asarray([batch_seed], dtype=np.int64),
            main_rng.randint(2 ** 31, size=249),
        ])
        rngs = [np.random.RandomState(int(seed)) for seed in episode_seeds]
        rngs[0].normal(0, 0.02, (250, 9))
        labels.extend(int(rng.choice(3)) for rng in rngs)
    labels = np.asarray(labels, dtype=np.int8)
    np.save(output, labels)
    return labels


def load_condition_encoder(checkpoint_path: Path, device: str):
    encoder = MultiImageObsConditionRobomimic(
        shape_meta=SHAPE_META, emb_dim=256, rgb_model_name="resnet18",
        resize_shape=None, crop_shape=[100, 100], random_crop=True,
        use_group_norm=True, use_seq=True, keep_horizon_dims=True,
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_ema"]
    prefix = "condition."
    condition_state = {
        key[len(prefix):]: value for key, value in state.items()
        if key.startswith(prefix)
    }
    message = encoder.load_state_dict(condition_state, strict=True)
    print(f"Loaded frozen condition encoder: {message}", flush=True)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def decision_examples(
    paths: Sequence[Path], lengths: np.ndarray, decision_step: int,
    hold_step: int, hold_steps: int, joints_min: np.ndarray,
    joints_max: np.ndarray,
):
    examples, target_actions, episode_ids, dataset_indices = [], [], [], []
    episode_offsets = np.cumsum(np.concatenate([[0], lengths[:-1]]))
    joints_range = np.where(joints_max > joints_min,
                            joints_max - joints_min, 1.0)
    for episode_id, (path, length) in enumerate(zip(paths, lengths)):
        if length <= decision_step:
            continue
        with np.load(path) as episode:
            rgb = insert_hold(episode["rgb"], "rgb", hold_step, hold_steps)
            joints = insert_hold(
                episode["joints"], "joints", hold_step, hold_steps)
            actions = insert_hold(
                episode["action"], "action", hold_step, hold_steps)
            start = decision_step - 1
            rgb_window = rgb[start:decision_step + 1]
            joint_window = joints[start:decision_step + 1]
            examples.append({
                "rgb": np.moveaxis(rgb_window[..., :3], -1, 1).astype(
                    np.float32) / 255.0,
                "rgb_wrist": np.moveaxis(
                    rgb_window[..., 3:], -1, 1).astype(np.float32) / 255.0,
                "joints": ((joint_window - joints_min) / joints_range * 2.0
                            - 1.0).astype(np.float32),
            })
            target_actions.append(
                np.asarray(actions[decision_step], dtype=np.float32))
        episode_ids.append(episode_id)
        # With horizon=9, pad_before=1, pad_after=7, every positive
        # episode contributes exactly ``length`` samples and local sample k
        # predicts the action at raw/derived timestep k.
        dataset_indices.append(int(episode_offsets[episode_id] + decision_step))
    return (
        examples,
        np.asarray(target_actions, dtype=np.float32),
        np.asarray(episode_ids, dtype=np.int64),
        np.asarray(dataset_indices, dtype=np.int64),
    )


def encode_examples(examples, encoder, batch_size: int, device: str):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            batch = {
                key: torch.from_numpy(np.stack([item[key] for item in chunk])).to(device)
                for key in ("rgb", "rgb_wrist", "joints")
            }
            encoded = encoder(batch).flatten(start_dim=1)
            encoded = torch.nn.functional.normalize(encoded, dim=-1)
            outputs.append(encoded.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def preserves_cross_branch_feasibility(
    remaining: np.ndarray, left: int, right: int,
) -> bool:
    updated = remaining.copy()
    updated[left] -= 1
    updated[right] -= 1
    total = int(updated.sum())
    return total == 0 or int(updated.max()) <= total // 2


def greedy_cross_branch_matching(
    indices: np.ndarray, branch: np.ndarray, similarity: np.ndarray,
    num_branches: int, maximize: bool = False,
) -> List[Tuple[int, int]]:
    """Build a deterministic feasible perfect matching across branches."""
    indices = np.asarray(indices, dtype=np.int64)
    local_branch = branch[indices]
    remaining_counts = np.bincount(
        local_branch, minlength=num_branches).astype(np.int64)
    edges = []
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            if local_branch[left] != local_branch[right]:
                score = float(similarity[indices[left], indices[right]])
                edges.append((score, left, right))
    edges.sort(key=lambda item: item[0], reverse=maximize)
    available = np.ones(len(indices), dtype=bool)
    pairs = []
    while available.any():
        selected = None
        for _, left, right in edges:
            if not (available[left] and available[right]):
                continue
            if preserves_cross_branch_feasibility(
                    remaining_counts, local_branch[left], local_branch[right]):
                selected = (left, right)
                break
        if selected is None:
            raise RuntimeError("Could not construct a cross-branch perfect matching")
        left, right = selected
        available[[left, right]] = False
        remaining_counts[local_branch[left]] -= 1
        remaining_counts[local_branch[right]] -= 1
        pairs.append((int(indices[left]), int(indices[right])))
    return pairs


def balanced_candidate_indices(
    eligible: np.ndarray, branch: np.ndarray, batch_size: int,
    num_branches: int, rng,
) -> np.ndarray:
    """Drop at most a small tail so every fixed batch is pair-feasible."""
    pools = {label: eligible[branch[eligible] == label].copy()
             for label in range(num_branches)}
    for pool in pools.values():
        rng.shuffle(pool)
    cursors = {label: 0 for label in pools}
    batches = []
    rotation = 0
    while True:
        base = batch_size // num_branches
        counts = [base] * num_branches
        for offset in range(batch_size - num_branches * base):
            counts[(rotation + offset) % num_branches] += 1
        if any(cursors[label] + counts[label] > len(pools[label])
               for label in range(num_branches)):
            break
        row = []
        for label in range(num_branches):
            begin = cursors[label]
            end = begin + counts[label]
            row.extend(pools[label][begin:end].tolist())
            cursors[label] = end
        rng.shuffle(row)
        batches.append(row)
        rotation = (rotation + 1) % num_branches
    return np.asarray(batches, dtype=np.int64)


def nearest_centroid_probe(features, labels, num_branches, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    split = int(0.8 * len(order))
    train, test = order[:split], order[split:]
    centroids = np.stack([
        features[train][labels[train] == label].mean(axis=0)
        for label in range(num_branches)
    ])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-8)
    prediction = (features[test] @ centroids.T).argmax(axis=1)
    return float(np.mean(prediction == labels[test]))


def raw_current_probe_features(examples):
    """Center-crop/pool raw current input for a non-learned leakage probe."""
    modalities = []
    for key in ("rgb", "rgb_wrist"):
        value = np.stack([example[key] for example in examples])
        value = value[..., 14:114, 14:114]
        value = value.reshape(
            len(value), 2, 3, 20, 5, 20, 5).mean(axis=(4, 6))
        modalities.append(value.reshape(len(value), -1))
    modalities.append(np.stack(
        [example["joints"] for example in examples]).reshape(len(examples), -1))
    standardized = []
    for value in modalities:
        value = (value - value.mean(axis=0, keepdims=True)) / value.std(
            axis=0, keepdims=True).clip(1e-6)
        standardized.append(value / np.sqrt(value.shape[1]))
    features = np.concatenate(standardized, axis=1).astype(np.float32)
    return features / np.linalg.norm(
        features, axis=1, keepdims=True).clip(1e-8)


def leave_one_group_out_1nn_probe(features, labels, groups):
    if len(np.unique(groups)) < 2:
        return float("nan")
    predictions, targets = [], []
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        similarity = features[test] @ features[train].T
        predictions.append(labels[train][similarity.argmax(axis=1)])
        targets.append(labels[test])
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    return float(np.mean(prediction == target))


def save_pair_audit(
    audit_dir: Path, paths: Sequence[Path], pairs, decision_step: int,
    hold_step: int, hold_steps: int, branch: np.ndarray, count: int,
):
    audit_dir.mkdir(parents=True, exist_ok=True)
    for pair_id, (left, right) in enumerate(pairs[:count]):
        panels = []
        for episode_id in (left, right):
            with np.load(paths[episode_id]) as episode:
                rgb = insert_hold(
                    episode["rgb"], "rgb", hold_step, hold_steps)
                panels.append(Image.fromarray(rgb[decision_step, ..., :3]))
        canvas = Image.new("RGB", (256, 152), "white")
        canvas.paste(panels[0], (0, 24))
        canvas.paste(panels[1], (128, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f"ep {left} branch {int(branch[left])}", fill="black")
        draw.text((132, 4), f"ep {right} branch {int(branch[right])}", fill="black")
        canvas.save(audit_dir / f"pair_{pair_id:03d}.png")


def main():
    args = parse_args()
    dataset = Path(args.dataset).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if args.hold_steps not in (0, 1):
        raise ValueError("hold_steps must be 0 (physical) or 1 (smoke only)")
    if args.decision_step != args.hold_step + 1:
        raise ValueError("decision_step must immediately follow hold_step")
    physical_hold_marker = dataset / "NCM_PHYSICAL_HOLD.json"
    marker = None
    if args.hold_steps == 0:
        if not physical_hold_marker.is_file():
            raise ValueError(
                "hold_steps=0 requires a physically collected neutral-prefix "
                f"dataset with marker {physical_hold_marker}; use hold_steps=1 "
                "only for a derived-data smoke test")
        with open(physical_hold_marker, encoding="utf-8") as stream:
            marker = json.load(stream)
        if not bool(marker.get("complete", False)):
            raise ValueError("Physical-hold dataset marker is not complete")
        if int(marker.get("neutral_through_step", -1)) != args.hold_step:
            raise ValueError("Physical-hold marker does not match hold_step")
        dataset_role = marker.get("dataset_role")
        valid_roles = {
            "physical_closed_loop_decision_supervision_at_step_"
            f"{args.decision_step}",
            "hybrid_decision_supervision_at_step_"
            f"{args.decision_step}_only",
        }
        if dataset_role not in valid_roles:
            raise ValueError(
                "Physical-hold data does not declare an audited decision "
                "supervision protocol")
        expected_shift = (
            0 if dataset_role.startswith("physical_closed_loop") else 1)
        if int(marker.get("source_action_shift", -1)) != expected_shift:
            raise ValueError(
                "Physical-hold source_action_shift disagrees with its role")
        if dataset_role.startswith("physical_closed_loop"):
            neutral_action = np.asarray(
                marker.get("neutral_action", []), dtype=np.float32)
            expected_neutral = np.asarray(
                [0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            if not np.array_equal(neutral_action, expected_neutral):
                raise ValueError(
                    "Closed-loop neutral prefix must use zero arm deltas "
                    "and an open-gripper (+1) command")
            physical_successes = int(marker.get(
                "physical_rollout_successful_episodes", -1))
            episodes = int(marker.get("episodes", 0))
            if episodes <= 0 or physical_successes / episodes < 0.90:
                raise ValueError(
                    "Closed-loop neutral-prefix demonstrations fail the "
                    "pre-registered 90% physical success gate")
    if args.batch_size % 2 or args.batch_size < 4:
        raise ValueError("batch_size must be even and at least four")

    num_branches = args.num_branches
    if num_branches is None and marker is not None:
        num_branches = marker.get("num_branches")
    if num_branches is None:
        num_branches = 3
    num_branches = int(num_branches)
    if num_branches < 2 or num_branches > args.batch_size:
        raise ValueError(
            "num_branches must be between two and the manifest batch size")

    episode_timeout = int(
        marker.get("episode_timeout", 90) if marker is not None else 90)

    paths, lengths, final_tcp_y, _, _ = scan_episodes(
        dataset, args.hold_step, args.hold_steps, episode_timeout)
    normalizer_dataset = (Path(args.normalizer_dataset).expanduser()
                          if args.normalizer_dataset else dataset)
    _, _, _, joints_min, joints_max = scan_episodes(
        normalizer_dataset, args.hold_step, 0, episode_timeout)
    labels_path = (Path(args.branch_labels).expanduser()
                   if args.branch_labels else output / "branch_labels.npy")
    if labels_path.exists():
        branch_by_episode = np.load(labels_path, allow_pickle=False).astype(np.int8)
    else:
        branch_by_episode = reconstruct_branches(len(paths), labels_path)
    if len(branch_by_episode) != len(paths):
        raise ValueError("Branch label count does not match episode count")
    if np.any(branch_by_episode < 0) or np.any(
            branch_by_episode >= num_branches):
        raise ValueError("Branch labels fall outside [0, num_branches)")

    examples, target_actions, episode_ids, sample_indices = decision_examples(
        paths, lengths, args.decision_step, args.hold_step, args.hold_steps,
        joints_min, joints_max)
    encoder = load_condition_encoder(checkpoint, args.device)
    features = encode_examples(
        examples, encoder, args.encoder_batch_size, args.device)
    decision_branch = branch_by_episode[episode_ids]
    similarity = features @ features.T

    total_samples = int(lengths.sum())
    branch = np.repeat(branch_by_episode, lengths).astype(np.int8)
    episode_id = np.repeat(np.arange(len(paths), dtype=np.int64), lengths)
    decision_steps = np.concatenate([
        np.arange(length, dtype=np.int16) for length in lengths
    ])
    matched_partner = np.full(total_samples, -1, dtype=np.int64)
    unmatched_partner = np.full(total_samples, -1, dtype=np.int64)
    alias_weight = np.zeros(total_samples, dtype=np.float32)

    rng = np.random.default_rng(args.pairing_seed)
    local_candidates = np.arange(len(sample_indices), dtype=np.int64)
    candidate_batches = balanced_candidate_indices(
        local_candidates, decision_branch, args.batch_size,
        num_branches, rng)
    if candidate_batches.size == 0:
        raise RuntimeError("No complete branch-balanced NCM batch is feasible")
    selected_candidates = candidate_batches.reshape(-1)
    # Match over the full selected pool first, then pack complete matched pairs
    # into fixed batches. Randomly pre-batching candidates would make
    # "matched" mean merely best-of-15 instead of a dataset-level alias.
    global_matched = greedy_cross_branch_matching(
        selected_candidates, decision_branch, similarity,
        num_branches, maximize=True)
    rng.shuffle(global_matched)
    fixed_batches, matched_episode_pairs = [], []
    matched_sims, unmatched_sims = [], []
    pairs_per_batch = args.batch_size // 2
    for begin in range(0, len(global_matched), pairs_per_batch):
        matched_local = global_matched[begin:begin + pairs_per_batch]
        if len(matched_local) != pairs_per_batch:
            raise RuntimeError("A partial NCM pair batch was produced")
        local_batch = np.asarray(
            [item for pair in matched_local for item in pair], dtype=np.int64)
        rng.shuffle(local_batch)
        unmatched_local = greedy_cross_branch_matching(
            local_batch, decision_branch, similarity,
            num_branches, maximize=False)
        global_batch = sample_indices[local_batch]
        fixed_batches.append(global_batch)
        for pair_set, destination, scores in (
            (matched_local, matched_partner, matched_sims),
            (unmatched_local, unmatched_partner, unmatched_sims),
        ):
            for left_local, right_local in pair_set:
                left = int(sample_indices[left_local])
                right = int(sample_indices[right_local])
                destination[left] = right
                destination[right] = left
                scores.append(float(similarity[left_local, right_local]))
        for left_local, right_local in matched_local:
            matched_episode_pairs.append((
                int(episode_ids[left_local]), int(episode_ids[right_local])))

    batches = np.asarray(fixed_batches, dtype=np.int64)
    used = np.unique(batches.reshape(-1))
    alias_weight[used] = 1.0
    if np.any(matched_partner[used] < 0) or np.any(unmatched_partner[used] < 0):
        raise RuntimeError("A fixed-batch sample is missing a partner")

    tcp_medians = {
        str(label): float(np.nanmedian(final_tcp_y[branch_by_episode == label]))
        for label in range(num_branches)
    }
    raw_probe_features = raw_current_probe_features(examples)
    collector_groups = episode_ids // 250
    standardized_actions = (
        target_actions - target_actions.mean(axis=0, keepdims=True)
    ) / target_actions.std(axis=0, keepdims=True).clip(1e-6)
    standardized_actions /= np.linalg.norm(
        standardized_actions, axis=1, keepdims=True).clip(1e-8)
    decision_lookup = {
        int(sample_index): position
        for position, sample_index in enumerate(sample_indices)
    }

    def partner_action_l2(partner_array):
        values = []
        for sample_index in used:
            donor_index = int(partner_array[sample_index])
            left = decision_lookup[int(sample_index)]
            right = decision_lookup[donor_index]
            values.append(float(np.linalg.norm(
                target_actions[left] - target_actions[right])))
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        }

    report = {
        "schema_version": NCM_MANIFEST_VERSION,
        "dataset": str(dataset),
        "normalizer_dataset": str(normalizer_dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "episodes": len(paths),
        "positive_episodes": int(np.sum(lengths > 0)),
        "zero_length_episode_ids": np.flatnonzero(lengths == 0).tolist(),
        "dataset_samples": total_samples,
        "decision_candidates": len(sample_indices),
        "paired_samples": int(len(used)),
        "pair_coverage": float(len(used) / len(sample_indices)),
        "num_batches": int(len(batches)),
        "batch_size": int(args.batch_size),
        "num_branches": num_branches,
        "decision_step": int(args.decision_step),
        "hold_step": int(args.hold_step),
        "hold_steps": int(args.hold_steps),
        "dataset_role": (
            marker.get("dataset_role") if marker is not None else
            "derived_virtual_hold_smoke"),
        "physical_success_rate": (
            float(marker.get(
                "physical_rollout_successful_episodes",
                marker.get("physical_replay_successful_episodes", 0))
                / marker["episodes"]) if marker is not None else None),
        "branch_counts": np.bincount(
            decision_branch, minlength=num_branches).tolist(),
        "matched_cosine": {
            "mean": float(np.mean(matched_sims)),
            "p05": float(np.quantile(matched_sims, 0.05)),
            "p50": float(np.quantile(matched_sims, 0.50)),
            "p95": float(np.quantile(matched_sims, 0.95)),
        },
        "unmatched_cosine": {
            "mean": float(np.mean(unmatched_sims)),
            "p05": float(np.quantile(unmatched_sims, 0.05)),
            "p50": float(np.quantile(unmatched_sims, 0.50)),
            "p95": float(np.quantile(unmatched_sims, 0.95)),
        },
        "current_branch_nearest_centroid_probe_acc": nearest_centroid_probe(
            features, decision_branch, num_branches, args.pairing_seed),
        "current_branch_lobo_1nn_probe_acc": leave_one_group_out_1nn_probe(
            raw_probe_features, decision_branch, collector_groups),
        "current_branch_probe_majority_acc": float(
            np.bincount(decision_branch, minlength=num_branches).max()
            / len(decision_branch)),
        "target_action_branch_lobo_1nn_probe_acc": (
            leave_one_group_out_1nn_probe(
                standardized_actions, decision_branch, collector_groups)),
        "matched_pair_target_action_l2": partner_action_l2(
            matched_partner),
        "unmatched_pair_target_action_l2": partner_action_l2(
            unmatched_partner),
        "terminal_tcp_y_median_by_branch": tcp_medians,
        "pairing_seed": int(args.pairing_seed),
    }
    metadata_json = json.dumps(report, sort_keys=True)
    manifest_path = output / "ncm_shellgame_pairs.npz"
    np.savez_compressed(
        manifest_path,
        schema_version=np.asarray(NCM_MANIFEST_VERSION, dtype=np.int64),
        metadata_json=np.asarray(metadata_json),
        batches=batches,
        branch=branch,
        episode_id=episode_id,
        decision_step=decision_steps,
        matched_partner=matched_partner,
        unmatched_partner=unmatched_partner,
        alias_weight=alias_weight,
    )
    report["manifest"] = str(manifest_path)
    report["manifest_sha256"] = sha256_file(manifest_path)
    with open(output / "pair_quality.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    with open(output / "pair_quality.md", "w", encoding="utf-8") as stream:
        stream.write("# NCM ShellGameTouch pair-quality audit\n\n")
        stream.write("```json\n")
        stream.write(json.dumps(report, indent=2, sort_keys=True))
        stream.write("\n```\n")
    save_pair_audit(
        output / "pair_audit", paths, matched_episode_pairs,
        args.decision_step, args.hold_step, args.hold_steps,
        branch_by_episode, args.audit_pairs)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
