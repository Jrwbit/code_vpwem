"""Utilities for fixed, auditable NCM counterfactual batches.

The manifest keeps pairing decisions outside the policy and the training
process.  Every ablation therefore consumes the exact same ordered examples;
only the partner permutation used by the counterfactual loss changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Union

import numpy as np
import torch


NCM_MANIFEST_VERSION = 1
REQUIRED_ARRAYS = (
    "batches",
    "branch",
    "episode_id",
    "decision_step",
    "matched_partner",
    "unmatched_partner",
    "alias_weight",
)


def manifest_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(Path(path).expanduser(), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ncm_manifest(
    path: Union[str, Path], dataset_size: int | None = None,
) -> Dict[str, np.ndarray]:
    """Load and validate a counterfactual-pair manifest without pickle."""
    resolved = Path(path).expanduser()
    with np.load(resolved, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_ARRAYS if key not in archive]
        if missing:
            raise ValueError(f"NCM manifest is missing arrays: {missing}")
        manifest = {key: np.asarray(archive[key]) for key in archive.files}

    version = int(np.asarray(manifest.get("schema_version", -1)).item())
    if version != NCM_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported NCM manifest version {version}; "
            f"expected {NCM_MANIFEST_VERSION}")

    batches = np.asarray(manifest["batches"], dtype=np.int64)
    if batches.ndim != 2 or batches.shape[0] == 0 or batches.shape[1] < 2:
        raise ValueError("NCM batches must have shape (num_batches, batch_size)")
    if np.any(batches < 0):
        raise ValueError("NCM batches contain negative dataset indices")
    if any(len(np.unique(row)) != len(row) for row in batches):
        raise ValueError("Every NCM batch must contain unique dataset indices")
    flat_batches = batches.reshape(-1)
    if len(np.unique(flat_batches)) != len(flat_batches):
        raise ValueError(
            "NCM dataset indices may occur in only one fixed batch per epoch")

    inferred_size = len(manifest["branch"])
    if dataset_size is not None and inferred_size != int(dataset_size):
        raise ValueError(
            f"Manifest dataset_size={inferred_size}, dataset has {dataset_size}")
    if np.any(batches >= inferred_size):
        raise ValueError("NCM batch index exceeds the manifest dataset size")

    for key in ("episode_id", "decision_step", "matched_partner",
                "unmatched_partner", "alias_weight"):
        if len(manifest[key]) != inferred_size:
            raise ValueError(
                f"Manifest array {key!r} has length {len(manifest[key])}; "
                f"expected {inferred_size}")

    used = np.unique(flat_batches)
    used_set = set(used.tolist())
    batch_membership = {
        int(index): set(row.tolist())
        for row in batches for index in row
    }
    branch = np.asarray(manifest["branch"], dtype=np.int64)
    episode = np.asarray(manifest["episode_id"], dtype=np.int64)
    for key in ("matched_partner", "unmatched_partner"):
        partner = np.asarray(manifest[key], dtype=np.int64)
        for index in used:
            donor = int(partner[index])
            if donor not in used_set:
                raise ValueError(
                    f"{key}[{index}]={donor} is absent from all fixed batches")
            if donor not in batch_membership[int(index)]:
                raise ValueError(
                    f"{key}[{index}]={donor} is absent from its fixed batch")
            if donor == int(index):
                raise ValueError(f"{key}[{index}] is a self-pair")
            if branch[donor] == branch[index]:
                raise ValueError(f"{key}[{index}] does not change hidden branch")
            if episode[donor] == episode[index]:
                raise ValueError(f"{key}[{index}] stays in the same episode")

    manifest["batches"] = batches
    manifest["sha256"] = np.asarray(manifest_sha256(resolved))
    return manifest


def manifest_metadata(manifest: Mapping[str, np.ndarray]) -> Dict:
    raw = manifest.get("metadata_json")
    if raw is None:
        return {}
    return json.loads(str(np.asarray(raw).item()))


class NCMPairedBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Repeat fixed pair-aware batches, shuffling only their order per epoch."""

    def __init__(self, batches: np.ndarray, seed: int = 0, shuffle: bool = True):
        batches = np.asarray(batches, dtype=np.int64)
        if batches.ndim != 2 or batches.shape[0] == 0:
            raise ValueError("batches must be a non-empty rank-2 array")
        self.batches = batches
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        order = np.arange(len(self.batches))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(order)
        self.epoch += 1
        for batch_index in order:
            row = self.batches[batch_index].copy()
            # Explicit matched/unmatched donors are resolved by global sample
            # ID, so row order may change without changing their semantics.
            # This restores the intended stochasticity of legacy batch.roll
            # for Random-CF while every arm still sees the same batch members.
            if self.shuffle:
                rng.shuffle(row)
            yield row.tolist()

    def __len__(self) -> int:
        return len(self.batches)


class NCMHybridBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Interleave full-dataset batches with fixed counterfactual pairs.

    Pair-only sampling is useful for short diagnostics, but a full-budget run
    would otherwise repeat a small decision subset thousands of times. This
    sampler keeps shuffled coverage of the complete sequence dataset and adds
    pair-aware batches at a declared rate. Equal seeds give every ablation arm
    the same base examples, pair examples, and ordering.
    """

    def __init__(
        self,
        pair_batches: np.ndarray,
        dataset_size: int,
        batch_size: int,
        pair_fraction: float,
        seed: int = 0,
    ):
        pair_batches = np.asarray(pair_batches, dtype=np.int64)
        if pair_batches.ndim != 2 or pair_batches.shape[0] == 0:
            raise ValueError("pair_batches must be a non-empty rank-2 array")
        if int(batch_size) != pair_batches.shape[1]:
            raise ValueError("batch_size must match pair batch width")
        if int(dataset_size) < int(batch_size):
            raise ValueError("dataset_size must contain at least one batch")
        if not 0.0 < float(pair_fraction) < 1.0:
            raise ValueError("pair_fraction must be strictly between 0 and 1")
        self.pair_batches = pair_batches
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.pair_fraction = float(pair_fraction)
        self.seed = int(seed)
        self.epoch = 0
        pair_indices = np.unique(self.pair_batches.reshape(-1))
        if np.any(pair_indices < 0) or np.any(pair_indices >= self.dataset_size):
            raise ValueError("pair batch index exceeds dataset_size")
        # A manifest sample has a valid partner only inside its fixed pair
        # batch.  Keep those samples out of ordinary shuffled batches; they
        # are still covered by the injected pair batches, while every other
        # sequence remains in the full-data stream.
        self.base_indices = np.setdiff1d(
            np.arange(self.dataset_size, dtype=np.int64),
            pair_indices,
            assume_unique=True,
        )
        self.base_batch_count = len(self.base_indices) // self.batch_size
        if self.base_batch_count == 0:
            raise ValueError(
                "hybrid sampling needs at least one non-pair base batch")
        self.pair_batch_count = max(1, int(round(
            self.base_batch_count * self.pair_fraction
            / (1.0 - self.pair_fraction))))

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        base_order = rng.permutation(self.base_indices)[
            :self.base_batch_count * self.batch_size]
        base_batches = base_order.reshape(
            self.base_batch_count, self.batch_size)

        pair_order = rng.integers(
            0, len(self.pair_batches), size=self.pair_batch_count)
        pair_batches = self.pair_batches[pair_order].copy()
        for row in pair_batches:
            rng.shuffle(row)

        schedule = np.concatenate([
            np.zeros(self.base_batch_count, dtype=np.int8),
            np.ones(self.pair_batch_count, dtype=np.int8),
        ])
        rng.shuffle(schedule)
        base_cursor = 0
        pair_cursor = 0
        for is_pair in schedule:
            if is_pair:
                yield pair_batches[pair_cursor].tolist()
                pair_cursor += 1
            else:
                yield base_batches[base_cursor].tolist()
                base_cursor += 1

    def __len__(self) -> int:
        return self.base_batch_count + self.pair_batch_count


def partner_positions(
    sample_indices: torch.Tensor,
    partner_indices: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve global manifest partner IDs to positions in the current batch."""
    sample_indices = sample_indices.reshape(-1).long()
    partner_indices = partner_indices.reshape(-1).long()
    if sample_indices.numel() != partner_indices.numel():
        raise ValueError("sample_indices and partner_indices must have equal length")
    matches = partner_indices[:, None].eq(sample_indices[None, :])
    if valid is None:
        valid = torch.ones_like(partner_indices, dtype=torch.bool)
    else:
        valid = valid.reshape(-1).bool()
    match_count = matches.sum(dim=1)
    if torch.any(match_count[valid] != 1):
        bad = torch.nonzero(valid & (match_count != 1)).reshape(-1).tolist()
        raise ValueError(f"NCM partners missing or duplicated in batch rows {bad}")
    positions = matches.to(torch.int64).argmax(dim=1)
    positions[~valid] = torch.arange(
        len(positions), device=positions.device, dtype=positions.dtype)[~valid]
    return positions
