"""Pure NumPy helpers for RememberColor rollout diagnostics.

The helpers in this module intentionally have no simulator or policy imports so
their metric semantics can be unit tested without reserving a GPU.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


COLOR_NAMES = ("Red", "Lime", "Blue", "Yellow", "Magenta", "Cyan",
               "Maroon", "Olive", "Teal")


def infer_primary_color_branch(rgb: np.ndarray) -> int:
    """Infer Red/Lime/Blue from a uint8 RGB image without simulator state."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    primary = image[..., :3].astype(np.float32)
    counts = []
    for channel in range(3):
        other = [idx for idx in range(3) if idx != channel]
        mask = ((primary[..., channel] >= 200)
                & (primary[..., other[0]] <= 80)
                & (primary[..., other[1]] <= 80))
        counts.append(int(mask.sum()))
    if max(counts) > 0:
        return int(np.argmax(counts))

    # Antialiasing or a changed renderer may eliminate fully saturated pixels.
    # Channel dominance remains a useful, deterministic fallback.
    dominance = []
    for channel in range(3):
        other = [idx for idx in range(3) if idx != channel]
        margin = primary[..., channel] - np.maximum(
            primary[..., other[0]], primary[..., other[1]])
        dominance.append(float(np.maximum(margin, 0).sum()))
    return int(np.argmax(dominance))


def array_comparison(reference: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Return compact exact and numeric comparison statistics."""
    reference = np.asarray(reference)
    actual = np.asarray(actual)
    if reference.shape != actual.shape:
        raise ValueError(
            f"shape mismatch: reference={reference.shape}, actual={actual.shape}")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "exact": bool(np.array_equal(reference, actual)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(np.max(np.abs(delta))),
    }


@dataclass
class EpisodeBranchTracker:
    """Accumulate policy-free branch and success metrics for a vector rollout.

    ``branch_choice`` is operationally defined as the branch with the smallest
    TCP-to-object distance reached after ``decision_start``.  Cue-only frames
    are excluded so displaying the target at the center cannot count as a
    policy choice.
    """

    ground_truth: np.ndarray
    n_branches: int
    decision_start: int = 10

    def __post_init__(self) -> None:
        self.ground_truth = np.asarray(self.ground_truth, dtype=np.int64).reshape(-1)
        if np.any((self.ground_truth < 0) | (self.ground_truth >= self.n_branches)):
            raise ValueError("ground-truth branch is outside the configured range")
        batch = self.ground_truth.shape[0]
        self.min_branch_distance = np.full(
            (batch, self.n_branches), np.inf, dtype=np.float32)
        self.final_branch_distance = np.full(
            (batch, self.n_branches), np.nan, dtype=np.float32)
        self.min_correct_target_distance = np.full(batch, np.inf, dtype=np.float32)
        self.min_correct_target_time = np.full(batch, -1, dtype=np.int32)
        self.static_at_min_correct = np.zeros(batch, dtype=bool)
        self.ever_static = np.zeros(batch, dtype=bool)
        self.ever_placed = np.zeros(batch, dtype=bool)
        self.success_time = np.full(batch, -1, dtype=np.int32)

    def update(
        self,
        elapsed_steps: np.ndarray,
        branch_distances: np.ndarray,
        is_static: np.ndarray,
        is_placed: np.ndarray,
        success: np.ndarray,
    ) -> None:
        elapsed_steps = np.asarray(elapsed_steps)
        branch_distances = np.asarray(branch_distances, dtype=np.float32)
        is_static = np.asarray(is_static, dtype=bool)
        is_placed = np.asarray(is_placed, dtype=bool)
        success = np.asarray(success, dtype=bool)

        if elapsed_steps.ndim == 1:
            elapsed_steps = elapsed_steps[:, None]
        if branch_distances.ndim == 2:
            branch_distances = branch_distances[:, None, :]
        expected = branch_distances.shape[:2]
        for name, value in (
            ("elapsed_steps", elapsed_steps), ("is_static", is_static),
            ("is_placed", is_placed), ("success", success),
        ):
            if value.ndim == 1:
                value = value[:, None]
                if name == "is_static":
                    is_static = value
                elif name == "is_placed":
                    is_placed = value
                elif name == "success":
                    success = value
            if value.shape[:2] != expected:
                raise ValueError(
                    f"{name} has {value.shape[:2]}, expected {expected}")
        if branch_distances.shape[0] != self.ground_truth.shape[0]:
            raise ValueError("rollout batch size changed")
        if branch_distances.shape[2] != self.n_branches:
            raise ValueError("rollout branch count changed")

        for episode in range(branch_distances.shape[0]):
            for substep in range(branch_distances.shape[1]):
                step = int(elapsed_steps[episode, substep])
                distances = branch_distances[episode, substep]
                self.final_branch_distance[episode] = distances
                self.ever_static[episode] |= bool(is_static[episode, substep])
                self.ever_placed[episode] |= bool(is_placed[episode, substep])
                if success[episode, substep] and self.success_time[episode] < 0:
                    self.success_time[episode] = step
                if step < self.decision_start:
                    continue
                self.min_branch_distance[episode] = np.minimum(
                    self.min_branch_distance[episode], distances)
                correct_distance = float(distances[self.ground_truth[episode]])
                if correct_distance < self.min_correct_target_distance[episode]:
                    self.min_correct_target_distance[episode] = correct_distance
                    self.min_correct_target_time[episode] = step
                    self.static_at_min_correct[episode] = bool(
                        is_static[episode, substep])

    def finalize(self) -> Dict[str, np.ndarray]:
        valid = np.isfinite(self.min_branch_distance).any(axis=1)
        branch_choice = np.full(self.ground_truth.shape, -1, dtype=np.int64)
        branch_choice[valid] = np.argmin(
            self.min_branch_distance[valid], axis=1).astype(np.int64)
        final_valid = np.isfinite(self.final_branch_distance).any(axis=1)
        final_branch = np.full(self.ground_truth.shape, -1, dtype=np.int64)
        final_branch[final_valid] = np.nanargmin(
            self.final_branch_distance[final_valid], axis=1).astype(np.int64)
        return {
            "ground_truth_branch": self.ground_truth.copy(),
            "branch_choice": branch_choice,
            "branch_choice_correct": branch_choice == self.ground_truth,
            "final_closest_branch": final_branch,
            "final_closest_correct": final_branch == self.ground_truth,
            "min_branch_distance": self.min_branch_distance.copy(),
            "final_branch_distance": self.final_branch_distance.copy(),
            "min_correct_target_distance": self.min_correct_target_distance.copy(),
            "min_correct_target_time": self.min_correct_target_time.copy(),
            "static_at_min_correct": self.static_at_min_correct.copy(),
            "ever_static": self.ever_static.copy(),
            "ever_placed": self.ever_placed.copy(),
            "success_time": self.success_time.copy(),
        }
