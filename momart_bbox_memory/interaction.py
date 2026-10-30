"""Small, RGB-only interaction annotation utilities for MoMaRT.

The annotation contract is deliberately independent of the policy model:
annotators mark the active object and (when present) the placement region on
keyframes, a tracker may fill the frames between them, and boxes are derived
from masks.  Occlusion keeps the target identity but never invents a visible
box.  Crop transforms are applied to boxes with the same geometry as images.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class InteractionFrame:
    """One frame's task-conditioned visual annotation.

    Boxes use normalized ``xyxy`` coordinates in the *uncropped* image.
    ``target_id`` is retained while an object is occluded; ``target_box`` is
    then ``None`` and ``visible`` is false.
    """

    target_id: Optional[str] = None
    target_box: Optional[tuple[float, float, float, float]] = None
    placement_box: Optional[tuple[float, float, float, float]] = None
    visible: bool = False
    occluded: bool = False
    source: str = "manual"

    def as_dict(self):
        return asdict(self)


def _normalise_box(box) -> tuple[float, float, float, float]:
    values = np.asarray(box, dtype=np.float32).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("box must contain four finite values")
    x1, y1, x2, y2 = values.tolist()
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("box must be normalized xyxy with positive area")
    return tuple(float(x) for x in values)


def mask_to_box(mask, minimum_pixels: int = 3):
    """Return the tight normalized xyxy box of a visible-object mask.

    The mask is expected to be in the original image coordinate system.  A
    missing or too-small mask returns ``None`` instead of a fake full-image
    box.  This keeps the same absence semantics for training and rollout.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    y, x = np.where(mask)
    if len(x) < int(minimum_pixels):
        return None
    h, w = mask.shape
    return _normalise_box((x.min() / w, y.min() / h,
                           (x.max() + 1) / w, (y.max() + 1) / h))


def _interp_box(left, right, alpha):
    if left is None or right is None:
        return None
    return tuple(float(a + alpha * (b - a)) for a, b in zip(left, right))


def propagate_keyframes(keyframes: Mapping[int, InteractionFrame], length: int,
                        max_gap: int = 30) -> list[InteractionFrame]:
    """Propagate annotations between reviewed keyframes conservatively.

    Only matching ``target_id`` values are interpolated.  If either side is
    occluded or absent, intermediate frames retain the identity but have no
    visible target box.  Gaps larger than ``max_gap`` are left unannotated;
    this prevents a stale box from silently spanning a long scene change.
    """

    if length < 0 or max_gap < 0:
        raise ValueError("length and max_gap must be non-negative")
    points = {int(i): v for i, v in keyframes.items()}
    if any(i < 0 or i >= length for i in points):
        raise ValueError("keyframe index outside sequence")
    if any(not isinstance(v, InteractionFrame) for v in points.values()):
        raise TypeError("keyframes must contain InteractionFrame values")
    output = [InteractionFrame(source="unannotated") for _ in range(length)]
    if not points:
        return output
    ordered = sorted(points)
    for i in ordered:
        output[i] = points[i]
    for left_i, right_i in zip(ordered, ordered[1:]):
        left, right = points[left_i], points[right_i]
        gap = right_i - left_i
        if gap > max_gap:
            continue
        same_target = left.target_id is not None and left.target_id == right.target_id
        for i in range(left_i + 1, right_i):
            alpha = (i - left_i) / gap
            target_box = _interp_box(left.target_box, right.target_box, alpha)
            placement_box = _interp_box(left.placement_box, right.placement_box, alpha)
            if not same_target or left.occluded or right.occluded:
                target_box = None
            output[i] = InteractionFrame(
                target_id=left.target_id if same_target else None,
                target_box=target_box,
                placement_box=placement_box,
                visible=target_box is not None,
                occluded=same_target and target_box is None,
                source="propagated",
            )
    return output


def transform_box_to_crop(box, crop_xyxy, image_size):
    """Apply an image crop to a normalized original-image box.

    ``crop_xyxy`` is pixel ``(x1, y1, x2, y2)`` with half-open bounds and
    ``image_size`` is ``(height, width)``.  The result is normalized in the
    cropped image, matching the image seen by the policy after augmentation.
    """

    if box is None:
        return None
    box = _normalise_box(box)
    h, w = map(int, image_size)
    if h <= 0 or w <= 0:
        raise ValueError("image_size must be positive")
    crop = np.asarray(crop_xyxy, dtype=np.float32).reshape(-1)
    if crop.size != 4 or not np.isfinite(crop).all():
        raise ValueError("crop_xyxy must contain four finite pixel values")
    cx1, cy1, cx2, cy2 = crop.tolist()
    if not (0 <= cx1 < cx2 <= w and 0 <= cy1 < cy2 <= h):
        raise ValueError("crop must be a positive in-image half-open rectangle")
    x1, y1, x2, y2 = np.asarray(box) * [w, h, w, h]
    x1, x2 = max(x1, cx1), min(x2, cx2)
    y1, y2 = max(y1, cy1), min(y2, cy2)
    if x2 <= x1 or y2 <= y1:
        return None
    return _normalise_box(((x1 - cx1) / (cx2 - cx1),
                           (y1 - cy1) / (cy2 - cy1),
                           (x2 - cx1) / (cx2 - cx1),
                           (y2 - cy1) / (cy2 - cy1)))


def annotation_hit_rate(predicted: Sequence[Optional[Sequence[float]]],
                        reference: Sequence[Optional[Sequence[float]]],
                        iou_threshold: float = .5) -> dict:
    """Report visible-frame IoU hit rate and missing-box rate.

    This is an annotation/detector diagnostic, not a policy success rate.
    Frames where both sides are absent are excluded from the visible IoU
    denominator and counted separately as ``true_absent``.
    """

    if len(predicted) != len(reference):
        raise ValueError("predicted and reference lengths differ")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        inter = max(0., min(ax2, bx2) - max(ax1, bx1)) * max(0., min(ay2, by2) - max(ay1, by1))
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.

    visible = hits = missing = false_positive = true_absent = 0
    for pred, ref in zip(predicted, reference):
        if ref is None and pred is None:
            true_absent += 1
        elif ref is None:
            false_positive += 1
        elif pred is None:
            visible += 1; missing += 1
        else:
            visible += 1
            hits += int(iou(_normalise_box(pred), _normalise_box(ref)) >= iou_threshold)
    return dict(visible_frames=visible, iou_hits=hits,
                hit_rate=hits / visible if visible else 0.,
                missing_rate=missing / visible if visible else 0.,
                false_positive=false_positive, true_absent=true_absent)
