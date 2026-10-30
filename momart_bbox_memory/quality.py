"""Shared RGB box acceptance rule for cache generation and online inference."""
import numpy as np

VERSION = "quality_gate_v1"
DEFAULT_THRESHOLD = 0.35

def quality_gate(boxes, valid, ambiguous, scores, threshold=0.35):
    boxes = np.asarray(boxes, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    ambiguous = np.asarray(ambiguous, dtype=bool)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if any(len(x) != len(boxes) for x in (valid, ambiguous, scores)):
        raise ValueError("sidecar arrays have different lengths")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    geometry = (np.isfinite(boxes).all(axis=1)
                & (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0)
                & (boxes[:, 2] <= 1) & (boxes[:, 3] <= 1)
                & (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1]))
    accepted = valid & ~ambiguous & np.isfinite(scores) & (scores >= threshold) & geometry
    return accepted

