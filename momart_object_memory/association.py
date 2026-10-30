"""Causal association using frozen ROI appearance and bounded image motion."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AssociationConfig:
    # Legacy remains available as the immutable diagnostic comparator.
    method: str = 'iou_greedy'
    appearance_weight: float = .5
    motion_weight: float = .2
    min_cosine: float = .8
    max_age_seconds: float = .8
    max_center_distance: float = .2
    prediction_seconds: float = .4

    def __post_init__(self):
        if self.method not in ('iou_greedy', 'iou_hungarian', 'appearance_motion'):
            raise ValueError('Unknown association method')
        values = np.array([self.appearance_weight, self.motion_weight, self.min_cosine,
                           self.max_age_seconds, self.max_center_distance, self.prediction_seconds])
        if (not np.isfinite(values).all() or self.appearance_weight < 0
                or self.motion_weight < 0 or self.appearance_weight+self.motion_weight > 1
                or not -1 <= self.min_cosine <= 1 or self.max_age_seconds <= 0
                or self.max_center_distance <= 0 or self.prediction_seconds < 0):
            raise ValueError('Invalid association parameters')


def association_contract(value=None):
    return dict(version='roi_motion_hungarian_v1', **asdict(AssociationConfig(**dict(value or {}))))


def cosine(a, b):
    norm = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(np.dot(a, b)/norm, -1, 1)) if norm > 1e-12 else None


def assign(tracks, observations, timestamp, match_iou, config, overlap_fn):
    """Return one-to-one track/observation pairs; never receives GT or events.

    Invalid pairs cannot consume a detection. Dummy columns let every track
    remain unmatched. Motion uses the last two real observations in seconds;
    predictions are bounded and never emitted as observed boxes/features.
    """
    candidates = []
    for i, track in enumerate(tracks):
        age = timestamp-track['events'][-1][0]
        center = (track['box'][:2]+track['box'][2:])/2
        predicted = center+track.get('velocity', np.zeros(2))*min(age, config.prediction_seconds)
        for j, obs in enumerate(observations):
            if obs.class_id != track['class_id']:
                continue
            overlap = overlap_fn(obs.box, track['box'])
            if config.method != 'appearance_motion':
                if overlap >= match_iou:
                    candidates.append((overlap, i, j))
                continue
            similarity = cosine(track['events'][-1][2], np.asarray(obs.feature))
            distance = float(np.linalg.norm((obs.box[:2]+obs.box[2:])/2-predicted))
            # Reject appearance conflicts even when boxes overlap. Missing/zero
            # appearance falls back to IoU inside the same age/motion gates.
            if (age > config.max_age_seconds or distance > config.max_center_distance
                    or (similarity is not None and similarity < config.min_cosine)
                    or (overlap < match_iou and similarity is None)):
                continue
            appearance = (similarity+1)/2 if similarity is not None else overlap
            score = ((1-config.appearance_weight-config.motion_weight)*overlap
                     + config.appearance_weight*appearance
                     + config.motion_weight*(1-distance/config.max_center_distance))
            candidates.append((score, i, j))
    if config.method == 'iou_greedy':
        used_tracks, used_obs, pairs = set(), set(), []
        for _, i, j in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
            if i not in used_tracks and j not in used_obs:
                pairs.append((i, j)); used_tracks.add(i); used_obs.add(j)
        return pairs
    if not tracks or not observations:
        return []
    costs = np.full((len(tracks), len(observations)+len(tracks)), 1e6)
    costs[:, len(observations):] = 0.
    for score, i, j in candidates:
        costs[i, j] = -max(score, 1e-9)
    rows, cols = linear_sum_assignment(costs)
    return [(int(i), int(j)) for i, j in zip(rows, cols)
            if j < len(observations) and costs[i, j] < 0]
