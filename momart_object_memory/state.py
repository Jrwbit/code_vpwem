"""Causal instance association and fixed-capacity memory, shared offline/online."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ObjectObservation:
    class_id: int
    box: np.ndarray
    score: float
    feature: np.ndarray


def iou(a, b):
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - inter
    return float(inter / max(float(union), 1e-8))


class ObjectBank:
    """Conservative IoU association; unmatched detections create new tracks.

    IDs are local to this episode. No appearance re-identification or motion
    prediction is claimed. Capacity defaults to an error; the explicit bounded
    policy evicts the oldest currently unmatched slot and clears its history.
    Age is measured in sensor seconds.
    """
    def __init__(self, slots=8, history=3, feature_dim=128, match_iou=0.3, overflow="error"):
        if min(slots, history, feature_dim) < 1 or not 0 < match_iou <= 1:
            raise ValueError("Invalid bank capacity or IoU threshold")
        self.slots, self.history, self.feature_dim = slots, history, feature_dim
        if overflow not in ("error", "evict_oldest_invisible"):
            raise ValueError("Unknown overflow policy")
        self.overflow = overflow
        self.match_iou = match_iou
        self.reset()

    def reset(self):
        self.tracks = []
        self.last_step = -1
        self.last_time = -1.

    def update(self, sensor_step, timestamp, observations):
        if sensor_step <= self.last_step or timestamp < 0 or timestamp <= self.last_time:
            raise ValueError("Bank requires strictly increasing unique sensor observations")
        observations = list(observations)
        for obs in observations:
            box = np.asarray(obs.box)
            feature = np.asarray(obs.feature)
            if (obs.class_id < 0 or box.shape != (4,) or feature.shape != (self.feature_dim,)
                    or not np.isfinite(box).all() or not np.isfinite(feature).all()
                    or np.any(box < 0) or np.any(box > 1) or np.any(box[2:] <= box[:2])
                    or not np.isfinite(obs.score) or not 0 <= obs.score <= 1):
                raise ValueError("Invalid object observation")
        # Global deterministic greedy assignment, rather than detection-order IDs.
        candidates = []
        for j, obs in enumerate(observations):
            for i, track in enumerate(self.tracks):
                overlap = iou(obs.box, track['box'])
                if obs.class_id == track['class_id'] and overlap >= self.match_iou:
                    candidates.append((-overlap, i, j))
        used_tracks, used_obs, assignments = set(), set(), []
        for _, i, j in sorted(candidates):
            if i not in used_tracks and j not in used_obs:
                used_tracks.add(i); used_obs.add(j); assignments.append((i, j))
        new = sorted((j for j in range(len(observations)) if j not in used_obs),
                     key=lambda j: (observations[j].class_id, *observations[j].box.tolist(),
                                    -observations[j].score))
        replace = []
        needed = max(0, len(self.tracks) + len(new) - self.slots)
        if needed:
            if self.overflow == 'evict_oldest_invisible':
                replace = sorted((i for i in range(len(self.tracks)) if i not in used_tracks),
                                 key=lambda i: (self.tracks[i]['events'][-1][0], i))[:needed]
            if len(replace) != needed:
                raise ValueError('Object bank capacity exceeded; audit cumulative instance count')
        for track in self.tracks:
            track['visible'] = False
        for j in new:
            obs = observations[j]
            entry = dict(class_id=obs.class_id, box=np.asarray(obs.box).copy(),
                         events=[], visible=False, score=0.)
            if replace:
                slot = replace.pop(0)
                self.tracks[slot] = entry
            else:
                slot = len(self.tracks)
                self.tracks.append(entry)
            assignments.append((slot, j))
        for i, j in assignments:
            track, obs = self.tracks[i], observations[j]
            track.update(box=np.asarray(obs.box, np.float32).copy(), visible=True,
                         score=float(obs.score))
            track['events'].append((timestamp, sensor_step, np.asarray(obs.feature, np.float32).copy()))
            track['events'] = track['events'][-self.history:]
        self.last_step, self.last_time = int(sensor_step), float(timestamp)

    def snapshot(self, mode):
        if mode not in ('a1', 'a2', 'a3', 'a3_latest_control'):
            raise ValueError("Expected a1, a2, a3 or a3_latest_control")
        n, h, d = self.slots, self.history, self.feature_dim
        out = dict(features=np.zeros((n, h, d), np.float32),
                   event_valid=np.zeros((n, h), bool), ages=np.zeros((n, h), np.float32),
                   boxes=np.zeros((n, 4), np.float32), visible=np.zeros(n, bool),
                   valid=np.zeros(n, bool), classes=np.zeros(n, np.int64),
                   confidence=np.zeros(n, np.float32))
        for i, track in enumerate(self.tracks):
            if mode == 'a1' and not track['visible']:
                continue
            events = track['events'] if mode == 'a3' else track['events'][-1:]
            if mode == 'a3_latest_control':
                events = track['events'][-1:] * self.history
            out['valid'][i] = True
            out['visible'][i] = track['visible']
            out['classes'][i] = track['class_id']
            out['confidence'][i] = track['score']
            if track['visible']:
                out['boxes'][i] = track['box']
            for j, (time, _, feature) in enumerate(events, start=h-len(events)):
                out['features'][i, j] = feature
                out['event_valid'][i, j] = True
                out['ages'][i, j] = self.last_time - time
        return out


def query_schedule(anchor, action_steps, obs_steps=2):
    """Deployment-cadence prefix ending at any training anchor.

    Each phase starts with reset frame 0; phase 0 matches actual rollout queries
    0,A,2A,... exactly. Other phases are explicit training cadence augmentation.
    """
    if anchor < 0 or action_steps < 1 or obs_steps != 2:
        raise ValueError("Only two-observation PTP streams are supported")
    phase = anchor % action_steps
    queries = [0] + list(range(phase or action_steps, anchor+1, action_steps))
    return [[max(0, t-1), t] for t in queries]


class ObservationStream:
    """Track each env by sensor step, never deduplicate by pixel equality."""
    def __init__(self, bank, perceive):
        self.bank, self.perceive = bank, perceive
        self.reset()

    def reset(self):
        self.bank.reset()
        self.cache = {}

    def observe(self, images, steps, timestamps):
        if not (len(images) == len(steps) == len(timestamps)):
            raise ValueError('Observation stream length mismatch')
        result = []
        for image, step, timestamp in zip(images, steps, timestamps):
            step, timestamp = int(step), float(timestamp)
            if step not in self.cache:
                observations = self.perceive(image)
                self.bank.update(step, timestamp, observations)
                self.cache[step] = (timestamp, observations)
            elif self.cache[step][0] != timestamp:
                raise ValueError("A sensor step cannot have two timestamps")
            result.append(self.cache[step][1])
        # Only the recent stack can overlap; bound online cache independently of bank.
        keep = set(map(int, steps))
        self.cache = {k: v for k, v in self.cache.items() if k in keep}
        return result
