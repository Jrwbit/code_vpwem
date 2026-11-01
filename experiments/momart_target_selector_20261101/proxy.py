"""Offline supervision candidates. Future events must never be policy inputs."""
import numpy as np


def motion_intervals(values, threshold, minimum_range, max_gap=4):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    changes = np.flatnonzero(np.linalg.norm(np.diff(values, axis=0), axis=1) >= threshold) + 1
    groups = []
    for step in changes:
        if not groups or step - groups[-1][-1] > max_gap:
            groups.append([int(step)])
        else:
            groups[-1].append(int(step))
    return [(group[0], group[-1]) for group in groups
            if np.linalg.norm(np.ptp(values[max(0,group[0]-1):group[-1]+1],axis=0)) >= minimum_range]


def target_at(events, step, lookahead=40):
    """Current moving object, else next proximity-supported motion within 2s.

    Ambiguous simultaneous objects and unlabelled periods are excluded rather
    than converted to false negative examples. This is not ground-truth intent.
    """
    active = [event for event in events if event['onset'] <= step <= event['end']]
    if active:
        identities = {event['instance_id'] for event in active}
        return active[0] if len(identities) == 1 else None
    upcoming = [event for event in events if step < event['onset'] <= step+lookahead]
    if not upcoming:
        return None
    onset = min(event['onset'] for event in upcoming)
    first = [event for event in upcoming if event['onset'] == onset]
    return first[0] if len({event['instance_id'] for event in first}) == 1 else None


def target_candidate(objects, truth, target, minimum_iou=.5):
    """Scoring/label generation only: locate a matching detector output."""
    if target is None:
        return None
    gt = next((obj for obj in truth if obj['instance_id'] == target['instance_id']), None)
    if gt is None:
        return -1
    best, index = 0., -1
    a = np.asarray(gt['box'], dtype=float)
    for i,obj in enumerate(objects):
        if obj['class_id'] != gt['class_id']:
            continue
        b = np.asarray(obj['box'], dtype=float)
        intersection = np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
        union = np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-intersection
        overlap = intersection/max(union,1e-8)
        if overlap >= minimum_iou and overlap > best:
            best,index = overlap,i
    return index
