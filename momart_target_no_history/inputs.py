"""Identical current-frame ROI construction for cached and online boxes."""
import numpy as np
import torch
import torch.nn.functional as F

from momart_boxed_no_history.painting import draw_bbox

ROI_SIZE = (64, 64)


def object_inputs(rgb, box, valid):
    """Read an unpainted CHW float RGB frame in [0,1], normalized xyxy."""
    image = np.asarray(rgb, dtype=np.float32)
    box = np.asarray(box, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3 or box.shape != (4,):
        raise ValueError("Expected CHW RGB and normalized xyxy")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("ROI source must be RGB in [0,1]")
    empty = np.zeros((3, *ROI_SIZE), np.float32)
    if not valid:
        return np.zeros(4, np.float32), empty, np.float32(0)
    if not np.isfinite(box).all():
        raise ValueError("Invalid normalized box")
    box = np.clip(box, 0, 1)
    h, w = image.shape[-2:]
    x1, y1 = np.floor(box[:2] * [w, h]).astype(int)
    x2, y2 = np.ceil(box[2:] * [w, h]).astype(int)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Valid detection has an empty ROI")
    crop = torch.from_numpy(image[:, y1:y2, x1:x2].copy()).unsqueeze(0)
    roi = F.interpolate(crop, size=ROI_SIZE, mode="bilinear", align_corners=False)
    return box.copy(), roi[0].numpy(), np.float32(1)


def replace_target_inputs(obs, boxes, valid, mode):
    """Consume frozen selector outputs for each of the two physical observations.

    No detector is called here. Online callers must associate each result with
    its real episode/sensor step before passing it, including duplicated reset
    frames. The latest ROI is always extracted before painting.
    """
    if mode not in ('box', 'box_coordinates_roi'):
        raise ValueError('Unknown target input mode')
    source = np.asarray(obs['rgb'])
    boxes = np.asarray(boxes, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if source.ndim != 5 or source.shape[2] != 3:
        raise ValueError('Expected B x T x CHW processed RGB')
    if boxes.shape != (*source.shape[:2], 4) or valid.shape != source.shape[:2]:
        raise ValueError('Every observation requires its own selector result')
    if not np.isfinite(source).all() or source.min() < 0 or source.max() > 1:
        raise ValueError('Expected RGB in [0,1]')
    chosen = boxes[valid]
    if (not np.isfinite(chosen).all() or (chosen < 0).any() or (chosen > 1).any()
            or (chosen[:, 2:] <= chosen[:, :2]).any()):
        raise ValueError('Expected valid normalized xyxy boxes')
    result = dict(obs)
    painted = source.copy()
    for b in range(len(source)):
        for t in range(source.shape[1]):
            if valid[b, t]:
                hwc = np.moveaxis(source[b, t], 0, -1)
                outline = draw_bbox(np.zeros(hwc.shape, np.uint8), boxes[b,t], True)[...,0] > 0
                np.moveaxis(painted[b,t],0,-1)[outline] = [1,0,0]
    result['rgb'] = painted
    if mode == 'box_coordinates_roi':
        objects = [object_inputs(source[b,-1], boxes[b,-1], valid[b,-1]) for b in range(len(source))]
        for key, column in zip(('_bbox','_roi','_bbox_valid'), zip(*objects)):
            result[key] = np.stack(column)
    return result
