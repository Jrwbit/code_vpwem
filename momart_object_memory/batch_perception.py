"""Batched execution of the frozen Mask protocol without changing its source hashes."""
import numpy as np
import torch
from .mask_detector import select_boxes
from .perception import bounded_detections, raw_roi_features
from .state import ObjectObservation


@torch.inference_mode()
def perceive_batch(perceive, images):
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (90, 120, 3) or images.dtype != np.uint8:
        raise ValueError('Expected batch of original 90x120 uint8 RGB')
    if perceive.backend != 'mask_rcnn':
        return [perceive(image) for image in images]
    if not len(images):
        return []
    rgb = torch.from_numpy(images.copy()).to(perceive.device).permute(0, 3, 1, 2).float()
    predictions = perceive.detector.model(list(rgb / 255))
    detections = [select_boxes(image, result, perceive.detector.thresholds)
                  for image, result in zip(images, predictions)]
    del predictions
    if perceive.max_detections is not None:
        detections = [bounded_detections(items, int(perceive.max_detections)) for items in detections]
    grids = perceive.visual(rgb / 127.5 - 1)
    observations = []
    for index, items in enumerate(detections):
        if not items:
            observations.append([])
            continue
        boxes = torch.tensor(np.stack([item[1] for item in items]), device=perceive.device)
        features = raw_roi_features(grids[index:index+1], boxes).cpu().numpy()
        observations.append([ObjectObservation(cls, box, score, feature)
                             for (cls, box, score), feature in zip(items, features)])
    return observations
