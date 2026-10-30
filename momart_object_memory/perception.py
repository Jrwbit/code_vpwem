"""Frozen multi-class GroundingDINO + raw-image layer2 ROIAlign features."""
import hashlib
import inspect
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.ops import roi_align
from cleandiffuser.nn_condition.multi_image_condition import get_resnet, replace_submodules
from momart_bbox_memory.detector import MODEL_ID, REVISION, red_support_filter
from .state import ObjectObservation, iou


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_roi_features(grid, normalized_boxes):
    """Map original 120x90 pixels to stride-8 layer2, not padded grid extents."""
    boxes = normalized_boxes * normalized_boxes.new_tensor([120/8, 90/8, 120/8, 90/8])
    return roi_align(grid, [boxes], output_size=3, spatial_scale=1.,
                     sampling_ratio=2, aligned=True).mean((-1, -2))


def bounded_detections(detections, limit):
    """Keep strongest instance per class, then strongest extras; causal only."""
    if limit < 1:
        raise ValueError('Positive detection limit required')
    order = sorted(range(len(detections)), key=lambda i: (-detections[i][2], detections[i][0],
                                                         *detections[i][1].tolist()))
    selected, classes = [], set()
    for i in order:
        if detections[i][0] not in classes:
            selected.append(i); classes.add(detections[i][0])
    selected = selected[:limit]
    for i in order:
        if len(selected) >= limit: break
        if i not in selected: selected.append(i)
    return [detections[i] for i in selected]


class FrozenPerception:
    """One frozen model, per-class prompts, deterministic per-class NMS.

    Red RGB support applies only to class 0 (red bowl); handles/racks must not
    inherit the old red-bowl color gate. Association happens in ObjectBank.
    """
    def __init__(self, config, device='cpu'):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.device = device
        self.classes = list(config.classes)
        if not self.classes or self.classes[0] != 'a red bowl.':
            raise ValueError('Class 0 must remain the baseline red bowl prompt')
        self.threshold = float(config.threshold)
        self.max_detections = config.get("max_detections", None)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION,
                                                       local_files_only=True)
        self.processor.image_processor.size = {'shortest_edge': 600, 'longest_edge': 1333}
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID, revision=REVISION, local_files_only=True).to(device).eval().requires_grad_(False)
        resnet = get_resnet('resnet18')
        resnet = replace_submodules(resnet, lambda x: isinstance(x, nn.BatchNorm2d),
                                    lambda x: nn.GroupNorm(x.num_features//16, x.num_features))
        checkpoint = torch.load(config.feature_checkpoint, map_location='cpu', weights_only=False)
        prefix = 'condition.encoder.key_model_map.rgb.'
        state = {k[len(prefix):]: v for k, v in checkpoint['model_ema'].items()
                 if k.startswith(prefix)}
        resnet.load_state_dict(state, strict=True)
        self.visual = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                    resnet.layer1, resnet.layer2).to(device).eval().requires_grad_(False)
        if int(config.feature_dim) != 128:
            raise ValueError('ResNet18 layer2 ROI features have dimension 128')
        self.metadata = dict(version='gdino_raw_layer2_roi_v1', classes=self.classes,
                             max_detections=self.max_detections, selection='class_balanced_confidence',
                             detector_model=MODEL_ID, detector_revision=REVISION,
                             threshold=self.threshold, text_threshold=0.25,
                             shortest_edge=600, longest_edge=1333, nms_iou=0.5,
                             feature_checkpoint_sha256=file_sha256(config.feature_checkpoint),
                             feature_weights='model_ema.condition.encoder.key_model_map.rgb',
                             feature_dim=128, feature_layer='layer2', input='raw90x120_rgb_minus1_plus1',
                             roi_align=dict(output_size=3, aligned=True, sampling_ratio=2, feature_stride=8),
                             implementation_sha256=file_sha256(__file__),
                             red_support_class=0)

    @torch.inference_mode()
    def __call__(self, image):
        from PIL import Image
        image = np.asarray(image)
        if image.shape != (90, 120, 3) or image.dtype != np.uint8:
            raise ValueError('Expected original 90x120 uint8 RGB')
        inputs = self.processor(images=[Image.fromarray(image)]*len(self.classes),
                                text=self.classes, padding=True, return_tensors='pt').to(self.device)
        outputs = self.detector(**inputs)
        post = self.processor.post_process_grounded_object_detection
        key = 'box_threshold' if 'box_threshold' in inspect.signature(post).parameters else 'threshold'
        results = post(outputs, inputs.input_ids, **{key: self.threshold}, text_threshold=0.25,
                       target_sizes=[image.shape[:2]]*len(self.classes))
        detections = []
        for cls, result in enumerate(results):
            boxes = result['boxes'].float().cpu().numpy()
            scores = result['scores'].float().cpu().numpy()
            if cls == 0:
                keep = red_support_filter(image, boxes)
                boxes, scores = boxes[keep], scores[keep]
            boxes = np.clip(boxes / [120, 90, 120, 90], 0, 1).astype(np.float32)
            order = np.argsort(-scores, kind='stable')
            kept = []
            for j in order:
                if (np.isfinite(boxes[j]).all() and np.isfinite(scores[j])
                        and np.all(boxes[j, 2:] > boxes[j, :2])
                        and all(iou(boxes[j], boxes[k]) <= 0.5 for k in kept)):
                    kept.append(j)
                    detections.append((cls, boxes[j], float(scores[j])))
        if self.max_detections is not None:
            detections = bounded_detections(detections, int(self.max_detections))
        if not detections:
            return []
        rgb = torch.from_numpy(image.copy()).to(self.device).permute(2, 0, 1)[None].float()/127.5-1
        grid = self.visual(rgb)
        # Original pixels / stride, including the padded non-multiple-of-8 height.
        # Scaling by grid.shape would incorrectly stretch 90 pixels to 96.
        boxes = torch.tensor(np.stack([d[1] for d in detections]), device=self.device)
        rois = raw_roi_features(grid, boxes).cpu().numpy()
        return [ObjectObservation(cls, box, score, feature)
                for (cls, box, score), feature in zip(detections, rois)]
