"""Production frozen Mask R-CNN; matches the archived detector pilot adapter."""
import numpy as np
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import nms
from torchvision.ops.misc import FrozenBatchNorm2d
from momart_bbox_memory.detector import red_support_filter

CLASSES = ['a red bowl.', 'a dishwasher door.', 'a dishwasher door handle.',
           'a dishwasher rack.', 'a drawer.', 'a drawer handle.']


def make_model():
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None,
                                  trainable_layers=3, norm_layer=FrozenBatchNorm2d)
    model = MaskRCNN(backbone, num_classes=7, min_size=320, max_size=448,
                    box_score_thresh=.01, box_nms_thresh=.5, box_detections_per_img=100)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 7)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 7)
    return model


def select_boxes(image, result, thresholds):
    """Six classes, pixel clipping, class-0 red support and per-class NMS.

    Matches diagnostic sanitize: threshold before clipping/NMS, then globally
    retain the strongest 100. The memory front end subsequently balances slots.
    """
    boxes = result['boxes'].detach().float().cpu().numpy()
    scores = result['scores'].detach().float().cpu().numpy()
    classes = result['labels'].detach().cpu().numpy()-1
    detections = []
    for cls in range(6):
        indices = np.flatnonzero((classes == cls) & np.isfinite(scores)
                                 & (scores >= max(.01, thresholds[cls])))
        clipped = np.clip(boxes[indices], [0,0,0,0], [120,90,120,90]).astype(np.float32)
        good = np.isfinite(clipped).all(1) & np.all(clipped[:, 2:] > clipped[:, :2], axis=1)
        if cls == 0:
            good &= red_support_filter(image, clipped)
        keep = np.flatnonzero(good)
        if not len(keep):
            continue
        chosen = nms(torch.from_numpy(clipped[keep]), torch.from_numpy(scores[indices[keep]]), .5)
        for k in chosen.tolist():
            j = keep[k]
            detections.append((cls, clipped[j]/np.array([120,90,120,90], np.float32),
                               float(scores[indices[j]])))
    return sorted(detections, key=lambda o: -o[2])[:100]


class FrozenMaskDetector:
    def __init__(self, checkpoint, thresholds, device):
        self.device = device
        self.thresholds = list(thresholds)
        self.model = make_model()
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state['model'], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.model.roi_heads.score_thresh = min(self.thresholds)

    @torch.inference_mode()
    def __call__(self, image):
        tensor = torch.from_numpy(image.copy()).to(self.device).permute(2,0,1).float()/255
        return select_boxes(image, self.model([tensor])[0], self.thresholds)
