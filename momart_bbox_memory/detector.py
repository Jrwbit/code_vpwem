"""Shared RGB-only detector for offline annotation and online rollout.

No simulator states, trajectory completion flags, or future frames are accepted.
A missing detection is missing evidence, never a full-image substitute.
"""
from dataclasses import dataclass
import numpy as np

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"

@dataclass(frozen=True)
class Detection:
    box: np.ndarray  # normalized xyxy in the ORIGINAL RGB frame
    score: float
    valid: bool
    ambiguous: bool
    candidates: int


def select_detection(boxes, scores, width, height, threshold=0.25):
    """Clip, filter, and select highest-confidence box; report ambiguity."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) != len(scores) or width <= 0 or height <= 0:
        raise ValueError("Invalid detector shapes or frame dimensions")
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    good = (np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
            & (scores >= threshold) & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1]))
    boxes, scores = boxes[good], scores[good]
    if len(scores) == 0:
        return Detection(np.zeros(4, np.float32), 0.0, False, False, 0)
    # Deduplicate overlapping proposals; distinct bowls remain ambiguous.
    order = np.argsort(-scores, kind="stable")
    keep = []
    for idx in order:
        box = boxes[idx]
        duplicate = False
        for j in keep:
            inter = np.maximum(0, np.minimum(box[2:], boxes[j, 2:])
                               - np.maximum(box[:2], boxes[j, :2])).prod()
            union = ((box[2:] - box[:2]).prod()
                     + (boxes[j, 2:] - boxes[j, :2]).prod() - inter)
            if inter / max(float(union), 1e-8) > 0.5:
                duplicate = True
                break
        if not duplicate:
            keep.append(int(idx))
    best = keep[0]
    normalized = boxes[best] / np.array([width, height, width, height], np.float32)
    return Detection(normalized, float(scores[best]), True, len(keep) > 1, len(keep))


def red_support_filter(image, boxes, min_fraction=0.10, min_pixels=3):
    """Benchmark-specific RGB evidence gate for the red YCB bowl.

    This is an explicit color prior, not a learned visibility probability.
    It rejects hallucinated background boxes; no future/state labels are used.
    """
    image=np.asarray(image)
    rgb=image.astype(np.float32)
    red=((rgb[...,0] >= 35) & (rgb[...,0] > 1.5*rgb[...,1])
         & (rgb[...,0] > 1.5*rgb[...,2])
         & (np.abs(rgb[...,1]-rgb[...,2]) <= 0.20 *
            (rgb[...,0]-np.minimum(rgb[...,1],rgb[...,2]))))
    h,w=image.shape[:2]
    keep=[]
    for box in np.asarray(boxes).reshape(-1,4):
        if not np.isfinite(box).all():
            keep.append(False); continue
        x1,y1=np.floor(box[:2]).astype(int)
        x2,y2=np.ceil(box[2:]).astype(int)
        x1,x2=np.clip([x1,x2],0,w);y1,y2=np.clip([y1,y2],0,h)
        region=red[y1:y2,x1:x2]
        keep.append(region.size>0 and region.sum()>=min_pixels
                    and float(region.mean())>=min_fraction)
    return np.asarray(keep,dtype=bool)


class BowlDetector:
    def __init__(self, device="cuda:0", threshold=0.25, text_threshold=0.25,
                 shortest_edge=600, model_id=MODEL_ID, revision=REVISION, prompt="a red bowl.",
                 local_files_only=False):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.prompt = str(prompt)
        self.torch = torch
        self.device = device
        self.threshold = float(threshold)
        self.text_threshold = float(text_threshold)
        self.shortest_edge = int(shortest_edge)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision,
                                                       local_files_only=local_files_only)
        self.processor.image_processor.size = {"shortest_edge": self.shortest_edge, "longest_edge": 1333}
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, revision=revision,local_files_only=local_files_only).to(device).eval().requires_grad_(False)
        self.metadata = dict(model_id=model_id, revision=revision, prompt=self.prompt,
                             threshold=self.threshold, text_threshold=self.text_threshold,
                             shortest_edge=self.shortest_edge, longest_edge=1333,
                             box_format="normalized_xyxy", source="rgb_only",
                             red_support_min_fraction=0.10, red_support_min_pixels=3,
                             red_hue_chroma_ratio_max=0.20)

    def proposals(self, images, threshold=None):
        """Return RGB-supported pixel proposals, retaining low-score candidates."""
        from PIL import Image
        import inspect
        if not images:
            return []
        for im in images:
            if im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3:
                raise ValueError("Expected uint8 HWC RGB")
        pil = [Image.fromarray(im) for im in images]
        inputs = self.processor(images=pil, text=[self.prompt] * len(pil),
                                padding=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        post = self.processor.post_process_grounded_object_detection
        threshold_key = "box_threshold" if "box_threshold" in inspect.signature(post).parameters else "threshold"
        results = post(outputs, inputs.input_ids,
                       **{threshold_key: self.threshold if threshold is None else float(threshold)},
                       text_threshold=self.text_threshold,
                       target_sizes=[(im.shape[0], im.shape[1]) for im in images])
        selected=[]
        for im,r in zip(images,results):
            boxes=r["boxes"].detach().cpu().numpy()
            scores=r["scores"].detach().cpu().numpy()
            keep=red_support_filter(im,boxes)
            selected.append(dict(boxes=boxes[keep], scores=scores[keep]))
        return selected

    def __call__(self, images):
        return [select_detection(r["boxes"], r["scores"], im.shape[1], im.shape[0], self.threshold)
                for im, r in zip(images, self.proposals(images))]


class QualityGatedDetector:
    """Apply the exact offline cache rule without inventing missing boxes."""
    def __init__(self, detector, threshold=0.35):
        from .quality import VERSION
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.detector = detector
        self.threshold = float(threshold)
        self.metadata = dict(detector.metadata, quality_gate=dict(
            version=VERSION, threshold=self.threshold))

    def filter(self, detections):
        from .quality import quality_gate
        detections = list(detections)
        if not detections:
            return []
        accepted = quality_gate(
            [d.box for d in detections], [d.valid for d in detections],
            [d.ambiguous for d in detections], [d.score for d in detections],
            self.threshold)
        return [Detection(d.box.copy() if keep else np.zeros(4, np.float32),
                          d.score, bool(keep), d.ambiguous, d.candidates)
                for d, keep in zip(detections, accepted)]

    def __call__(self, images):
        return self.filter(self.detector(images))
