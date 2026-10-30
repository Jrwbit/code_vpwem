"""Deploy a dev-selected RGB detector with explicit per-environment state.

Only an RGB image and stream identifier enter detect(). Masks, simulator state
and future images have no input path. Stream state must reset at episode reset.
"""
import json
from pathlib import Path
import numpy as np
from .detector import BowlDetector, select_detection
from .strategies import Config, CausalTracker, pixels
from .cache_strategy_proposals import restore_crop
from .compare_strategies import predict_stream


class PrecisionDetector:
    def __init__(self,bundle,device="cuda:0"):
        bundle=Path(bundle)
        self.bundle=json.loads(bundle.read_text())
        self.spec=self.bundle["spec"]
        self.device=device
        self.global_detector=None
        self.tiny=None
        if self.spec["method"]!="tiny":
            self.global_detector=BowlDetector(device=device,local_files_only=True)
        if self.spec["method"] in ("tiny","tiny_refine","hybrid"):
            import torch
            from .tiny_mask import TinyMask
            self.torch=torch
            checkpoint=torch.load(bundle.parent/self.bundle["weights"],map_location="cpu",weights_only=False)
            self.tiny=TinyMask().to(device).eval()
            self.tiny.load_state_dict(checkpoint["model"])
        self.states={}
        self.metadata=dict(source="rgb_only",spec=self.spec,stream_state="explicit_episode_reset")

    def reset(self,stream_id=0):
        self.states.pop(stream_id,None)

    def detect(self,image,stream_id=0):
        image=np.asarray(image)
        if image.shape!=(90,120,3) or image.dtype!=np.uint8:
            raise ValueError("Expected original 90x120 uint8 RGB")
        if stream_id not in self.states:
            self.states[stream_id]=dict(previous=None,age=0,
                tracker=CausalTracker(Config(**self.spec["track"])) if self.spec["method"]=="track" else None)
        state=self.states[stream_id]
        boxes=np.empty((0,4),np.float32);scores=np.empty(0,np.float32)
        roi_boxes=np.empty((0,4),np.float32);roi_scores=np.empty(0,np.float32)
        if self.global_detector is not None:
            result=self.global_detector.proposals([image],threshold=.10)[0]
            boxes=result["boxes"]/np.array([120,90,120,90],np.float32)
            scores=result["scores"]
            strong=select_detection(boxes,scores,1,1,.25)
            state["raw"]=strong
            if (self.spec["method"]=="roi" or self.spec.get("use_roi")) and not strong.valid and state["previous"] is not None and state["age"]<3:
                b=state["previous"]
                pad=max(8,int(max((b[2:]-b[:2])*[120,90])))
                crop=pixels(b,image.shape,pad)
                x1,y1,x2,y2=crop
                if x2-x1>=4 and y2-y1>=4:
                    result=self.global_detector.proposals([image[y1:y2,x1:x2]],threshold=.10)[0]
                    roi_boxes=restore_crop(result["boxes"],crop,image.shape);roi_scores=result["scores"]
            if strong.valid:state["previous"]=strong.box.copy();state["age"]=0
            else:state["age"]+=1
        probability=None
        if self.tiny is not None:
            torch=self.torch
            batch=torch.from_numpy(image).permute(2,0,1).float().unsqueeze(0).to(self.device)/255
            with torch.inference_mode():probability=self.tiny(batch).sigmoid().cpu().numpy()[:,0]
        prop=dict(boxes=boxes.tolist(),scores=scores.tolist(),roi_boxes=roi_boxes.tolist(),roi_scores=roi_scores.tolist())
        if state["tracker"] is not None:
            if self.spec.get("use_roi"):
                boxes=np.concatenate([boxes,roi_boxes]);scores=np.concatenate([scores,roi_scores])
            return state["tracker"].step(image,boxes,scores)[0]
        return predict_stream(self.spec,[image],[prop],probability)[0][0]


class PrecisionSequenceDetector:
    """Batch facade that advances the stateful detector once per unique frame.

    Boxed policy observations contain overlapping recent-frame stacks. Reusing
    the same RGB frame must reuse its detection rather than advancing tracker
    state a second time. ``reset_all`` is called between evaluation episodes.
    """
    def __init__(self, bundle, device="cuda:0", num_streams=1):
        self.detector = PrecisionDetector(bundle, device=device)
        self.num_streams = int(num_streams)
        self.cache = [None for _ in range(self.num_streams)]
        self.processed_frames = 0
        self.valid_frames = 0
        self.reset_count = 0

    def reset_all(self):
        for stream_id in range(self.num_streams):
            self.detector.reset(stream_id)
            self.cache[stream_id] = None
        self.reset_count += self.num_streams

    def __call__(self, images):
        if len(images) % self.num_streams:
            raise ValueError("Frame batch must contain a whole number of streams")
        frames_per_stream = len(images) // self.num_streams
        detections = []
        for stream_id in range(self.num_streams):
            previous = self.cache[stream_id]
            for frame_index in range(frames_per_stream):
                image = np.asarray(images[stream_id * frames_per_stream + frame_index])
                key = image.tobytes()
                # Only frame 0 can overlap the preceding observation stack.
                # Repeated-looking images later in this stack still represent
                # later timesteps and must advance tracker state.
                detection = (previous[1] if frame_index == 0 and
                             previous is not None and key == previous[0] else None)
                if detection is None:
                    detection = self.detector.detect(image, stream_id=stream_id)
                    self.processed_frames += 1
                    self.valid_frames += int(detection.valid)
                detections.append(detection)
            last_image = np.asarray(images[(stream_id + 1) * frames_per_stream - 1])
            self.cache[stream_id] = (
                last_image.tobytes(), detections[(stream_id + 1) * frames_per_stream - 1])
        return detections
