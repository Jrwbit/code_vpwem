"""Lean raw-sensor frontend for spatial/mask-spatial models, with no ROI encoder."""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import torch
from robot_geometry import RobotGeometry,box_geometry
from mask_surface import surface_features,overlap
from runtime import FrameFeatures
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from momart_object_memory.mask_detector import FrozenMaskDetector,select_boxes,CLASSES
from momart_object_memory.perception import bounded_detections,file_sha256,FrozenPerception,raw_roi_features


@dataclass
class RobotObservation:
    rgb: np.ndarray
    depth: np.ndarray
    proprio: np.ndarray
    torso: float
    episode_id: str
    step: int
    task: str


class SpatialFrontEnd:
    def __init__(self,config,urdf,tasks,device='cpu',fov=45.,depth_high=3.,robot_scale=1.,include_visual=False):
        if list(config.classes)!=CLASSES or int(config.max_detections)!=8:raise ValueError('Expected frozen six-class, eight-slot Mask configuration')
        if file_sha256(config.detector_checkpoint)!=config.detector_checkpoint_sha256:raise ValueError('Detector checkpoint checksum differs')
        self.tasks=list(tasks);self.device=device;self.fov=fov;self.depth_high=depth_high
        if len(self.tasks)!=4:raise ValueError('This validated context has four task entries')
        self.thresholds=list(config.thresholds);self.visual=None
        if include_visual:
            perception=FrozenPerception(config,device);self.detector=perception.detector;self.visual=perception.visual
        else:self.detector=FrozenMaskDetector(config.detector_checkpoint,self.thresholds,device)
        self.geometry=RobotGeometry(urdf,robot_scale)

    def close(self):self.geometry.close()

    @staticmethod
    def validate_pair(current,past):
        if current.episode_id!=past.episode_id or current.task!=past.task:raise ValueError('Current/past observations cross episode or task')
        if current.step<0 or past.step!=max(0,current.step-8):raise ValueError('Expected causal observation exactly eight sensor steps earlier')
        if current.step==0 and (not np.array_equal(current.proprio,past.proprio) or current.torso!=past.torso):raise ValueError('Step zero must reuse the same robot observation')
        if current.proprio.shape!=(10,) or past.proprio.shape!=(10,):raise ValueError('Expected raw 10D robot proprioception')
        if current.rgb.shape!=(90,120,3) or current.rgb.dtype!=np.uint8:raise ValueError('Expected current raw 90x120 uint8 RGB')

    @torch.inference_mode()
    def build(self,current,past):
        self.validate_pair(current,past)
        if current.task not in self.tasks:raise ValueError('Task outside validated vocabulary')
        image=current.rgb;tensor=torch.from_numpy(image.copy()).permute(2,0,1).float().to(self.device)/255
        prediction=self.detector.model([tensor])[0];items=bounded_detections(select_boxes(image,prediction,self.thresholds),8)
        boxes=prediction['boxes'].detach().cpu().numpy()/[120,90,120,90];classes=prediction['labels'].detach().cpu().numpy()-1;masks=prediction['masks'][:,0].detach().cpu().numpy()
        ee,(uv,forward)=self.geometry.project(current.proprio,current.torso,self.fov)
        old_ee,(old_uv,_)=self.geometry.project(past.proprio,past.torso,self.fov)
        geometry_context=np.r_[ee,uv,current.torso,float(forward),self.fov/90,self.depth_high,ee-old_ee,uv-old_uv].astype(np.float32)
        candidates=np.zeros((8,139),np.float32);valid=np.zeros(8,bool);geometry=np.zeros((8,16),np.float32);mask_features=np.zeros((8,23),np.float32)
        for slot,(cls,box,score) in enumerate(items):
            candidates[slot,128:]=np.r_[box,score,np.eye(6)[cls]];valid[slot]=True
            geometry[slot]=box_geometry(box,current.depth,ee,self.fov,self.depth_high)
            same=np.flatnonzero(classes==cls)
            if len(same):
                iou=overlap(box,boxes[same]);best=int(iou.argmax())
                if iou[best]>=.9:mask_features[slot]=surface_features(box,current.depth,masks[same[best]],ee,uv,self.fov,self.depth_high)
        global_features=np.zeros(256,np.float32)
        if self.visual is not None:
            if past.rgb is None or past.rgb.shape!=(90,120,3) or past.rgb.dtype!=np.uint8:raise ValueError('Visual model needs causal past RGB')
            images=torch.from_numpy(np.stack([current.rgb,past.rgb])).permute(0,3,1,2).float().to(self.device)
            grids=self.visual(images/127.5-1);global_features=grids.mean((-1,-2)).cpu().numpy().reshape(-1)
            if items:
                boxes_tensor=torch.tensor(np.stack([item[1] for item in items]),device=self.device)
                candidates[:len(items),:128]=raw_roi_features(grids[:1],boxes_tensor).cpu().numpy()
        context=np.r_[global_features,current.proprio,current.proprio-past.proprio,np.eye(4)[self.tasks.index(current.task)]].astype(np.float32)
        return FrameFeatures(candidates,context,valid,current.task,geometry_candidates=geometry,geometry_context=geometry_context,
            mask_candidates=mask_features,visual_features_available=self.visual is not None)
