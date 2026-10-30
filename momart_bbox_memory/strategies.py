"""Causal RGB-only proposal association and verified local tracking.

State is per stream, reset explicitly at episode/segment boundaries. Invalid
frames never reuse an unverified box. Scores are evidence, not probabilities.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from .detector import Detection, select_detection


def invalid():
    return Detection(np.zeros(4,np.float32),0.,False,False,0)


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter
    return float(inter/max(float(union),1e-12))


def red_mask(image):
    r,g,b=np.moveaxis(image.astype(np.float32),-1,0)
    return ((r>=35)&(r>1.5*g)&(r>1.5*b)&(np.abs(g-b)<=.2*(r-np.minimum(g,b))))


def pixels(box, shape, pad=0):
    h,w=shape[:2]
    b=np.asarray(box)*[w,h,w,h]
    return (max(0,int(np.floor(b[0]))-pad),max(0,int(np.floor(b[1]))-pad),
            min(w,int(np.ceil(b[2]))+pad),min(h,int(np.ceil(b[3]))+pad))


def color_refine(image, box, max_shrink=.25):
    """Use current RGB evidence inside a proposal; no reference labels."""
    x1,y1,x2,y2=pixels(box,image.shape,pad=1)
    region=red_mask(image)[y1:y2,x1:x2]
    ys,xs=np.where(region)
    if len(xs)<3:return box
    h,w=image.shape[:2]
    refined=np.array([(xs.min()+x1)/w,(ys.min()+y1)/h,
                      (xs.max()+x1+1)/w,(ys.max()+y1+1)/h],np.float32)
    if iou(refined,box)<max_shrink:return box
    return refined


@dataclass(frozen=True)
class Config:
    method: str = "association"
    init_threshold: float = .25
    low_threshold: float = .15
    association_iou: float = .15
    max_age: int = 3
    template_threshold: float = .85
    refine: bool = False
    refine_alpha: float = 1.0


class CausalTracker:
    def __init__(self,config=Config()):
        self.config=config
        self.reset()

    def reset(self):
        self.box=None;self.previous=None;self.patch=None;self.age=0

    def _local(self,image):
        if self.box is None or self.previous is None or self.age>=self.config.max_age:
            return None
        h,w=image.shape[:2]
        old=np.asarray(self.box)
        if self.config.method=="flow":
            x1,y1,x2,y2=pixels(old,image.shape,pad=1)
            mask=np.zeros((h,w),np.uint8);mask[y1:y2,x1:x2]=255
            gray=cv2.cvtColor(self.previous,cv2.COLOR_RGB2GRAY)
            curr=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            pts=cv2.goodFeaturesToTrack(gray,20,.01,2,mask=mask)
            if pts is None or len(pts)<3:return None
            next_pts,ok,_=cv2.calcOpticalFlowPyrLK(gray,curr,pts,None,winSize=(9,9),maxLevel=2)
            if next_pts is None:return None
            back,ok2,_=cv2.calcOpticalFlowPyrLK(curr,gray,next_pts,None,winSize=(9,9),maxLevel=2)
            if back is None:return None
            good=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm((back-pts).reshape(-1,2),axis=1)<1.)
            if good.sum()<3:return None
            delta=np.median((next_pts-pts).reshape(-1,2)[good],axis=0)/[w,h]
            candidate=np.clip(old+np.tile(delta,2),0,1)
        else:
            if self.patch is None or min(self.patch.shape[:2])<3:return None
            x1,y1,x2,y2=pixels(old,image.shape,pad=max(6,int(max((old[2:]-old[:2])*[w,h]))))
            region=image[y1:y2,x1:x2]
            ph,pw=self.patch.shape[:2]
            if region.shape[0]<ph or region.shape[1]<pw:return None
            response=cv2.matchTemplate(region,self.patch,cv2.TM_CCOEFF_NORMED)
            _,value,_,loc=cv2.minMaxLoc(response)
            if not np.isfinite(value) or value<self.config.template_threshold:return None
            xx,yy=loc[0]+x1,loc[1]+y1
            candidate=np.array([xx/w,yy/h,(xx+pw)/w,(yy+ph)/h],np.float32)
        x1,y1,x2,y2=pixels(candidate,image.shape)
        evidence=red_mask(image)[y1:y2,x1:x2]
        if evidence.size==0 or evidence.sum()<3 or evidence.mean()<.1:return None
        # Appearance check also guards optical-flow points that follow the gripper.
        ox1,oy1,ox2,oy2=pixels(old,self.previous.shape)
        old_patch=self.previous[oy1:oy2,ox1:ox2]
        patch=image[y1:y2,x1:x2]
        if min(old_patch.shape[:2])<2 or min(patch.shape[:2])<2:return None
        patch=cv2.resize(patch,(old_patch.shape[1],old_patch.shape[0]))
        distance=float(np.abs(patch.astype(float)-old_patch.astype(float)).mean())/255
        if distance>.18:return None
        return candidate

    def step(self,image,boxes,scores):
        boxes=np.asarray(boxes,np.float32).reshape(-1,4)
        scores=np.asarray(scores,np.float32)
        cfg=self.config
        strong=select_detection(boxes,scores,1,1,cfg.init_threshold)
        selected=strong
        source="detector" if strong.valid else "missing"
        if self.box is not None and self.age<cfg.max_age:
            overlaps=np.array([iou(b,self.box) for b in boxes])
            keep=(scores>=cfg.low_threshold)&(overlaps>=cfg.association_iou)
            if keep.any():
                # Prefer the established identity over an unrelated higher score.
                ids=np.where(keep)[0]
                k=ids[np.argmax(scores[ids]+.1*overlaps[ids])]
                if not strong.valid or strong.ambiguous or scores[k]>=cfg.init_threshold:
                    selected=Detection(boxes[k].copy(),float(scores[k]),True,False,1)
                    source="association"
        if not selected.valid and cfg.method in ("template","flow","combined"):
            local=self._local(image)
            if local is not None:
                selected=Detection(local,.0,True,False,1);source=cfg.method
        if selected.valid:
            refined=color_refine(image,selected.box) if cfg.refine else selected.box
            box=(1-cfg.refine_alpha)*selected.box+cfg.refine_alpha*np.asarray(refined) if cfg.refine else selected.box
            selected=Detection(box,selected.score,True,selected.ambiguous,selected.candidates)
            self.box=box.copy()
            # Local-only evidence cannot keep refreshing an indefinite track.
            self.age=self.age+1 if source in ("template","flow","combined") else 0
            x1,y1,x2,y2=pixels(box,image.shape)
            self.patch=image[y1:y2,x1:x2].copy()
        else:
            self.age+=1
            if self.age>=cfg.max_age:self.box=None;self.patch=None
        self.previous=image.copy()
        return selected,source
