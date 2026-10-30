"""One drawing rule shared by offline training and online inference."""
import numpy as np

def draw_bbox(rgb, box, valid, thickness=1):
    """Draw an RGB-red outline in original pixels; missing keeps image unchanged."""
    image=np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected uint8 HWC RGB")
    out=image.copy()
    if not valid:
        return out
    box=np.asarray(box,dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Invalid normalized xyxy")
    h,w=out.shape[:2]
    box=np.clip(box,0,1)*np.array([w,h,w,h])
    x1,y1=np.floor(box[:2]).astype(int)
    x2,y2=np.ceil(box[2:]).astype(int)-1
    x1,x2=np.clip([x1,x2],0,w-1);y1,y2=np.clip([y1,y2],0,h-1)
    if x2<x1 or y2<y1:
        return out
    color=np.array([255,0,0],dtype=np.uint8)
    for offset in range(thickness):
        if x1+offset>x2-offset or y1+offset>y2-offset: break
        out[y1+offset,x1+offset:x2-offset+1]=color
        out[y2-offset,x1+offset:x2-offset+1]=color
        out[y1+offset:y2-offset+1,x1+offset]=color
        out[y1+offset:y2-offset+1,x2-offset]=color
    return out

class OnlineBoxedRGB:
    """Replace all recent main-view RGB frames before baseline normalization."""
    def __init__(self,detector,batch_size=4):
        self.detector=detector
        self.batch_size=batch_size

    def __call__(self,obs):
        values=np.asarray(obs["rgb"])
        if values.ndim != 5 or values.shape[2]!=3:
            raise ValueError("Expected batch x recent-frames x 3 x H x W")
        if not np.isfinite(values).all() or values.min()<0 or values.max()>1:
            raise ValueError("Expected processed RGB in [0,1]")
        images=np.moveaxis(values.reshape(-1,*values.shape[2:]),1,-1)
        raw=np.rint(images*255).astype(np.uint8)
        detections=[]
        for i in range(0,len(raw),self.batch_size):
            detections.extend(self.detector(list(raw[i:i+self.batch_size])))
        if len(detections)!=len(raw):
            raise RuntimeError("Detector batch mismatch")
        boxed=images.copy()
        for i,d in enumerate(detections):
            if d.valid:
                # Modify only outline pixels, preserving original float RGB elsewhere.
                outline=draw_bbox(np.zeros_like(raw[i]),d.box,True)[...,0]>0
                boxed[i][outline]=[1,0,0]
        result=dict(obs)
        result["rgb"]=np.moveaxis(boxed,-1,1).reshape(values.shape)
        return result
