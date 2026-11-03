"""Train-only viewport crops, aligned across current/past RGB, depth and labels."""
import numpy as np
from PIL import Image


def validate_crop(crop):
    x,y,w,h=map(int,crop)
    if list(crop)!=[x,y,w,h] or min(w,h)<=0 or x<0 or y<0 or x+w>120 or y+h>90 or w*3!=h*4:
        raise ValueError('Expected integer4:3 viewport inside120x90')
    return x,y,w,h


def pixel_affine(crop):
    x,y,w,h=validate_crop(crop)
    return np.array([120/w,90/h,-x/w,-y/h],np.float32)


def apply_crop(array,crop,nearest=False):
    x,y,w,h=validate_crop(crop);a=np.asarray(array)
    if a.shape[:2]!=(90,120):raise ValueError('Expected120x90 sensor array')
    singleton=a.ndim==3 and a.shape[-1]==1
    if singleton:a=a[...,0]
    boolean=a.dtype==bool
    image=Image.fromarray(a.astype(np.uint8) if boolean else a)
    value=np.asarray(image.crop((x,y,x+w,y+h)).resize((120,90),Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR))
    if boolean:value=value.astype(bool)
    if singleton:value=value[...,None]
    return value.copy()


def mask_box(mask):
    y,x=np.where(mask)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None


def choose_crop(target_mask,rng):
    # Target-aware transform choice is training-only; its parameters are never online labels.
    w,h=96,72;fallback=[int(rng.integers(0,25)),int(rng.integers(0,19)),w,h]
    if target_mask is None or not np.any(target_mask):return fallback,False
    box=mask_box(target_mask);x1,y1,x2,y2=box;scale=1.25
    if min(x2-x1,y2-y1)<=8:return fallback,False
    candidates=[]
    for leave in [2,4,8]:
        for x in [round(x2-leave/scale),round(x1+leave/scale-w)]:
            if 0<=x<=24:candidates.append([x,int(rng.integers(0,19)),w,h])
        for y in [round(y2-leave/scale),round(y1+leave/scale-h)]:
            if 0<=y<=18:candidates.append([int(rng.integers(0,25)),y,w,h])
    best=None
    for crop in candidates:
        cropped=apply_crop(target_mask,crop,nearest=True);b=mask_box(cropped)
        if b is None:continue
        short=min(b[2]-b[0],b[3]-b[1]);retention=cropped.sum()/(scale*scale*target_mask.sum())
        if 1<=short<=8 and retention<.9:
            score=abs(short-4)
            if best is None or score<best[0]:best=(score,crop)
    return (best[1],True) if best is not None else (fallback,False)


def transform_geometry_context(context,crop):
    result=np.asarray(context,np.float32).copy()
    if result.shape!=(14,):raise ValueError('Expected own-robot geometry14')
    affine=pixel_affine(crop);result[3:5]=result[3:5]*affine[:2]+affine[2:];result[12:14]*=affine[:2]
    return result
