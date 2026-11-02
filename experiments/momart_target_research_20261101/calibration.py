"""Rejection using predicted public proposal classes, never reference roles."""
import numpy as np


def apply_calibration(prob,cal,classes=None):
    order=np.sort(prob,axis=-1);chosen=prob.argmax(-1);threshold=cal['threshold'];margin=cal.get('margin',0)
    if 'class_thresholds' in cal or 'class_margins' in cal:
        if classes is None or np.shape(classes)!=(len(prob),8):
            raise ValueError('Class calibration requires public proposal classes [N,8]')
        thresholds=np.full(len(prob),threshold,dtype=prob.dtype)
        selected_class=np.asarray(classes)[np.arange(len(prob)),np.minimum(chosen,7)]
        for key,value in cal.get('class_thresholds',{}).items():
            if int(key) not in range(6) or not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid class threshold')
            thresholds[(chosen<8)&(selected_class==int(key))]=value
        threshold=thresholds
        margins=np.full(len(prob),margin,dtype=prob.dtype)
        for key,value in cal.get('class_margins',{}).items():
            if int(key) not in range(6) or not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid class margin')
            margins[(chosen<8)&(selected_class==int(key))]=value
        margin=margins
    return np.where((prob.max(-1)>=threshold)&((order[:,-1]-order[:,-2])>=margin),chosen,8)
