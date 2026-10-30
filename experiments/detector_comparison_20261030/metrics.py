"""Class-aware, one-to-one matching and development-only threshold selection."""
import numpy as np


def box_iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    size=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-size
    return float(size/union) if union>0 else 0.


def measure(records, labels, thresholds, overlap=.5):
    counts=np.zeros((6,3),dtype=int) # TP, FP, FN
    for r in records:
        truth=labels[r['id']]['objects']
        used=set()
        predictions=sorted(r['objects'],key=lambda x:-x['score'])
        for pred in predictions:
            cls=pred['class_id']
            if pred['score']<thresholds[cls]:continue
            candidates=[(box_iou(pred['box'],g['box']),i) for i,g in enumerate(truth)
                        if g['class_id']==cls and i not in used]
            best=max(candidates,default=(0.,-1))
            if best[0]>=overlap:
                used.add(best[1]);counts[cls,0]+=1
            else:counts[cls,1]+=1
        for i,g in enumerate(truth):
            if i not in used:counts[g['class_id'],2]+=1
    def rates(row):
        tp,fp,fn=map(int,row)
        return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
                    recall=tp/(tp+fn) if tp+fn else None,
                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return dict(per_class=[rates(row) for row in counts],micro=rates(counts.sum(0)),
                macro_f1=float(np.mean([rates(row)['f1'] or 0 for row in counts])),
                iou=overlap,thresholds=list(map(float,thresholds)))


def calibrate(records, labels):
    """Prefer P>=.95 with at least five predictions; otherwise max F1, flag failure.

    Selected once on dev and frozen for test. This is a pilot operating point,
    not a confidence bound on population precision.
    """
    grid=np.unique(np.r_[.01,.025,np.arange(.05,.96,.05)])
    metrics=[measure(records,labels,[t]*6) for t in grid]
    thresholds=[];feasible=[]
    for cls in range(6):
        candidates=[(m['per_class'][cls],float(t)) for t,m in zip(grid,metrics)]
        good=[(r,t) for r,t in candidates if r['precision'] is not None and
              r['precision']>=.95 and r['tp']+r['fp']>=5]
        pool=good or candidates
        row,t=max(pool,key=lambda x:((x[0]['f1'] or 0),(x[0]['recall'] or 0),-x[1]))
        thresholds.append(t);feasible.append(bool(good))
    return dict(thresholds=thresholds,dev_precision_target_feasible=feasible,
                dev=measure(records,labels,thresholds))
