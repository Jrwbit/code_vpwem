"""Fixed-threshold detection diagnostics; weak intention targets scored separately."""
import numpy as np


def iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-intersection
    return float(intersection/max(union,1e-8))


def instance_counts(predictions,truth,threshold=.5,return_matches=False):
    counts={str(c):dict(tp=0,fp=0,fn=0) for c in range(6)};matched=set()
    for obj in sorted(predictions,key=lambda x:-x['score']):
        cls=int(obj['class_id']);available=[j for j,t in enumerate(truth) if j not in matched and t['class_id']==cls]
        best=max(available,key=lambda j:iou(obj['box'],truth[j]['box'])) if available else None
        if best is not None and iou(obj['box'],truth[best]['box'])>=threshold:matched.add(best);counts[str(cls)]['tp']+=1
        else:counts[str(cls)]['fp']+=1
    for j,t in enumerate(truth):
        if j not in matched:counts[str(int(t['class_id']))]['fn']+=1
    return (counts,matched) if return_matches else counts


def aggregate_counts(records):
    result={str(c):dict(tp=0,fp=0,fn=0) for c in range(6)}
    for record in records:
        for c,values in record.items():
            for key,value in values.items():result[c][key]+=value
    for values in result.values():
        tp,fp,fn=values['tp'],values['fp'],values['fn'];values.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
    totals={k:sum(v[k] for v in result.values()) for k in ['tp','fp','fn']};tp,fp,fn=totals['tp'],totals['fp'],totals['fn']
    totals.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
    return dict(classes=result,micro=totals)
