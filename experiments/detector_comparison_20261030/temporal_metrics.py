"""Instance-specific temporal metrics. GT stays on the scoring side."""
import math
import numpy as np
from metrics import box_iou


def wilson(tp, fp):
    n=tp+fp
    if not n:return None
    p=tp/n;z=1.959963984540054
    mid=(p+z*z/(2*n))/(1+z*z/n)
    width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [mid-width,mid+width]


def match(objects, truth):
    used=set();result={}
    for i in sorted(range(len(objects)),key=lambda i:-objects[i]['score']):
        o=objects[i]
        choices=[(box_iou(o['box'],g['box']),j) for j,g in enumerate(truth)
                 if j not in used and g['class_id']==o['class_id']]
        overlap,j=max(choices,default=(0.,-1))
        if overlap>=.5:used.add(j);result[i]=truth[j]['instance_id']
    return result


def stats(values):
    return dict(count=len(values),median=float(np.median(values)) if values else None,
                p95=float(np.percentile(values,95)) if values else None,
                p99=float(np.percentile(values,99)) if values else None,
                maximum=float(max(values)) if values else None)


def miss_runs(rows, truth, hits, observed_only=False):
    """Invisible frames break runs; sparse anchors never count as dense misses."""
    identities={o['instance_id'] for r in rows for o in truth[r['id']]}
    lengths=[];seconds=[]
    for identity in identities:
        run=[];previous=None
        def close():
            if run:
                lengths.append(len(run));seconds.append((run[-1]-run[0]+1)*.05);run.clear()
        for r in rows:
            step=r['step']
            if not observed_only and previous is not None and step!=previous+1:close()
            visible=any(o['instance_id']==identity for o in truth[r['id']])
            if visible and identity not in hits[r['id']]:run.append(step)
            else:close()
            previous=step
        close()
    return dict(length_in_observed_samples=stats(lengths),span_seconds=stats(seconds),
                boundary_runs_may_be_censored=True)


def event_scores(events, rows, truth, hits, observed):
    by_episode={}
    for r in rows:by_episode.setdefault(r['episode_id'],[]).append(r)
    result=[]
    for e in events:
        available=by_episode[e['episode_id']];target=e['instance_id'];t=e['onset']
        pre=[r for r in available if max(0,t-16)<=r['step']<t]
        critical=[r for r in available if max(0,t-8)<=r['step']<=t+8]
        one=dict(e,pre_visible_frames=sum(any(o['instance_id']==target for o in truth[r['id']]) for r in pre))
        for tag,subset in [('dense',set(r['id'] for r in available)),('policy',observed)]:
            candidates=[r for r in pre if r['id'] in subset]
            windows=[r for r in critical if r['id'] in subset and any(o['instance_id']==target for o in truth[r['id']])]
            positive=sum(target in hits[r['id']] for r in windows)
            after=[r for r in available if t<=r['step']<=e['end']+8 and r['id'] in subset and target in hits[r['id']]]
            one[tag]=dict(first_hit=any(target in hits[r['id']] for r in candidates),
                pre_observable=any(any(o['instance_id']==target for o in truth[r['id']]) for r in candidates),
                critical_tp=positive,critical_visible=len(windows),critical_recall=positive/len(windows) if windows else None,
                first_post_motion_hit_delay_seconds=(after[0]['step']-t)*.05 if after else None)
        result.append(one)
    summary={}
    for tag in ('dense','policy'):
        opportunities=[e[tag] for e in result if e[tag]['pre_observable']]
        denom=sum(e[tag]['critical_visible'] for e in result);tp=sum(e[tag]['critical_tp'] for e in result)
        summary[tag]=dict(events=len(result),first_hit_events=sum(e[tag]['first_hit'] for e in result),
            observable_pre_events=len(opportunities),first_hit_rate_all=sum(e[tag]['first_hit'] for e in result)/len(result) if result else None,
            first_hit_rate_given_pre_observable=sum(e['first_hit'] for e in opportunities)/len(opportunities) if opportunities else None,
            critical_visible_observations=denom,critical_tp=tp,critical_recall=tp/denom if denom else None)
    return dict(events=result,summary=summary)


def identity_continuity(samples, event_count):
    """Conservative coverage+identity metric; invisible samples omitted upstream.

    A visible missed target has track=None and stays in the denominator. This
    intentionally differs from continuity conditional on a successful detection.
    """
    from collections import Counter
    events=[]
    for event in range(event_count):
        rows=[r for r in samples if r['event']==event]
        counts=Counter(r['track'] for r in rows if r['track'] is not None)
        dominant=max(counts.values(),default=0)
        events.append(dict(event=event,dominant_track_hits=dominant,
            gt_visible_policy_samples=len(rows),continuity=dominant/len(rows) if rows else None))
    denominator=sum(e['gt_visible_policy_samples'] for e in events)
    numerator=sum(e['dominant_track_hits'] for e in events)
    return dict(definition='sum per-event dominant-slot correct detections / GT-visible policy samples in onset +/-8 steps; misses count against continuity',
        numerator=numerator,denominator=denominator,rate=numerator/denominator if denominator else None,events=events)


def mixed_identity_history(labels, target):
    """Known mixed histories only; unknown matches cannot certify cleanliness."""
    return target in labels and any(label is not None and label!=target for label in labels)
