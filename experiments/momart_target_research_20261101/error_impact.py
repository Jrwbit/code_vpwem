"""Count wrong acquisition queries, runs and switch recovery without exemptions."""
import numpy as np


def analyze_impact(data,pred,index,max_gap=16):
    pred=np.asarray(pred);known=set(map(int,index));rows=data['rows'];events=[];changes=[];episodes={rows[i]['episode_id'] for i in known}
    wrong=[];rack=[];acquire=[];by_episode={};stable_pairs=0;stable_selected_pairs=0;stable_jumps=[]
    def correct(i):return bool(pred[i]<8 and data['positive'][i,pred[i]])
    def cls(i):return rows[i]['objects'][pred[i]]['class_id'] if pred[i]<8 else None
    for ep in sorted(episodes):
        ids=sorted([i for i,r in enumerate(rows) if r['episode_id']==ep],key=lambda i:rows[i]['step']);runs=[];active=[]
        ep_wrong=[];ep_acquire=[]
        for i in ids:
            bad=i in known and data['roles'][i]==0 and pred[i]<8 and cls(i)!=0
            if i in known and data['roles'][i]==0:acquire.append(i);ep_acquire.append(i)
            if bad:
                wrong.append(i);ep_wrong.append(i)
                if cls(i)==3:rack.append(i)
            if active and (not bad or rows[i]['step']-rows[active[-1]]['step']>max_gap):runs.append(active);active=[]
            if bad:active.append(i)
        if active:runs.append(active)
        for run in runs:
            events.append(dict(episode_id=ep,indices=run,first_step=int(rows[run[0]]['step']),last_step=int(rows[run[-1]]['step']),
                observed_step_span=int(rows[run[-1]]['step']-rows[run[0]]['step']),queries=len(run),selected_classes=[int(cls(i)) for i in run]))
        for k,(left,right) in enumerate(zip(ids,ids[1:])):
            if left not in known or right not in known or not 0<rows[right]['step']-rows[left]['step']<=max_gap:continue
            old=rows[left]['label']['target'];new=rows[right]['label']['target']
            if old['instance_id']==new['instance_id']:
                stable_pairs+=1
                if pred[left]<8 and pred[right]<8:
                    stable_selected_pairs+=1
                    if cls(left)!=cls(right):stable_jumps.append(dict(episode_id=ep,left=left,index=right,step=int(rows[right]['step'])))
                continue
            recovery=None;follow=[];censored=True
            for j in ids[k+1:]:
                if j not in known or rows[j]['label']['target']['instance_id']!=new['instance_id']:break
                if follow and rows[j]['step']-rows[follow[-1]]['step']>max_gap:break
                follow.append(j)
                if correct(j):recovery=int(rows[j]['step']-rows[right]['step']);censored=False;break
            changes.append(dict(episode_id=ep,left=left,index=right,step=int(rows[right]['step']),correct=correct(right),selected_class=cls(right),
                weak_target_class=int(new['class_id']),reject=bool(pred[right]==8),target_available=bool(data['positive'][right].any()),
                stale_previous_class=bool(pred[right]<8 and old['class_id']!=new['class_id'] and cls(right)==old['class_id']),
                recovery_steps=recovery,followup_queries=len(follow),followup_step_span=int(rows[follow[-1]]['step']-rows[right]['step']),
                censored_without_correct=censored))
        by_episode[ep]=dict(known_queries=sum(i in known for i in ids),acquire_queries=len(ep_acquire),wrong_acquire=len(ep_wrong),events=len(runs))
    failed=[x for x in changes if not x['correct']]
    summary=dict(known_queries=len(known),episodes=len(episodes),acquire_queries=len(acquire),wrong_acquire_queries=len(wrong),wrong_acquire_to_rack=len(rack),
        wrong_acquire_other_classes=len(wrong)-len(rack),wrong_acquire_events=len(events),wrong_acquire_episodes=sum(x['wrong_acquire']>0 for x in by_episode.values()),
        episodes_with_acquire=sum(x['acquire_queries']>0 for x in by_episode.values()),single_query_acquire_events=sum(x['queries']==1 for x in events),
        multi_query_acquire_events=sum(x['queries']>1 for x in events),max_acquire_event_queries=max([x['queries'] for x in events],default=0),
        max_acquire_observed_step_span=max([x['observed_step_span'] for x in events],default=0),
        subject_changes=len(changes),failed_at_change=len(failed),failed_change_episodes=len({x['episode_id'] for x in failed}),
        stale_previous_class=sum(x['stale_previous_class'] for x in changes),rejected_at_change=sum(x['reject'] for x in changes),
        failed_change_target_available=sum(x['target_available'] for x in failed),
        failed_change_recovered_within16steps=sum(x['recovery_steps'] is not None and x['recovery_steps']<=16 for x in failed),
        failed_change_no_correct_in_observed_followup=sum(x['censored_without_correct'] for x in failed),
        episodes_with_either_issue=len({x['episode_id'] for x in events}|{x['episode_id'] for x in failed}),
        stable_subject_pairs=stable_pairs,stable_both_selected_pairs=stable_selected_pairs,stable_class_jumps=len(stable_jumps),
        stable_class_jump_episodes=len({x['episode_id'] for x in stable_jumps}))
    return dict(summary=summary,episodes=by_episode,acquire_events=events,subject_changes=changes,
        stable_class_jumps=stable_jumps,
        caveat='Known weak-label queries only; query gaps/unknowns split events. Span is last-first observation, not exact physical error duration. Censored recovery is not permanent failure. No policy rollout or task-success causality.')
