"""Offline subject-switch diagnostics; labels never enter selector inference."""
import numpy as np


def sequence_metrics(pred,data,index,max_gap_steps=16):
    pred=np.asarray(pred);known=set(int(i) for i in index)
    result=dict(eligible_pairs=0,stable_subject_pairs=0,stable_both_selected=0,stable_class_switches=0,
        stable_correct_to_null=0,gold_subject_changes=0,correct_box_at_change=0,stale_previous_class_at_change=0)
    for episode in sorted({r['episode_id'] for r in data['rows']}):
        ids=sorted([i for i,r in enumerate(data['rows']) if r['episode_id']==episode],key=lambda i:data['rows'][i]['step'])
        for left,right in zip(ids,ids[1:]):
            if left not in known or right not in known:continue
            if not 0<data['rows'][right]['step']-data['rows'][left]['step']<=max_gap_steps:continue
            result['eligible_pairs']+=1
            a=data['rows'][left]['label']['target'];b=data['rows'][right]['label']['target']
            old,new=int(pred[left]),int(pred[right]);selected_old=old<8;selected_new=new<8
            old_class=data['rows'][left]['objects'][old]['class_id'] if selected_old else None
            new_class=data['rows'][right]['objects'][new]['class_id'] if selected_new else None
            if a['instance_id']==b['instance_id']:
                result['stable_subject_pairs']+=1
                if selected_old and selected_new:
                    result['stable_both_selected']+=1;result['stable_class_switches']+=int(old_class!=new_class)
                if selected_old and data['positive'][left,old] and not selected_new:result['stable_correct_to_null']+=1
            else:
                result['gold_subject_changes']+=1
                result['correct_box_at_change']+=int(selected_new and data['positive'][right,new])
                result['stale_previous_class_at_change']+=int(selected_new and a['class_id']!=b['class_id'] and new_class==a['class_id'])
    den=result['stable_both_selected'];result['stable_class_switch_rate']=result['stable_class_switches']/den if den else None
    result['max_gap_steps']=max_gap_steps;result['caveat']='Proxy subject changes; abstention separately counted; no assertion of true-intention switching or same-class instance tracking'
    return result
