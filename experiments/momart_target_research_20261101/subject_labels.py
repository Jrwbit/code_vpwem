"""Separate manipulation subject from unlabelled placement recipient."""
from copy import deepcopy


def subject_label(record):
    old=record['old_target'];evidence=record['evidence']
    bowl_id=record['label']['subject'] if record['label'] and record['label']['subject'] else None
    if record['held_now']:
        if bowl_id is None:raise ValueError('Holding evidence requires an identified bowl')
        return dict(role=4,target=dict(instance_id=bowl_id,class_id=0),weight=.95,
                    reason='holding same bowl; recipient remains separately unknown')
    if record['held_before']:
        return None  # Release/transition is not a verified negative or known next target.
    if record['held_soon'] and evidence['bowl_distance']<=.10:
        if bowl_id is None:raise ValueError('Pre-grasp evidence requires a bowl identity')
        return dict(role=0,target=dict(instance_id=bowl_id,class_id=0),weight=.9,reason='pre-grasp proxy')
    if old is None:return None
    cls=old['class_id']
    roles={0:0,2:1,3:2,5:3}
    if cls not in roles:return None
    # A released/falling bowl moving without a nearby gripper is ambiguous.
    if cls==0 and old['onset']<=record['step'] and evidence['bowl_distance']>.10 and evidence['grasped']<0:
        return None
    return dict(role=roles[cls],target=deepcopy(old),weight=.65 if old['onset']>record['step'] else .8,
                reason='motion/proximity proxy; pre-contact intention may be ambiguous')
