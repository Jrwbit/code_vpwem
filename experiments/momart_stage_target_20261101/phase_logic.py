"""Offline stage supervision; privileged geometry and future evidence are labels only."""
import numpy as np

ROLES=['acquire_bowl','operate_door_handle','operate_rack','operate_drawer_handle','carry_place_unavailable']
ROLE_CLASS=[0,2,3,5,-1]


def holding_proxy(current,previous):
    return (current['grasped']>0 and previous['grasped']>0
            and current['bowl_distance']<=.035 and previous['bowl_distance']<=.035
            and np.linalg.norm(np.asarray(current['bowl_relative'])-previous['bowl_relative'])<=.025)


def stage_label(old_target,held_now,held_before,held_soon,near_bowl,bowl_id):
    if held_now or held_before:
        return dict(role=4,target=None,subject=bowl_id,reason='held/recently_released_bowl; recipient region unavailable')
    if held_soon and near_bowl:
        return dict(role=0,target=dict(instance_id=bowl_id,class_id=0),subject=bowl_id,reason='pre-grasp proxy')
    if old_target is None:
        return None
    cls=old_target['class_id']
    if cls not in ROLE_CLASS[:-1]:return None
    return dict(role=ROLE_CLASS.index(cls),target=old_target,subject=bowl_id if cls==0 else None,
                reason='motion/proximity interaction proxy')
