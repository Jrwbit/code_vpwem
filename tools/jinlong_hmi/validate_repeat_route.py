#!/usr/bin/env python3
"""Recheck an unchanged archived route at current feedback; no SDK or writes."""
import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads((args.source/'input.json').read_text())
    plan = json.loads((args.source/'plan.json').read_text())
    receipt = json.loads((args.source/'receipt.json').read_text())
    state = json.loads(args.snapshot.read_text())
    control, root = Path(data['control_root']), Path(data['sim_root'])
    app = Path.home()/'r435-release/r435-vision-1080ti-v1/app'
    sys.path[:0] = [str(control), str(root), str(app)]
    from sim.helios_insert import corridor as cr, fixtures as fx, scene as sc
    from hk_pointcloud_registration.robot.corridor_worker import configure_geometry, planning_limits
    from helios_hmi.kinematic_reference import live_vectors
    from helios_hmi.controller_kinematics import attach_models, signature, apply_collision_model
    live_site = attach_models({}, state)
    if signature(data['site']) != signature(live_site):
        raise ValueError('控制器运动学参数与原路线不一致，请重新规划')
    if receipt.get('kinematic_sha256') != signature(data['site']):
        raise ValueError('路线运动学来源摘要不一致')
    original_build = cr.build
    cr.build = lambda cfg: apply_collision_model(original_build(cfg), data['site'])
    waist, arm, _ = live_vectors(state)
    for names, values in (([f'{i}-Joint' for i in (5, 6)], data['static_pose']['head']),
                          ([f'AR5-5_07R_joint_{i}' for i in range(1, 8)], data['static_pose']['right_arm'])):
        actual = np.array([state['real']['joints'][n] for n in names])
        if np.max(abs(actual-np.asarray(values))) > np.radians(1):
            raise ValueError('头部／右臂偏离原路线场景')
    if not receipt.get('collision_checked') or len(plan['segments']) != len(receipt['stages']):
        raise ValueError('原路线没有完整的碰撞检查来源')
    limits = planning_limits(state['limits'], data['site'])
    sc.VENDOR = control/'helios_hmi/model/Wheel_Robot'
    scene = json.loads((args.source/'scene_used.json').read_text())
    scene['static_pose'] = {'head': [state['real']['joints'][f'{i}-Joint'] for i in (5, 6)],
                           'right_arm': [state['real']['joints'][f'AR5-5_07R_joint_{i}'] for i in range(1, 8)]}
    scene['left_arm_soft_limits_rad'] = {'lower': [limits[n][0] for n in sc.LEFT],
                                        'upper': [limits[n][1] for n in sc.LEFT]}
    previous = arm
    minimum = float('inf')
    checked = 0
    with tempfile.TemporaryDirectory(prefix='repeat-route-validation-') as directory:
        configure_geometry(scene, fx, data['site'], root, Path(directory))
        ctx = cr.Ctx(copy.deepcopy(scene), viz=False)
        for segment, stage in zip(plan['segments'], receipt['stages']):
            if segment['group'] != 'left_arm' or segment['name'] != stage['name']:
                raise ValueError('重复试验仅允许原左臂路线')
            if np.max(abs(waist-np.asarray(segment['hold']['waist']))) > np.radians(1):
                raise ValueError('腰姿偏离原路线')
            if np.max(abs(previous-np.asarray(segment['start']))) > np.radians(1):
                raise ValueError('原路线起点或连续性不匹配')
            for step in segment['steps']:
                target = np.asarray(step['target'])
                for n, v in zip(sc.WAIST+sc.LEFT, np.r_[waist, target]):
                    if not limits[n][0]+np.radians(2) <= v <= limits[n][1]-np.radians(2):
                        raise ValueError('当前限位或2°余量检查失败：'+n)
                points = cr.check_line(ctx, np.r_[waist, previous], np.r_[waist, target],
                                       adjust=stage['role'] == 'adjust',
                                       contact_motor=data['site']['selected_fixture'] if stage['role']=='insert' else None)
                minimum = min(minimum, *(point['clear']*1000 for point in points))
                previous = target
                checked += 1
    if np.max(abs(previous-np.asarray(plan['segments'][0]['start']))) > 1e-8:
        raise ValueError('原路线不是返回同一home的完整闭环')
    result = {'collision_checked': True, 'steps': checked, 'minimum_model_clearance_mm': minimum,
              'source_plan_sha256': hashlib.sha256((args.source/'plan.json').read_bytes()).hexdigest(),
              'current_waist_model_rad': waist.tolist(), 'controller_writes': False}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
