#!/usr/bin/env python3
"""Offline check of reverse traversal of a natively acknowledged route prefix.

Produces a diagnostic receipt only. Never loads a plan or connects to an SDK.
"""
import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def completed_prefix(plan, events):
    """Bind every native acknowledgement to the exact archived target."""
    steps = [(s, step) for s in plan['segments'] for step in s['steps']]
    done = [e for e in events if e.get('event') == 'route_step_done']
    if not done or len(done) >= len(steps):
        raise ValueError('需要未完成路线的非空逐点完成记录')
    for i, event in enumerate(done):
        segment, step = steps[i]
        target = np.asarray(event.get('target'), float)
        expected = np.asarray(step['target'], float)
        if (segment['group'] != 'left_arm' or event.get('group') != 'left_arm'
                or type(event.get('step')) is not int or event['step'] != i+1
                or target.shape != (7,) or expected.shape != (7,)
                or not np.isfinite(target).all() or not np.isfinite(expected).all()
                or np.max(abs(target-expected)) > 1e-10):
            raise ValueError('原生完成事件与原路线不匹配')
    return len(done)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('native_log', type=Path)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = json.loads((args.source/'input.json').read_text())
    plan = json.loads((args.source/'plan.json').read_text())
    receipt = json.loads((args.source/'receipt.json').read_text())
    state = json.loads(args.snapshot.read_text())
    events = [json.loads(line) for line in args.native_log.read_text().splitlines() if line.strip()]
    completed = completed_prefix(plan, events)
    motion = state['motion']
    if any(motion.get(k) for k in ('busy', 'capturing', 'latched')) or state['plan'].get('active'):
        raise ValueError('需要空闲且未锁存的现场状态')
    for part in state['real']['components'].values():
        if (part.get('ok') is not True or not 0 <= part['age_s'] <= 1
                or (part['power'], part['mode'], part['op']) != ('off', 'manual', 'idle')
                or abs(part['max_vel']) > .01):
            raise ValueError('需要停稳下电的新鲜反馈')
    root, control = Path(source['sim_root']), Path(source['control_root'])
    app = Path.home()/'r435-release/r435-vision-1080ti-v1/app'
    sys.path[:0] = [str(control), str(root), str(app)]
    from sim.helios_insert import corridor as cr, fixtures as fx, scene as sc
    from helios_hmi.controller_kinematics import attach_models, signature, apply_collision_model
    from hk_pointcloud_registration.robot.corridor_worker import configure_geometry, planning_limits
    if signature(source['site']) != signature(attach_models({}, state)):
        raise ValueError('控制器运动学与原路线不一致')
    if not receipt.get('collision_checked') or len(plan['segments']) != len(receipt['stages']):
        raise ValueError('缺少原逐阶段碰撞检查来源')
    original_build = cr.build
    cr.build = lambda cfg: apply_collision_model(original_build(cfg), source['site'])
    sc.VENDOR = control/'helios_hmi/model/Wheel_Robot'
    scene = json.loads((args.source/'scene_used.json').read_text())
    joints = state['real']['joints']
    waist = np.array([joints[f'{i}-Joint'] for i in range(1, 5)])
    arm = np.array([joints[f'AR5-5_07L_joint_{i}'] for i in range(1, 8)])
    static = {'head': [joints[f'{i}-Joint'] for i in (5, 6)],
              'right_arm': [joints[f'AR5-5_07R_joint_{i}'] for i in range(1, 8)]}
    for name, actual in static.items():
        if np.max(abs(np.asarray(actual)-source['static_pose'][name])) > np.radians(1):
            raise ValueError('头部或右臂改变，原场景失效')
    scene['static_pose'] = static
    limits = planning_limits(state['limits'], source['site'])
    scene['left_arm_soft_limits_rad'] = {'lower': [limits[n][0] for n in sc.LEFT],
                                        'upper': [limits[n][1] for n in sc.LEFT]}
    traversed = []
    for segment, stage in zip(plan['segments'], receipt['stages']):
        if segment['group'] != 'left_arm' or segment['name'] != stage['name']:
            raise ValueError('仅检查来源一致的左臂原路线')
        if np.max(abs(waist-np.asarray(segment['hold']['waist']))) > np.radians(1):
            raise ValueError('腰姿偏离原路线超过1度')
        previous = np.array(segment['start'])
        for step in segment['steps']:
            traversed.append((previous.copy(), np.asarray(step['target']), stage['role']))
            previous = np.asarray(step['target'])
    # The interrupted edge is not completed. Rejoin the last acknowledged point
    # through an independently checked line; then reverse only completed edges.
    rows = [(traversed[completed-1][1], traversed[completed][2])]
    rows += [(start, role) for start, _, role in reversed(traversed[:completed])]
    minimum = float('inf')
    with tempfile.TemporaryDirectory(prefix='interrupted-return-') as directory:
        configure_geometry(scene, fx, source['site'], root, Path(directory))
        ctx = cr.Ctx(copy.deepcopy(scene), viz=False)
        for index, (target, role) in enumerate(rows):
            for name, value in zip(sc.WAIST+sc.LEFT, np.r_[waist, target]):
                if not limits[name][0]+np.radians(2) <= value <= limits[name][1]-np.radians(2):
                    raise ValueError('当前限位余量不满足：'+name)
            print('Checking reverse line', index+1, role, flush=True)
            frames = cr.check_line(ctx, np.r_[waist, arm], np.r_[waist, target], adjust=role=='adjust',
                                   contact_motor=source['site']['selected_fixture'] if role=='insert' else None)
            minimum = min(minimum, *(frame['clear']*1000 for frame in frames))
            arm = target
    result = {'collision_checked': True, 'completed_native_steps': completed, 'reverse_lines': len(rows),
              'minimum_model_clearance_mm': minimum, 'controller_writes': False,
              'source_plan_sha256': hashlib.sha256((args.source/'plan.json').read_bytes()).hexdigest(),
              'native_log_sha256': hashlib.sha256(args.native_log.read_bytes()).hexdigest(),
              'snapshot_sha256': hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
              'assumption': 'Original scene and tool unchanged; receipt alone does not authorize execution.'}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
