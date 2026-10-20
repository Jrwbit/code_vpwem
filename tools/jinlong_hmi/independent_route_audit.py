#!/usr/bin/env python3
"""Independent, offline kinematic route calculation; never exports an HMI plan.

Uses raw feedback, device dimensions, URDF waist geometry and measured box pose.
Does not import the production FK, IK, corridor or collision planner. Results
are kinematic candidates, not collision-checked or approved robot commands.
Translations are metres, angles radians, and transforms map child to parent.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# Modified DH dimensions documented for the AR5-5_0.8 controller model.
# alpha (deg), a (mm), d (mm), fixed theta (deg); last row is fixed flange.
DIMENSIONS = ((0, 0, 174.5, 0), (-90, 0, 0, 0), (90, 0, 332, 0),
              (-90, 10.5, 0, 0), (90, -10.5, 278, 0), (-90, 0, 0, -90),
              (-90, 0, 0, 90), (90, 0, 90, 90))


def transform(rotation=None, translation=None):
    t = np.eye(4)
    if rotation is not None:
        t[:3, :3] = rotation
    if translation is not None:
        t[:3, 3] = translation
    return t


def flange(q):
    """Reconstruct modified DH multiplication directly from device dimensions."""
    t = np.eye(4)
    for i, (alpha, a, d, theta) in enumerate(DIMENSIONS):
        t = t @ transform(Rotation.from_rotvec([np.radians(alpha), 0, 0]).as_matrix(),
                          [a / 1000, 0, 0])
        angle = np.radians(theta) + (q[i] if i < 7 else 0)
        t = t @ transform(Rotation.from_rotvec([0, 0, angle]).as_matrix(), [0, 0, d / 1000])
    return t


def link_transform(urdf, link, joint_values):
    """Parse the waist/mounting chain independently; arm URDF is not used as DH."""
    by_child = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}
    chain = []
    while link in by_child:
        j = by_child[link]
        chain.append(j)
        link = j.find('parent').get('link')
    t = np.eye(4)
    for j in reversed(chain):
        origin = j.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        t = t @ transform(Rotation.from_euler('xyz', rpy).as_matrix(), xyz)
        if j.get('type') in ('revolute', 'continuous'):
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
            t = t @ transform(Rotation.from_rotvec(axis * joint_values.get(j.get('name'), 0)).as_matrix())
        elif j.get('type') != 'fixed':
            raise ValueError('Unsupported waist chain joint type')
    return t


def error(actual, reference):
    delta = (actual[:3, 3] - reference[:3, 3]) * 1000
    return {'delta_xyz_mm': delta.tolist(), 'position_mm': float(np.linalg.norm(delta)),
            'rotation_deg': float(np.degrees(Rotation.from_matrix(reference[:3, :3].T @ actual[:3, :3]).magnitude()))}


def inverse(target, seed, bounds, *, restarts=0):
    """Bounded numerical IK, with deterministic independent starting guesses."""
    lo, hi = bounds.T
    def residual(q):
        t = flange(q)
        return np.r_[(t[:3, 3] - target[:3, 3]) * 1000,
                     Rotation.from_matrix(target[:3, :3].T @ t[:3, :3]).as_rotvec() * 200]
    rng = np.random.default_rng(20261020)
    seeds = [np.clip(seed, lo + 1e-8, hi - 1e-8)]
    seeds.extend(rng.uniform(lo, hi) for _ in range(restarts))
    solutions = []
    for start in seeds:
        result = least_squares(residual, start, bounds=(lo, hi), max_nfev=350,
                               ftol=1e-11, xtol=1e-11, gtol=1e-9)
        e = error(flange(result.x), target)
        if e['position_mm'] <= .02 and e['rotation_deg'] <= .005:
            solutions.append(result.x)
            if restarts == 0:
                break
    if not solutions:
        raise ValueError('Independent IK did not meet 0.02 mm / 0.005 degree model tolerance')
    return min(solutions, key=lambda q: np.linalg.norm(q - seed))


def calculate(snapshot, urdf, job_id):
    data = snapshot['files'][job_id]['input.json']
    site, cal, state = data['site'], data['calibration'], snapshot['state']
    # This routine deliberately never reads any saved plan to choose its route.
    q0 = np.asarray(state['raw']['left_arm']['q'])
    waist = np.asarray([state['real']['joints'][f'{i}-Joint'] for i in range(1, 5)])
    base = link_transform(urdf, 'AR5-5_07L_base', dict(zip((f'{i}-Joint' for i in range(1, 5)), waist)))
    ref = link_transform(urdf, 'AR5-5_07L_base', dict(zip((f'{i}-Joint' for i in range(1, 5)), cal['waist_model_rad'])))
    world_camera = ref @ np.asarray(cal['base_T_camera'])
    box = world_camera @ np.asarray(data['camera_T_box'])
    fid = site['selected_fixture']
    shift = transform(translation=np.asarray(site.get('box_center_adjustments_mm', {}).get(fid, [0, 0, 0])) / 1000)
    offset = np.array(site['tcp_T_objects'][fid], dtype=float)
    offset[2, 3] += site.get('fixture_z_adjustments_mm', {}).get(fid, 0) / 1000
    tool = np.array(site['flange_T_tcp']) @ transform(translation=[0, 0, site.get('tcp_z_adjustment_mm', 0) / 1000])
    goal = box @ shift @ np.linalg.inv(offset) @ np.linalg.inv(tool)
    limits = np.array([state['limits'][f'AR5-5_07L_joint_{i}'] for i in range(1, 8)])
    cap = np.radians(site.get('left_arm_j7_limit_deg', 90))
    limits[6] = [max(limits[6, 0], -cap), min(limits[6, 1], cap)]
    limits += np.radians([2, -2])
    if np.any(q0 < limits[:, 0]) or np.any(q0 > limits[:, 1]):
        raise ValueError('Current arm pose outside controller limits with 2 degree margin')
    hover = goal.copy()
    hover[:3, 3] -= box[:3, 2] * .11
    q_hover = inverse(np.linalg.inv(base) @ hover, q0, limits, restarts=18)
    rows = [q0.copy()]
    segments = []
    def append(name, values):
        start = len(rows) - 1
        rows.extend(np.asarray(q) for q in values)
        segments.append({'name': name, 'start_index': start, 'end_index': len(rows) - 1})
    count = max(1, int(np.ceil(np.max(abs(q_hover - q0)) / np.radians(5))))
    append('当前位 → 110 mm悬停（关节插值，未检查碰撞）',
           [q0 + (q_hover - q0) * i / count for i in range(1, count + 1)])
    approach = [q_hover]
    for gap in np.r_[np.arange(100, 9, -10), np.arange(9, -1, -1)]:
        t = goal.copy()
        t[:3, 3] -= box[:3, 2] * gap / 1000
        q = inverse(np.linalg.inv(base) @ t, approach[-1], limits)
        if np.max(abs(q - approach[-1])) > np.radians(15):
            raise ValueError('Independent Cartesian path has a branch jump')
        approach.append(q)
    append('110 mm悬停 → 10 mm接近', approach[1:11])
    append('10 mm接近 → 检测对位', approach[11:])
    contact_index = len(rows) - 1
    outbound = np.array(rows)
    append('检测对位 → 10 mm接近', list(reversed(approach[10:-1])))
    append('10 mm接近 → 110 mm悬停', list(reversed(approach[:10])))
    append('悬停 → 原当前位', list(reversed(outbound[:count])))
    route = np.array(rows)
    sdk = np.array(state['raw']['left_arm']['T_base_flange_rowmajor']).reshape(4, 4)
    return {'kind': 'independent_kinematic_candidate', 'controller_writes': False,
            'executable': False, 'collision_checked': False,
            'shared_assumptions': 'Same nominal device dimensions, hand-eye and tool definitions; not an independent physical ground truth',
            'input_observed_at': snapshot['observed_at'], 'job_id_for_input_only': job_id,
            'fixture': fid, 'waist_model_rad': waist.tolist(),
            'current_q_rad': q0.tolist(), 'bounds_with_margin_rad': limits.tolist(),
            'world_T_camera': world_camera.tolist(), 'world_T_armbase': base.tolist(),
            'world_T_goal_flange': goal.tolist(), 'base_T_goal_flange': (np.linalg.inv(base) @ goal).tolist(),
            'current_nominal_vs_sdk': error(flange(q0), sdk),
            'segments': segments, 'contact_index': contact_index,
            'joint_waypoints_rad': route.tolist(),
            'world_T_flange_waypoints': [(base @ flange(q)).tolist() for q in route],
            'endpoint_q_deg': np.degrees(route[contact_index]).tolist(),
            'endpoint_error': error(base @ flange(route[contact_index]), goal),
            'returns_to_start': bool(np.allclose(route[-1], q0, atol=1e-12)),
            'all_waypoints_within_limits': bool(np.all((route >= limits[:, 0]) & (route <= limits[:, 1]))),
            'largest_joint_waypoint_step_deg': float(np.degrees(np.max(abs(np.diff(route, axis=0)))))}


def compare_page(result, page, urdf):
    """Read the production route only after independent generation is complete."""
    plan = page['files']['plan.json']
    goal = np.array(result['world_T_goal_flange'])
    independent_end = np.array(result['world_T_flange_waypoints'][result['contact_index']])
    stages = []
    for segment in plan['segments']:
        if segment['group'] != 'left_arm':
            raise ValueError('Comparison expects a fixed-waist left-arm route')
        waist = segment['hold']['waist']
        base = link_transform(urdf, 'AR5-5_07L_base',
                              dict(zip((f'{i}-Joint' for i in range(1, 5)), waist)))
        end_q = np.array(segment['steps'][-1]['target'])
        end = base @ flange(end_q)
        stages.append({'name': segment['name'], 'steps': len(segment['steps']),
                       'end_q_deg': np.degrees(end_q).tolist(), 'world_T_end_flange': end.tolist()})
        if segment.get('pause_at_target'):
            contact = {'page_endpoint_vs_independent': error(end, independent_end),
                       'page_endpoint_vs_analytic_goal': error(end, goal),
                       'endpoint_joint_difference_deg': (np.degrees(end_q) - np.array(result['endpoint_q_deg'])).tolist(),
                       'page_endpoint_if_actual_waist_held': error(np.array(result['world_T_armbase']) @ flange(end_q), goal)}
    if 'contact' not in locals():
        raise ValueError('Page plan has no detection pause endpoint')
    independent_rows = np.array(result['joint_waypoints_rad'])
    page_rows = np.array([plan['segments'][0]['start']] +
                         [step['target'] for s in plan['segments'] for step in s['steps']])
    return {'page_job_id': page['trajectory']['id'],
            'page_task_id': page['trajectory']['task_id'],
            'page_state_at_snapshot': page['trajectory']['state'],
            'page_segments': stages, 'page_steps': sum(len(s['steps']) for s in plan['segments']),
            'independent_steps': len(result['joint_waypoints_rad']) - 1,
            'page_waist_minus_actual_deg': np.degrees(np.array(plan['segments'][0]['hold']['waist']) - np.array(result['waist_model_rad'])).tolist(),
            'scene_target_vs_independent': error(np.array(page['files']['scene_used.json']['targets'][0]['T']), goal),
            'target_comparison': contact,
            'same_joint_route': bool(independent_rows.shape == page_rows.shape and
                                     np.allclose(independent_rows, page_rows, atol=1e-9, rtol=0)),
            'route_difference_reason': 'Independent direct joint transit to hover versus production obstacle-avoidance corridor; redundant arm solutions differ',
            'page_collision_receipt': page['files'].get('receipt.json', {}).get('collision_checked'),
            'independent_collision_checked': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-page', type=Path)
    args = parser.parse_args()
    result = calculate(json.loads(args.snapshot.read_text()), args.urdf, args.job_id)
    if args.compare_page:
        page = json.loads(args.compare_page.read_text())
        expected = json.loads(args.snapshot.read_text())['files'][args.job_id]['input.json']
        if page['files']['input.json'] != expected:
            raise ValueError('Page input changed; comparison would mix different inputs')
        result['comparison'] = compare_page(result, page, args.urdf)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('kind', 'collision_checked', 'endpoint_q_deg',
                                            'endpoint_error', 'returns_to_start', 'all_waypoints_within_limits')}))


if __name__ == '__main__':
    main()
