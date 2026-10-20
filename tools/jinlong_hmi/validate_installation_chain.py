#!/usr/bin/env python3
"""Offline reconstruction of active/candidate hand-eye and mounting chains.

Reads saved GET/file snapshots only. Never connects to a robot or imports a
calibration. Internal closure cannot establish physical installation accuracy.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from independent_route_audit import error, flange, link_transform


def read(path):
    return json.loads(Path(path).read_text())


def validate(live, samples, sources, urdf, ur_frames, camera_box, solve):
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != live['urdf_sha256']:
        raise ValueError('URDF does not match the live snapshot')
    cal, site = live['calibration'], live['site']
    if cal.get('source_arm') != 'right_arm':
        raise ValueError('This audit requires the right-arm source calibration')

    def base(arm, waist):
        return link_transform(urdf, f'AR5-5_07{arm}_base',
                              dict(zip((f'{i}-Joint' for i in range(1, 5)), waist)))

    ref, work = cal['waist_model_rad'], site['simulation_initial']['working_waist_model_rad']
    mount = np.linalg.inv(base('L', ref)) @ base('R', ref)
    original = solve(samples[:cal['sample_count']])
    active = np.asarray(cal['base_T_camera'])
    rebuilt = mount @ np.asarray(original['T_base_cam'])
    reconstruction = error(rebuilt, active)
    if reconstruction['position_mm'] > 1e-6 or reconstruction['rotation_deg'] > 1e-6:
        raise ValueError('First archived samples do not reconstruct the active calibration')
    # Do not assume a mutable result.json is the source pinned by active metadata.
    pointers = [x['value'] for x in sources['selection_pointers']
                if x['value']['session_basename'] == Path(cal['source_session']).name]
    if len(pointers) != 1:
        raise ValueError('Missing or ambiguous current sample selection')
    indices = pointers[0]['active_indices']
    candidate = solve([samples[i] for i in indices])
    candidate_left = mount @ np.asarray(candidate['T_base_cam'])
    source_left = np.asarray(live['source_result']['T_base_cam'])
    source_reconstruction = error(candidate_left, source_left)
    if max(source_reconstruction['position_mm'], source_reconstruction['rotation_deg']) > 1e-6:
        raise ValueError('Current selected samples do not reconstruct result.json')

    parameters = yaml.safe_load(ur_frames.read_text())['rcell_tf_broadcaster']['ros__parameters']
    index = parameters['tool_names'].index(site['selected_fixture'])
    pose = parameters['tool_offsets_flat'][6*index:6*index+6]
    ur_tool = np.eye(4)
    ur_tool[:3, :3] = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    ur_tool[:3, 3] = pose[:3]
    fixture = np.asarray(site['tcp_T_objects'][site['selected_fixture']])

    world_camera = base('L', ref) @ active
    working_camera = np.linalg.inv(base('L', work)) @ world_camera
    old_box = world_camera @ camera_box
    new_box = base('L', ref) @ candidate_left @ camera_box
    point_delta = old_box[:3, :3].T @ (new_box[:3, 3]-old_box[:3, 3])*1000
    result = {
        'robot_writes': False, 'active_n': cal['sample_count'], 'selected_source_n': len(indices),
        'active_source_hash_matches_mutable_result':
            sources['current_source_sha256'] == sources['active_expected_source_sha256'],
        'active_reconstruction': reconstruction, 'selected_source_reconstruction': source_reconstruction,
        'active_internal_fit': {k: original[k] for k in ('method', 'trans_spread_mm', 'rot_spread_deg')},
        'candidate_internal_fit': {k: candidate[k] for k in ('method', 'trans_spread_mm', 'rot_spread_deg')},
        'ur_vs_rokae_fixture_matrix': error(fixture, ur_tool),
        'mount_invariance_between_waists': error(np.linalg.inv(base('L', work)) @ base('R', work), mount),
        'waist_transform_roundtrip': error(base('L', work) @ working_camera, world_camera),
        'candidate_vs_active_left_camera': error(candidate_left, active),
        'candidate_minus_active_box_center_old_box_axes_mm': point_delta.tolist(),
        'candidate_vs_active_box': error(new_box, old_box),
        'limitation': 'Internal closure and candidate differences only. The box observation is archived. '
                      'No independent metrology, new detection, robot movement or calibration activation.'}

    arm = live['state']['raw']['right_arm']
    dimensions = np.asarray(arm['kinematic_config']['effective_dh']['alpha_a_d_theta_deg_mm']).reshape(8, 4)
    measured_samples = []
    residuals = []
    for sample in samples[:cal['sample_count']]:
        if sample['controller_ip'] != arm['controller_ip']:
            raise ValueError('Calibration sample arm identity mismatch')
        pose = flange(sample['right_q_rad'], dimensions)
        residuals.append(error(pose, np.asarray(sample['T_base_flange'])))
        measured_samples.append({**sample, 'T_base_flange': pose.tolist()})
    measured = solve(measured_samples)
    measured_left = mount @ np.asarray(measured['T_base_cam'])
    measured_box = base('L', ref) @ measured_left @ camera_box
    result['historical_feedback_FK_recalibration_sensitivity'] = {
        'sample_flange_difference_range_mm': [min(x['position_mm'] for x in residuals),
                                             max(x['position_mm'] for x in residuals)],
        'camera_difference': error(measured_left, active),
        'box_center_delta_old_box_axes_mm':
            (old_box[:3, :3].T @ (measured_box[:3, 3]-old_box[:3, 3])*1000).tolist(),
        'assumption': 'Current effective DH also applied at historical sample time; sensitivity only'}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('live', 'samples', 'sources', 'urdf', 'ur-frames', 'box-input', 'control-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.control_root))
    from helios_hmi.handeye import solve_all
    result = validate(read(args.live), read(args.samples)['samples'], read(args.sources),
                      args.urdf, args.ur_frames, np.asarray(read(args.box_input)['camera_T_box']), solve_all)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
