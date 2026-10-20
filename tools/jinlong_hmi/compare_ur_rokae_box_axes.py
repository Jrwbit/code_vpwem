#!/usr/bin/env python3
"""Read archived trials and project model differences onto junction-box axes.

No SDK, HTTP, robot commands, or production IK. Results use configured geometry
and are not independent physical measurements. Archive parameters stay outside Git.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from independent_route_audit import error, flange, link_transform


def read(path):
    return json.loads(Path(path).read_text())


def compare(comparison, site, controller, current, change, urdf):
    arm = controller['state']['raw']['left_arm']
    if any(config['robot_controller_id'] != arm['controller_id']
           for config in (site, current['site'])):
        raise ValueError('Controller identity differs between archives')
    cfg = arm['kinematic_config']
    dimensions = np.asarray(cfg['effective_dh']['alpha_a_d_theta_deg_mm']).reshape(8, 4)
    if not np.isfinite(dimensions).all():
        raise ValueError('Invalid archived effective DH')

    def base(waist):
        return link_transform(urdf, 'AR5-5_07L_base',
                              dict(zip((f'{i}-Joint' for i in range(1, 5)), waist)))

    def tooling(config, fixture):
        offset = np.array(config['tcp_T_objects'][fixture], dtype=float, copy=True)
        offset[2, 3] += config.get('fixture_z_adjustments_mm', {}).get(fixture, 0) / 1000
        z = np.eye(4)
        z[2, 3] = config.get('tcp_z_adjustment_mm', 0) / 1000
        return np.asarray(config['flange_T_tcp']) @ z @ offset

    fixture = site['selected_fixture']
    box_rotation = np.asarray(comparison['world_T_box'])[:3, :3]
    chain = tooling(site, fixture)
    trials = []
    for sample in comparison['settled']:
        if sample['position'] != 'target':
            continue
        actual = np.asarray(sample['left_mean_rad'])
        command = actual - np.radians(sample['left_target_error_deg'])
        world_base = base(sample['waist_mean_model_rad'])
        actual_pose = world_base @ flange(actual, dimensions) @ chain
        command_pose = world_base @ flange(command, dimensions) @ chain
        trials.append({'trial': sample['trial'],
                       'feedback_minus_command_box_xyz_mm':
                           (box_rotation.T @ (actual_pose[:3, 3]-command_pose[:3, 3])*1000).tolist(),
                       'fixture_feedback_vs_command': error(actual_pose, command_pose)})
    if not trials:
        raise ValueError('No target samples in archived comparison')
    config = current['site']
    if config['selected_fixture'] != fixture:
        raise ValueError('Archives describe different fixtures')
    chain = tooling(config, fixture)
    world_base = base(config['simulation_initial']['working_waist_model_rad'])
    calibration = current['calibration']
    world_box = (base(calibration['waist_model_rad'])
                 @ np.asarray(calibration['base_T_camera']) @ np.asarray(current['camera_T_box']))
    old = world_base @ flange(np.radians(change['old_target_deg']), dimensions) @ chain
    new = world_base @ flange(np.radians(change['new_target_deg']), dimensions) @ chain
    return {'robot_commands': False, 'fixture': fixture,
            'old_trials_effective_DH_feedback_vs_command': trials,
            'new_minus_old_DH_target_box_xyz_mm':
                (world_box[:3, :3].T @ (new[:3, 3]-old[:3, 3])*1000).tolist(),
            'new_minus_old_target': error(new, old),
            'limitation': 'Configured geometry only; archives use different observations. '
                          'Not a physical before/after trial or post-DH execution measurement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison', 'site', 'controller', 'current', 'change', 'urdf', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(*(read(getattr(args, name)) for name in
                       ('comparison', 'site', 'controller', 'current', 'change')), args.urdf)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
