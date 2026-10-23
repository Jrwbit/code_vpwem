#!/usr/bin/env python3
"""Substitute an archived fixed-camera calibration into a saved station cycle.

Offline only: preserves motor poses, waist, home and every saved arm waypoint.
Does not search alternative stations or establish physical camera installation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from screen_shared_station_camera import digest, evaluate_measured_depths, plot, rebuild


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('result', 'calibration', 'intrinsics', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    result = json.loads(args.result.read_text())
    if not result['full_common_home_cycles_verified']:
        raise ValueError('Requires verified saved common-home cycles')
    calibration = json.loads(args.calibration.read_text())
    if calibration['base_frame'] != 'rokae_left_sdk_base' or calibration['translation_unit'] != 'm':
        raise ValueError('Requires left-base camera calibration')
    sys.path.insert(0, result['run_arguments']['control_root'])
    from helios_hmi import urdf_kin
    kin = urdf_kin.UrdfKinematics(urdf_kin.default_urdf_path())
    # A fixed external camera uses the calibration reference waist, not the
    # candidate working waist. Moving the waist must not move this world pose.
    world_camera = kin.link_pose('AR5-5_07L_base', dict(zip(
        urdf_kin.WAIST_JOINTS, calibration['waist_model_rad']))) @ np.asarray(calibration['base_T_camera'])
    candidate = copy.deepcopy(result['candidate'])
    designed_camera = np.asarray(candidate['world_T_camera'])
    candidate['world_T_camera'] = world_camera.tolist()
    intrinsics = json.loads(args.intrinsics.read_text())
    geometry_files = {}
    for fid in candidate['world_T_boxes']:
        matches = [Path(p) for p in result['geometry_sha256_at_build'] if Path(p).name == fid + '_collision.json']
        if len(matches) != 1 or digest(matches[0]) != result['geometry_sha256_at_build'][str(matches[0])]:
            raise ValueError('Validated geometry missing or changed: ' + fid)
        geometry_files[fid] = matches[0]
    screen = evaluate_measured_depths(candidate, intrinsics, geometry_files)
    qr_path = args.output / 'camera_projection.json'
    qr_path.write_text(json.dumps(screen, indent=2) + '\n')
    study = rebuild(result, args.output / 'scenes')
    study.args.qr_layout = qr_path
    boxes = {f: np.asarray(t) for f, t in candidate['world_T_boxes'].items()}
    waist = np.radians(candidate['waist_model_deg'])
    candidate['path'] = {'home_arm_rad': candidate['common_home_validation']['common_home_arm_rad']}
    cycles, visibility = {}, {}
    for fid in boxes:
        ctx = study.build(fid, boxes, world_camera, args.output / 'scenes' / fid)
        cycles[fid] = study.validator.validate_cycle(ctx, candidate['paths'][fid]['common_home_cycle_segments'], fid, waist)
        visibility[fid] = study.home_camera_visibility({fid: ctx}, waist, candidate)['fixtures'][fid]
    for path, actual in study.geometry_hashes.items():
        expected = result['geometry_sha256_at_build'].get(path)
        if expected is None and Path(path).name == 'rotated_tool_hulls.json':
            suffix = '/' + '/'.join(Path(path).parts[-2:])
            matches = [v for k, v in result['geometry_sha256_at_build'].items() if k.endswith(suffix)]
            expected = matches[0] if len(matches) == 1 else None
        if expected != actual:
            raise ValueError('Collision geometry differs from saved cycle: ' + path)
    source_files = [args.result, args.calibration, args.intrinsics, Path(__file__),
                    Path(urdf_kin.__file__), urdf_kin.default_urdf_path()]
    sources = {str(Path(p).resolve()): digest(p) for p in source_files}
    for path, expected in study.loaded_source_hashes.items():
        if digest(path) != expected:
            raise ValueError('Source changed during audit: ' + path)
        sources[path] = expected
    delta = (designed_camera[:3, 3] - world_camera[:3, 3]) * 1000
    rotation = world_camera[:3, :3].T @ designed_camera[:3, :3]
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Same station, waist, common home and saved waypoints; only camera substituted',
        'world_T_calibrated_camera': world_camera.tolist(),
        'world_T_designed_camera': designed_camera.tolist(),
        'designed_minus_calibrated_translation_mm': delta.tolist(),
        'translation_difference_mm': float(np.linalg.norm(delta)),
        'full_orientation_difference_deg': float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))),
        'calibrated_optical_axis_tilt_from_down_deg': float(np.degrees(np.arccos(np.clip(world_camera[:3, 2] @ [0, 0, -1], -1, 1)))),
        'calibration_independent_validation': calibration['independent_validation'],
        'projection': screen, 'common_home_visibility': visibility, 'same_saved_cycles': cycles,
        'all_projection_gates_pass': screen['all_projection_gates_pass'],
        'all_home_sample_rays_clear': not any(r['blocked_samples'] for r in visibility.values()),
        'all_same_saved_cycles_pass': all(r['ok'] for r in cycles.values()),
        'physical_acceptance': False,
        'limitations': [
            'Saved calibration is not a new measurement of the current physical mount.',
            'Failure of these exact saved paths does not prove a fixed-camera station impossible.',
            'Ray tests cover robot/tool at common home; QR locations are proposals, no actual decoding.',
            'Motor foot-to-observation registration and physical insertion remain unvalidated.',
        ],
        'sources_sha256': sources,
    }
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    plot(candidate, screen, intrinsics, args.output / 'calibrated_camera_projection.png')
    print(json.dumps({k: report[k] for k in (
        'designed_minus_calibrated_translation_mm', 'translation_difference_mm',
        'all_projection_gates_pass', 'all_home_sample_rays_clear', 'all_same_saved_cycles_pass')}))


if __name__ == '__main__':
    main()
