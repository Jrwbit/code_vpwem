#!/usr/bin/env python3
"""Offline optical screen for a camera lifted along laboratory Z only.

Retains camera XY/orientation, robot installation, shared station, waist and
the three measured box envelopes. Does not open a camera or operate a robot.
The depth window is configurable; the historical 300--700 mm default is not
a camera hardware specification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from screen_shared_station_camera import evaluate_measured_depths
from screen_cycle_camera import (validate_depth_window, HISTORICAL_DEPTH_WINDOW_MM,
                                 DEPTH_WINDOW_HELP)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lift_in_lab(candidate, lift_mm):
    """Legacy world keys use the model frame; lift in laboratory coordinates."""
    lifted = copy.deepcopy(candidate)
    lab_T_robot = np.asarray(candidate['frame_metadata']['lab_T_robot'], float)
    original = lab_T_robot @ np.asarray(candidate['world_T_camera'], float)
    camera = original.copy()
    camera[2, 3] += lift_mm / 1000
    lifted['world_T_camera'] = (np.linalg.inv(lab_T_robot) @ camera).tolist()
    lifted['frame_metadata']['lab_T_camera'] = camera.tolist()
    lifted['frame_metadata']['camera_fixed_in_lab'] = bool(lift_mm == 0)
    lifted['frame_metadata']['physical_camera_movement_mm'] = abs(lift_mm)
    lifted['frame_metadata']['physical_camera_rotation_deg'] = 0.0
    lifted['frame_metadata']['camera_model_transform_change_is_coordinate_change_only'] = bool(lift_mm == 0)
    return lifted, camera


def projection_summary(screen):
    lower, upper = validate_depth_window(screen['software_depth_window_mm'])
    rows = {}
    for fid, model in screen['models'].items():
        planes = [model['rim'], model['conservative_geometry_depth_envelope'],
                  model['proposed_QR_patch']['screen']]
        maximum = max(plane['depth_range_mm'][1] for plane in planes)
        minimum = min(plane['depth_range_mm'][0] for plane in planes)
        rows[fid] = {
            'projection_passed': model['projection_passed'],
            'rim_depth_range_mm': model['rim']['depth_range_mm'],
            'bottom_envelope_depth_range_mm': model['conservative_geometry_depth_envelope']['depth_range_mm'],
            'maximum_optical_depth_mm': maximum,
            'minimum_optical_depth_mm': minimum,
            'remaining_upper_depth_window_mm': upper - maximum,
            'remaining_lower_depth_window_mm': minimum - lower,
            'all_outlines_inside_image': all(plane['all_samples_inside_image'] for plane in planes),
            'minimum_pixel_border_margin': min(plane['minimum_pixel_border_margin'] for plane in planes),
            'proposed_qr_side': model['proposed_QR_patch']['side'],
        }
    return rows


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    steps = result['evaluated_lifts']
    x = [step['lift_lab_z_mm'] for step in steps]
    for fid, colour in [('JM160', '#c76328'), ('JL132', '#247e99'), ('JL100', '#7554a6')]:
        axes[0].plot(x, [step['models'][fid]['maximum_optical_depth_mm'] for step in steps],
                     'o-', color=colour, label=fid + ' maximum')
        axes[0].plot(x, [step['models'][fid]['minimum_optical_depth_mm'] for step in steps],
                     ':', color=colour, label=fid + ' minimum')
        axes[1].plot(x, [step['models'][fid]['minimum_pixel_border_margin'] for step in steps],
                     'o-', color=colour, label=fid)
    lower, upper = result['software_depth_window_mm']
    axes[0].axhline(upper, color='red', linestyle='--', label=f'Screen upper bound {upper:g} mm')
    axes[0].axhline(lower, color='red', linestyle='-.', label=f'Screen lower bound {lower:g} mm')
    axes[0].axvline(result['depth_lift_limit']['maximum_theoretical_lift_mm'], color='gray', linestyle=':')
    axes[0].set(ylabel='Sampled optical depth range (mm)', title='Rim / geometry bottom / proposed QR')
    axes[1].set(ylabel='Smallest image border margin (px)', title='All three outlined targets stay in image')
    for axis in axes:
        axis.set_xlabel('Camera lift along laboratory Z (mm)')
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.suptitle('Same shared station, waist and camera XY/orientation')
    fig.text(.06, .025, 'Archived geometry and intrinsics only; no RGB-D capture, occlusion or motion acceptance.', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--intrinsics', type=Path, required=True)
    parser.add_argument('--lifts-mm', type=float, nargs='+', default=[0, 40, 60, 80, 100, 110, 120, 150])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth-window-mm', nargs=2, type=float, default=HISTORICAL_DEPTH_WINDOW_MM,
                        help=DEPTH_WINDOW_HELP)
    args = parser.parse_args()
    depth_window_mm = validate_depth_window(args.depth_window_mm)
    if not np.isfinite(args.lifts_mm).all() or any(v < 0 for v in args.lifts_mm):
        parser.error('Camera lifts must be finite and nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.result.read_text())
    candidate = source['candidate']
    intr = json.loads(args.intrinsics.read_text())
    geometry_files = {}
    for fid in candidate['world_T_boxes']:
        matches = [Path(p) for p in source['geometry_sha256_at_build'] if Path(p).name == fid + '_collision.json']
        if len(matches) != 1 or digest(matches[0]) != source['geometry_sha256_at_build'][str(matches[0])]:
            raise ValueError('Expected one unchanged registered collision geometry for ' + fid)
        geometry_files[fid] = matches[0]

    baseline = evaluate_measured_depths(candidate, intr, geometry_files, depth_window_mm)
    _, lab_T_camera = lift_in_lab(candidate, 0)
    slope = -float(lab_T_camera[2, 2])
    if slope <= 0:
        raise ValueError('This upper-depth-limit analysis requires a downward-facing camera')
    baseline_summary = projection_summary(baseline)
    limiting_fid = max(baseline_summary, key=lambda fid: baseline_summary[fid]['maximum_optical_depth_mm'])
    deepest = baseline_summary[limiting_fid]['maximum_optical_depth_mm']
    depth_limit = (depth_window_mm[1] - deepest) / slope
    steps = []
    for lift in sorted(set(args.lifts_mm)):
        lifted, camera = lift_in_lab(candidate, lift)
        screen = evaluate_measured_depths(lifted, intr, geometry_files, depth_window_mm)
        filename = f'lift_{lift:g}mm_projection.json'
        (args.output / filename).write_text(json.dumps(screen, indent=2, allow_nan=False) + '\n')
        steps.append({
            'lift_lab_z_mm': lift,
            'lab_T_camera': camera.tolist(),
            'world_T_camera': lifted['world_T_camera'],
            'all_projection_gates_pass': screen['all_projection_gates_pass'],
            'models': projection_summary(screen),
            'full_projection_result': filename,
        })

    # Verify the analytic depth limit numerically just below and above it.
    checks = []
    for offset in [-.001, .001]:
        lifted, _ = lift_in_lab(candidate, depth_limit + offset)
        result = evaluate_measured_depths(lifted, intr, geometry_files, depth_window_mm)
        largest = max(row['maximum_optical_depth_mm'] for row in projection_summary(result).values())
        checks.append({'lift_mm': depth_limit + offset, 'maximum_depth_mm': largest,
                       'all_projection_gates_pass': result['all_projection_gates_pass']})
    if not checks[0]['maximum_depth_mm'] < depth_window_mm[1] < checks[1]['maximum_depth_mm']:
        raise AssertionError('Depth-limit analytic and numerical calculations disagree')

    paths = [args.result, args.intrinsics, Path(__file__),
             Path(__file__).with_name('screen_shared_station_camera.py'),
             Path(__file__).with_name('screen_cycle_camera.py'),
             Path(__file__).with_name('check_layout_camera.py'), *geometry_files.values()]
    output = {
        'scope': 'Offline camera vertical-lift optical screen; no route search or hardware actions',
        'controller_writes': False,
        'camera_capture_performed': False,
        'physical_camera_moved': False,
        'camera_motion_is_design_only': True,
        'all_other_layout_and_camera_orientation_values_unchanged': True,
        'lab_T_camera_baseline': lab_T_camera.tolist(),
        'shared_station_world_T': candidate['world_T_station'],
        'waist_model_deg': candidate['waist_model_deg'],
        'software_depth_window_mm': depth_window_mm,
        'depth_window_note': DEPTH_WINDOW_HELP,
        'evaluated_lifts': steps,
        'depth_lift_limit': {
            'maximum_theoretical_lift_mm': depth_limit,
            'limiting_model': limiting_fid,
            'baseline_largest_optical_depth_mm': deepest,
            'depth_increase_per_mm_lab_z_lift': slope,
            'formula': '(configured upper depth bound - baseline largest optical depth) / (-lab_T_camera[2,2])',
            'numerical_boundary_checks': checks,
            'scope': 'Depth-window boundary only, with zero reserve; not an installation recommendation',
        },
        'optical_candidate_note': 'Compare the remaining depth-window reserve at each passing lift; route feasibility must be checked separately.',
        'limitations': [
            'The selected depth window is an offline screening input; the historical 300--700 mm default is not a camera hardware specification.',
            'Optical depth is measured along camera Z, not the vertical lens-to-table distance.',
            'A visible projection is not a guarantee of real depth completeness, QR detection or measurement accuracy.',
            'QR patch location is proposed; target self-occlusion and wire obstacles are not measured.',
            'Robot/tool occlusion and route collision are separate checks not performed by this script.',
            'Nominal CAD foot-to-observed-box registration remains physically unaccepted.',
            'Any physical camera movement requires updated extrinsics and independent validation.',
        ],
        'source_sha256': {str(path.resolve()): digest(path) for path in paths},
    }
    (args.output / 'camera_vertical_lift_optics.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    plot(output, args.output / 'camera_vertical_lift_optics.png')
    print(json.dumps({'depth_lift_limit': output['depth_lift_limit'],
                      'passing_lifts_mm': [s['lift_lab_z_mm'] for s in steps if s['all_projection_gates_pass']]}))


if __name__ == '__main__':
    main()
