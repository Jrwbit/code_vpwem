#!/usr/bin/env python3
"""Project a shared-flange layout through archived camera intrinsics.

Checks the conservative outer box envelope and a proposed 40 mm QR label
patch, one motor at a time. This is a geometric screen, not an RGB-D or QR
detection experiment. The label patch is a design proposal, not an observation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_layout_camera import screen


SIZES = {'JM160': [154.6, 149.4], 'JL132': [112., 112.], 'JL100': [103.5, 103.5]}
COLOURS = {'JM160': '#c76328', 'JL132': '#247e99', 'JL100': '#7554a6'}


def evaluate(candidate, intr):
    camera = np.asarray(candidate['world_T_camera'])
    results = {}
    for fid, pose in candidate['world_T_boxes'].items():
        T = np.linalg.inv(camera) @ pose
        half = np.array(SIZES[fid]) / 2000
        box = screen(intr, T, (-half, half), 100, [300, 700])
        bottom = T.copy()
        bottom[:3, 3] += bottom[:3, 2] * .09
        cavity = screen(intr, bottom, (-half, half), 100, [300, 700])
        labels = []
        for axis in (0, 1):
            for sign in (-1, 1):
                center = np.zeros(2)
                center[axis] = sign * (half[axis] + .030)
                lo, hi = center - .020, center + .020
                projection = screen(intr, T, (lo, hi), 100, [300, 700])
                labels.append({'side': ('X', 'Y')[axis] + ('+' if sign > 0 else '-'),
                               'bounds_box_mm': [(lo * 1000).tolist(), (hi * 1000).tolist()],
                               'screen': projection})
        label = max(labels, key=lambda x: x['screen']['minimum_pixel_border_margin'])
        results[fid] = {'camera_T_box': T.tolist(), 'outer_size_mm': SIZES[fid],
                        'rim': box, 'conservative_90mm_depth_envelope': cavity,
                        'proposed_QR_patch': label,
                        'projection_passed': box['projection_sample_gates_pass'] and
                        cavity['projection_sample_gates_pass'] and label['screen']['projection_sample_gates_pass']}
    return {'models': results, 'all_projection_gates_pass': all(r['projection_passed'] for r in results.values()),
            'world_T_camera': camera.tolist(), 'sampled_outline_points_per_plane': 400,
            'software_depth_window_mm': [300, 700], 'actual_detection_tested': False,
            'limitations': ['Outer envelope dimensions, not inferred free interior dimensions.',
                            'Only one motor occupies its indexed position at a time.',
                            'QR patches are proposed upward-facing 40mm labels, 10mm from the box outer edge.',
                            'Distortion-aware projection does not establish depth completeness, accuracy or QR decoding.',
                            'This script does not test arm occlusion; use the separate home-pose ray audit.',
                            'Camera relocation requires calibration of its new world pose.']}


def plot(candidate, result, intr, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    a, b = axes
    for fid, row in result['models'].items():
        color = COLOURS[fid]
        T = np.array(candidate['world_T_boxes'][fid])
        h = np.array(SIZES[fid]) / 2
        xy = np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]], [-h[0], -h[1]]])
        points = np.c_[xy, np.zeros(5)] @ T[:3, :3].T + T[:3, 3] * 1000
        a.plot(points[:, 0], points[:, 1], color=color, label=fid)
        a.text(T[0, 3]*1000, T[1, 3]*1000, fid, color=color, ha='center')
        for key, style in [('rim', '-'), ('proposed_QR_patch', '--')]:
            projection = row[key] if key == 'rim' else row[key]['screen']
            corners = np.array(projection['corner_pixels'])
            corners = np.vstack([corners, corners[0]])
            b.plot(corners[:, 0], corners[:, 1], style, color=color,
                   label=fid + (' rim' if key == 'rim' else ' proposed QR'))
    poses = np.array(candidate['path']['outward_flange_poses'])[:, :3, 3] * 1000
    a.plot(poses[:, 0], poses[:, 1], 'k.-', lw=1, label='Shared flange path')
    a.annotate('HOME / SCAN', poses[-1, :2], xytext=(6, 8), textcoords='offset points')
    camera = np.array(candidate['world_T_camera'])[:3, 3] * 1000
    a.plot(*camera[:2], 'rx', ms=10, label='Camera position (XY)')
    a.set(xlabel='World X (mm, robot front)', ylabel='World Y (mm)',
          title='Three indexed placements, one motor loaded at a time')
    a.axis('equal'); a.grid(alpha=.2); a.legend(loc='best', fontsize=8)
    b.set(xlim=(0, intr['width']), ylim=(intr['height'], 0), xlabel='Image u (px)', ylabel='Image v (px)',
          title='Archived intrinsics / proposed camera pose')
    b.set_aspect('equal'); b.grid(alpha=.2); b.legend(loc='best', fontsize=8)
    fig.suptitle('Fixed waist: ' + str(candidate['waist_model_deg']) + ' deg; actual tool mount Rz(-90 deg)')
    fig.text(.05, .02, 'Offline geometry only. Dashed patches are proposed label positions. '
             'Overlaid models do not occupy the station simultaneously.', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--intrinsics', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    candidate = json.loads(a.result.read_text())['candidate']
    intr = json.loads(a.intrinsics.read_text())
    result = evaluate(candidate, intr)
    result['sources_sha256'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (a.result, a.intrinsics, Path(__file__))}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'camera_screen.json').write_text(json.dumps(result, indent=2) + '\n')
    plot(candidate, result, intr, a.output/'layout_camera.png')
    print(json.dumps({'all_projection_gates_pass': result['all_projection_gates_pass'],
                      'rim_pixel_margin': {f: r['rim']['minimum_pixel_border_margin']
                                           for f, r in result['models'].items()}}))


if __name__ == '__main__':
    main()
