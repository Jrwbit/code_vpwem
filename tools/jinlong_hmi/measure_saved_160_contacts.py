#!/usr/bin/env python3
"""Measure manually identified features in one archived 160 RGB-D frame.

The independently fitted rim from audit_saved_box_scan.py defines Z.  The
visible outer-edge envelope defines only a provisional XY frame; it is not a
measurement of the mounting-hole datum. No camera or robot is accessed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


CENTRES = [(1300, 794), (1485, 979), (1660, 1161), (1675, 438),
           (1850, 620), (2015, 808), (1873, 1312)]
DISTORTION_KEYS = ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6',
                   's1', 's2', 's3', 's4')


def measure(scan, audit_path):
    if scan.name != '20261022_110619_cb9e44':
        raise ValueError('Manual ROIs apply only to 20261022_110619_cb9e44')
    intr = json.loads((scan / 'camera_intrinsics.json').read_text())
    audit = json.loads(audit_path.read_text())
    depth = np.load(scan / '03_depth_raw.npy')
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    distortion = np.array([intr['dist_coeffs'][k] for k in DISTORTION_KEYS])

    def rays(uv):
        xy = cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), K, distortion)[:, 0]
        return np.c_[xy, np.ones(len(xy))]

    plane = audit['independent_rim_plane']
    n = np.array(plane['normal'])
    corners = rays(audit['manual_visible_edge_estimates']['outer']['virtual_corner_pixels'])
    corners *= (-plane['d_m'] / (corners @ n))[:, None]
    x = (corners[0] - corners[1] + corners[3] - corners[2]) / 2
    x -= n * (x @ n)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(n, x), n]
    T[:3, 3] = corners.mean(0)
    features = []
    for i, (u, v) in enumerate(CENTRES):
        # Strict radius reproduces the original 437-pixel measurement.
        yy, xx = np.mgrid[v-12:v+13, u-12:u+13]
        mask = (xx-u)**2 + (yy-v)**2 < 12**2
        mask &= depth[yy, xx] > 0
        xx, yy = xx[mask], yy[mask]
        camera = rays(np.c_[xx, yy]) * (depth[yy, xx] * intr['depth_unit_raw'] / 1000)[:, None]
        box = (camera - T[:3, 3]) @ T[:3, :3] * 1000
        features.append({'id': f'stud_{i+1}' if i < 6 else 'ground',
                         'pixel_center': [u, v], 'radius_px': 12,
                         'xyz_box_mm': np.median(box, axis=0).tolist(),
                         'xyz_camera_m': np.median(camera, axis=0).tolist(),
                         'depth_percentile_5_50_95_mm': np.percentile(box[:, 2], [5, 50, 95]).tolist(),
                         'valid_pixels': len(box)})
    return {'camera_T_box': T.tolist(), 'frame':
            'origin = independently measured outer rim straight-edge envelope centre; '
            'x towards image left, y towards image up, z into box',
            'features': features, 'source_scan': str(scan.resolve()), 'rim_plane': plane,
            'manual_roi_estimates': True, 'ground_cap_wide_roi_depth_median_mm':
            audit['ground_screw_cap_roi']['below_rim_plane_mm_percentiles_5_50_95'][1],
            'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [scan/'03_depth_raw.npy', scan/'camera_intrinsics.json', audit_path]},
            'notes': ['Manual approximate centres, not independent metrology or submillimetre hole-axis fits',
                      'Ground centre ROI includes Phillips recess; broad cap ROI is a separate surface estimate',
                      'No production calibration or target changed']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = measure(a.scan, a.audit)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'features': len(result['features']), 'output': str(a.output)}))


if __name__ == '__main__':
    main()
