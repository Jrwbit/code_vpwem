#!/usr/bin/env python3
"""Offline projection screen for a layout study; never opens a camera or robot.

Inputs are archived camera intrinsics and C_T_box. The box outline is an assumed
planar envelope, not a measured box or a substitute for RGB-D quality checks.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np


DISTORTION_KEYS = ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6',
                   's1', 's2', 's3', 's4')


def transform(value):
    T = np.asarray(value, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-6)):
        raise ValueError('Expected a finite rigid 4x4 transform, translation in metres')
    return T


def outline_bounds(scene, size_mm):
    if scene is None:
        half = np.asarray(size_mm, dtype=float) / 2000
        if not np.isfinite(half).all() or np.any(half <= 0):
            raise ValueError('Outline dimensions must be positive finite millimetres')
        return -half, half, 'Explicit assumed outer rectangle; default 155 x 155 mm'
    # Historical scene key stores W_T_box despite its reversed-looking name.
    box_T_world = np.linalg.inv(transform(scene['box_T_base_link']))
    corners = []
    walls = [o for o in scene['obstacles'] if o['name'].startswith('box_wall_')]
    if len(walls) != 4:
        raise ValueError('Scene must contain the four box_wall_ obstacles')
    for wall in walls:
        size = np.asarray(wall['size'], dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError('Scene wall size must contain three positive dimensions')
        local = np.asarray(list(itertools.product((-1, 1), repeat=3))) * size / 2
        T = box_T_world @ transform(wall['T'])
        corners.extend((local @ T[:3, :3].T + T[:3, 3])[:, :2])
    corners = np.asarray(corners)
    return corners.min(axis=0), corners.max(axis=0), (
        'XY envelope of scene wall solids, projected on box z=0; scene geometry is assumed')


def screen(intrinsics, camera_T_box, bounds, samples_per_edge, depth_mm):
    width, height = int(intrinsics['width']), int(intrinsics['height'])
    fx, fy, cx, cy = [float(intrinsics[k]) for k in ('fx', 'fy', 'cx', 'cy')]
    if (width <= 0 or height <= 0 or fx <= 0 or fy <= 0
            or not np.isfinite([fx, fy, cx, cy]).all()):
        raise ValueError('Invalid image dimensions or camera intrinsics')
    distortion = np.asarray([intrinsics['dist_coeffs'][k] for k in DISTORTION_KEYS], float)
    if not np.isfinite(distortion).all():
        raise ValueError('Distortion coefficients must be finite')
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    lo, hi = bounds
    corners = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0],
                        [hi[0], hi[1], 0], [lo[0], hi[1], 0]])
    # Excluding each edge endpoint avoids duplicating its next corner.
    t = np.linspace(0, 1, samples_per_edge, endpoint=False)[:, None]
    points = np.concatenate([a + t * (b - a)
                             for a, b in zip(corners, np.roll(corners, -1, axis=0))])
    camera_points = points @ camera_T_box[:3, :3].T + camera_T_box[:3, 3]
    z = camera_points[:, 2] * 1000
    if np.any(z <= 0):
        return {'sample_count': len(points), 'positive_optical_depth': False,
                'depth_range_mm': [float(z.min()), float(z.max())],
                'projection_sample_gates_pass': False,
                'reason': 'At least one outline sample is behind or on the camera plane'}
    pixels = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), K, distortion)[0][:, 0]
    if not np.isfinite(pixels).all():
        raise ValueError('Projection contains nonfinite image coordinates')
    margins = np.column_stack([pixels[:, 0], width - 1 - pixels[:, 0],
                               pixels[:, 1], height - 1 - pixels[:, 1]])
    inside = bool(np.all(margins >= 0))
    depth_ok = bool(np.all((z >= depth_mm[0]) & (z <= depth_mm[1])))
    return {'sample_count': len(points), 'positive_optical_depth': True,
            'depth_range_mm': [float(z.min()), float(z.max())],
            'all_samples_inside_software_depth_window': depth_ok,
            'all_samples_inside_image': inside,
            'minimum_pixel_border_margin': float(margins.min()),
            'pixel_bounds_xy_min_max': [pixels.min(axis=0).tolist(), pixels.max(axis=0).tolist()],
            'corner_pixels': pixels[::samples_per_edge].tolist(),
            'projection_sample_gates_pass': inside and depth_ok}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intrinsics', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True,
                        help='Archived issue_evidence.json containing camera_T_box')
    parser.add_argument('--camera-T-box', type=Path,
                        help='Optional JSON 4x4 C_T_box override, translation in metres')
    parser.add_argument('--scene', type=Path, help='Derive outline XY bounds from scene wall geometry')
    parser.add_argument('--outline-size-mm', nargs=2, type=float, default=[155., 155.],
                        help='Assumed outer X Y dimensions when --scene is absent')
    parser.add_argument('--software-depth-mm', nargs=2, type=float, default=[300., 700.])
    parser.add_argument('--samples-per-edge', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.samples_per_edge < 2 or not np.isfinite(args.software_depth_mm).all()
            or not 0 < args.software_depth_mm[0] < args.software_depth_mm[1]):
        parser.error('Use at least two samples per edge and an increasing positive depth window')
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    intrinsics, evidence = read(args.intrinsics), read(args.evidence)
    C_T_box = transform(read(args.camera_T_box) if args.camera_T_box else evidence['camera_T_box'])
    lo, hi, geometry_note = outline_bounds(read(args.scene) if args.scene else None, args.outline_size_mm)
    # Nonidentity rigid move proves the camera-to-box invariant numerically.
    D = np.eye(4)
    D[:3, :3] = cv2.Rodrigues(np.array([.1, -.2, .3]))[0]
    D[:3, 3] = [.18, -.11, .04]
    invariant_error = float(np.max(np.abs(np.linalg.inv(D) @ (D @ C_T_box) - C_T_box)))
    result = {
        'execution_authorized': False, 'executable': False,
        'input_sha256': inputs, 'camera_T_box': C_T_box.tolist(),
        'image_size_px': [intrinsics['width'], intrinsics['height']],
        'distortion_coefficient_order': DISTORTION_KEYS,
        'software_depth_window_mm': args.software_depth_mm,
        'outline_bounds_box_xy_mm': [(lo * 1000).tolist(), (hi * 1000).tolist()],
        'outline_plane_box_z_mm': 0, 'outline_assumption': geometry_note,
        'screen': screen(intrinsics, C_T_box, (lo, hi), args.samples_per_edge, args.software_depth_mm),
        'rigid_camera_box_move': {
            'identity': 'inverse(D @ W_T_C) @ (D @ W_T_box) = C_T_box',
            'numerical_check_max_abs_error': invariant_error,
            'scope': 'Same rigid transform for camera and box preserves projection and optical depth; '
                     'the robot-camera calibration must be updated or independently verified after relocation'},
        'limitations': [
            'Archived inputs; timestamps and calibration compatibility require review',
            'Software depth window is not the camera hardware measurement specification',
            'Assumed planar outline and sampled edges; no measured box shape or interior depth guarantee',
            'Does not model robot occlusion, QR label visibility, reflectivity, or actual RGB-D detection quality',
            'No robot reachability, collision, trajectory, or physical motion acceptance'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['screen'], ensure_ascii=False))


if __name__ == '__main__':
    main()
