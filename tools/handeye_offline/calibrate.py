#!/usr/bin/env python3
"""Offline hand-eye calibration. No device SDK or network connections."""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np


def matrix(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Transform must be a finite 4x4 matrix')
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('Invalid homogeneous transform last row')
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError('Rotation must be orthonormal with determinant +1')
    return t


def rt(r, t):
    result = np.eye(4)
    result[:3, :3] = r
    result[:3, 3] = np.asarray(t).reshape(3)
    return matrix(result)


def delta(a, b):
    d = np.linalg.inv(a) @ b
    angle = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
    return {'translation_mm': float(np.linalg.norm(d[:3, 3]) * 1000), 'rotation_deg': float(angle)}


def summary(items):
    return {key: {stat: float(fn([v[key] for v in items])) for stat, fn in
                 [('mean', np.mean), ('max', np.max), ('p95', lambda x: np.percentile(x, 95))]}
            for key in ['translation_mm', 'rotation_deg']}


def coverage(poses):
    vectors = [cv2.Rodrigues((np.linalg.inv(a) @ b)[:3, :3])[0].reshape(3)
               for a, b in itertools.combinations(poses, 2)]
    singular = np.linalg.svd(np.asarray(vectors), compute_uv=False)
    if singular[0] < np.radians(5) or singular[1] < .05 * singular[0]:
        raise ValueError('Insufficient rotation diversity: rotate around at least two different axes')
    return singular.tolist()


def solve(samples, mode, method):
    # OpenCV solves A X = Z C. Fixed camera: X=tcp_T_board, Z=base_T_camera.
    # Mounted camera uses A=inv(base_T_tcp): X=base_T_board, Z=tcp_T_camera.
    a = [matrix(s['base_T_tcp']) for s in samples]
    if mode == 'eye-in-hand':
        a = [np.linalg.inv(t) for t in a]
    c = [matrix(s['camera_T_board']) for s in samples]
    coverage(a)
    rx, tx, rz, tz = cv2.calibrateRobotWorldHandEye(
        [t[:3, :3] for t in a], [t[:3, 3].reshape(3, 1) for t in a],
        [t[:3, :3] for t in c], [t[:3, 3].reshape(3, 1) for t in c], method=method)
    x, z = rt(rx, tx), rt(rz, tz)
    errors = [dict(id=s['id'], **delta(aa @ x, z @ cc)) for s, aa, cc in zip(samples, a, c)]
    names = ('tcp_T_board', 'base_T_camera') if mode == 'eye-to-hand' else ('base_T_board', 'tcp_T_camera')
    return {names[0]: x.tolist(), names[1]: z.tolist(), 'residuals': errors,
            'residual_summary': summary(errors)}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def load_samples(path):
    path = Path(path).resolve()
    data = read_json(path)
    if data.get('translation_unit') != 'm' or data.get('timestamp_unit') != 's':
        raise ValueError('Declare translation_unit=m and timestamp_unit=s; matrices map source into target')
    if data.get('mode') not in ['eye-to-hand', 'eye-in-hand']:
        raise ValueError('Select mode eye-to-hand or eye-in-hand explicitly')
    limits = data['limits']
    for key in ['max_time_delta_s', 'max_reprojection_rms_px', 'max_fit_translation_mm', 'max_fit_rotation_deg']:
        if not np.isfinite(float(limits[key])) or float(limits[key]) <= 0:
            raise ValueError(f'Invalid positive limit: {key}')
    if not isinstance(limits['min_samples'], int) or limits['min_samples'] < 10:
        raise ValueError('min_samples must be an integer >=10')
    board_cfg = data['board']
    board = cv2.aruco.CharucoBoard((board_cfg['squares_x'], board_cfg['squares_y']),
        board_cfg['square_size_m'], board_cfg['marker_size_m'],
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board_cfg['dictionary'])))
    if not 0 < board_cfg['marker_size_m'] < board_cfg['square_size_m']:
        raise ValueError('Require 0 < marker_size_m < square_size_m')
    intr = data['intrinsics']
    k = np.asarray(intr['camera_matrix'], dtype=float)
    dist = np.asarray(intr['distortion_coefficients'], dtype=float)
    if not isinstance(intr['width'], int) or not isinstance(intr['height'], int) or min(intr['width'], intr['height']) <= 0:
        raise ValueError('Replace placeholder intrinsics with real positive image dimensions')
    if k.shape != (3, 3) or not np.isfinite(k).all() or not np.isfinite(dist).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Invalid camera intrinsics')
    accepted, rejected, ids = [], [], set()
    for s in data['samples']:
        ident = s['id']
        if ident in ids:
            raise ValueError(f'Duplicate sample id: {ident}')
        ids.add(ident)
        try:
            a = matrix(s['base_T_tcp'])
            times = [float(s['image_timestamp_s']), float(s['pose_timestamp_s'])]
            if not np.isfinite(times).all() or abs(times[0] - times[1]) > limits['max_time_delta_s']:
                raise ValueError('Image/pose timestamp mismatch')
            if s.get('robot_stationary') is not True:
                raise ValueError('Owner must confirm robot_stationary=true')
            if 'camera_T_board' in s:
                c = matrix(s['camera_T_board'])
                error = float(s['reprojection_rms_px'])
            else:
                image_path = (path.parent / s['image']).resolve()
                if not image_path.is_relative_to(path.parent):
                    raise ValueError('Image must be inside the dataset directory')
                image = cv2.imread(str(image_path))
                if image is None or image.shape[1] != intr['width'] or image.shape[0] != intr['height']:
                    raise ValueError('Image missing or resolution differs from intrinsics')
                corners, corner_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                if corner_ids is None or len(corner_ids) < 8:
                    raise ValueError('Fewer than 8 detected ChArUco corners')
                obj = board.getChessboardCorners()[corner_ids.flatten()].astype(float)
                pts = corners.reshape(-1, 2).astype(float)
                ok, rv, tv = cv2.solvePnP(obj, pts, k, dist)
                if not ok or tv[2, 0] <= 0:
                    raise ValueError('PnP failed or board is behind camera')
                projected = cv2.projectPoints(obj, rv, tv, k, dist)[0].reshape(-1, 2)
                error = float(np.sqrt(np.mean(np.sum((projected - pts) ** 2, axis=1))))
                c = rt(cv2.Rodrigues(rv)[0], tv)
            if not np.isfinite(error) or error < 0 or error > limits['max_reprojection_rms_px']:
                raise ValueError('Reprojection error exceeds limit')
            accepted.append(dict(id=ident, base_T_tcp=a.tolist(), camera_T_board=c.tolist(), reprojection_rms_px=error))
        except (ValueError, KeyError, cv2.error) as exc:
            rejected.append({'id': ident, 'reason': str(exc)})
    return data, accepted, rejected


def run(path, output):
    data, samples, rejected = load_samples(path)
    if len(samples) < data['limits']['min_samples']:
        write_json(output, {'ready': False, 'accepted': len(samples), 'rejected': rejected, 'reason': 'Too few valid samples'})
        return 2
    fits = {name: solve(samples, data['mode'], method) for name, method in
            [('shah', cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH), ('li', cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)]}
    held = []
    for i, sample in enumerate(samples):
        fit = solve(samples[:i] + samples[i+1:], data['mode'], cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
        a, c = matrix(sample['base_T_tcp']), matrix(sample['camera_T_board'])
        if data['mode'] == 'eye-to-hand':
            held.append(delta(a @ matrix(fit['tcp_T_board']), matrix(fit['base_T_camera']) @ c))
        else:
            held.append(delta(np.linalg.inv(a) @ matrix(fit['base_T_board']), matrix(fit['tcp_T_camera']) @ c))
    names = ['base_T_camera', 'tcp_T_board'] if data['mode'] == 'eye-to-hand' else ['tcp_T_camera', 'base_T_board']
    sensitivity = {name: delta(matrix(fits['shah'][name]), matrix(fits['li'][name])) for name in names}
    limits = data['limits']
    checks = [fit['residual_summary'] for fit in fits.values()] + [summary(held)]
    ready = all(x['translation_mm']['max'] <= limits['max_fit_translation_mm'] and
                x['rotation_deg']['max'] <= limits['max_fit_rotation_deg'] for x in checks)
    ready = ready and all(x['translation_mm'] <= limits['max_fit_translation_mm'] and
                          x['rotation_deg'] <= limits['max_fit_rotation_deg'] for x in sensitivity.values())
    write_json(output, {'ready': ready, 'mode': data['mode'], 'accepted': len(samples), 'rejected': rejected,
        'translation_unit': 'm', 'fits': fits, 'leave_one_out': summary(held), 'solver_difference': sensitivity,
        'rotation_coverage_singular_values': coverage([matrix(s['base_T_tcp']) for s in samples]),
        'scope': 'Internal closure validation only; physical accuracy requires independent measurements. No device configuration was changed.'})
    return 0 if ready else 2


def check_environment():
    required = ['CharucoBoard', 'CharucoDetector', 'getPredefinedDictionary']
    missing = [name for name in required if not hasattr(getattr(cv2, 'aruco', None), name)]
    if not hasattr(cv2, 'calibrateRobotWorldHandEye'):
        missing.append('calibrateRobotWorldHandEye')
    print(json.dumps({'python': sys.version, 'opencv': cv2.__version__, 'numpy': np.__version__,
                      'missing': missing, 'device_access': False}, indent=2))
    return bool(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check-environment')
    solve_parser = sub.add_parser('solve')
    solve_parser.add_argument('--input', required=True, type=Path)
    solve_parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'check-environment':
        return check_environment()
    if args.input.resolve() == args.output.resolve():
        parser.error('Output must differ from input')
    try:
        return run(args.input, args.output)
    except (ValueError, KeyError, TypeError, OSError, cv2.error, np.linalg.LinAlgError) as exc:
        print(f'Calibration failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
