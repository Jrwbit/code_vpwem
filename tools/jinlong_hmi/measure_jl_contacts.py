#!/usr/bin/env python3
"""Measure saved JL100/JL132 contact depths from independent manual rim ROIs.

No legacy fitted pose/result is consumed. Fixed pixel ROIs apply only to the
two archived scans named below; this is a measurement audit, not a detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SCANS = {
    'JL100': {
        'task': '20260730_205445_815e27',
        'rim': {
            'top': [[1740, 510], [2160, 501], [2160, 529], [1740, 536]],
            'left': [[1680, 587], [1714, 587], [1714, 980], [1680, 980]],
            'right': [[2208, 580], [2238, 580], [2238, 990], [2208, 990]],
            'bottom': [[1760, 1034], [2180, 1040], [2180, 1062], [1760, 1057]],
        },
        'studs': [[1915, 667], [1914, 788], [1911, 903], [2033, 667], [2030, 788], [2028, 903]],
        'ground': [1983, 998], 'view': [1620, 2290, 460, 1110],
        'outer_corner_estimates': [[1670, 508], [2248, 490], [2240, 1070], [1665, 1066]],
    },
    'JL132': {
        'task': '20260729_214004_7c4f40',
        'rim': {
            'top': [[1128, 501], [1560, 343], [1570, 368], [1138, 526]],
            'left': [[1070, 620], [1094, 614], [1265, 1040], [1240, 1049]],
            'right': [[1718, 430], [1740, 422], [1883, 811], [1861, 820]],
            'bottom': [[1400, 1104], [1800, 952], [1810, 977], [1410, 1129]],
        },
        'studs': [[1344, 623], [1395, 761], [1450, 902], [1483, 567], [1538, 703], [1594, 847]],
        'ground': [1553, 978], 'view': [980, 1970, 250, 1230],
        'outer_corner_estimates': [[1030, 534], [1680, 297], [1916, 936], [1289, 1173]],
    },
}


def measure(fid, source_root, output):
    definition = SCANS[fid]
    source = source_root / definition['task']
    rgb = cv2.imread(str(source / '01_rgb_original.png'))
    depth = cv2.imread(str(source / '03_depth_raw_u16.png'), cv2.IMREAD_UNCHANGED)
    if depth is None or rgb is None:
        raise ValueError(f'Incomplete or unreadable archived image: {source}')
    intr = json.loads((source / 'camera_intrinsics.json').read_text())
    if depth.shape != rgb.shape[:2] or depth.shape != (intr['height'], intr['width']):
        raise ValueError('RGB/depth/intrinsic dimensions differ')
    npy_match = None
    if (source / '03_depth_raw.npy').exists():
        npy_match = bool(np.array_equal(depth, np.load(source / '03_depth_raw.npy')))
        if not npy_match:
            raise ValueError('Raw NPY and PNG depths differ')
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    keys = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4']
    distortion = np.array([intr['dist_coeffs'][key] for key in keys])

    def sample(mask):
        yy, xx = np.where(mask & (depth > 0))
        if not len(xx):
            return np.empty((0, 3)), 0.0
        ray = cv2.undistortPoints(np.c_[xx, yy].astype(float).reshape(-1, 1, 2), K, distortion)[:, 0]
        points = np.c_[ray, np.ones(len(ray))] * (depth[yy, xx] * intr['depth_unit_raw'] / 1000)[:, None]
        return points, float(len(xx) / np.count_nonzero(mask))

    rim_points, rim_stats = [], {}
    for name, polygon in definition['rim'].items():
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 1)
        points, fraction = sample(mask)
        rim_points.append(points)
        rim_stats[name] = {'roi': polygon, 'valid_points': len(points), 'valid_fraction': fraction}
    points = np.concatenate(rim_points)
    rng = np.random.default_rng(20261023)
    subset = points[rng.choice(len(points), min(6000, len(points)), replace=False)]
    best = None
    for _ in range(300):
        p, q, r = subset[rng.choice(len(subset), 3, replace=False)]
        normal = np.cross(q - p, r - p)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        offset = -normal @ p
        count = np.count_nonzero(abs(subset @ normal + offset) < .0005)
        if best is None or count > best[0]:
            best = count, normal, offset
    normal, offset = best[1:]
    keep = abs(points @ normal + offset) < .0005
    for _ in range(5):
        centre = points[keep].mean(0)
        normal = np.linalg.svd(points[keep] - centre, full_matrices=False)[2][-1]
        if normal[2] < 0:
            normal = -normal
        offset = -normal @ centre
        residual = points @ normal + offset
        keep = abs(residual) < .0005
    for (name, stats), patch in zip(rim_stats.items(), rim_points):
        signed = (patch @ normal + offset) * 1000
        stats['residual_mm_percentiles_5_50_95'] = np.percentile(signed, [5, 50, 95]).tolist() if len(signed) else None
        stats['plane_inlier_fraction'] = float(np.mean(abs(signed) < .5)) if len(signed) else 0

    features = []
    for i, xy in enumerate(definition['studs'] + [definition['ground']]):
        stats = {'id': f'stud_{i + 1}' if i < 6 else 'ground', 'pixel_center': xy, 'rois': []}
        for inner, outer in [(0, 5), (0, 8), (7, 12), (0, 16)]:
            mask = np.zeros(depth.shape, np.uint8)
            cv2.circle(mask, xy, outer, 1, -1)
            if inner:
                cv2.circle(mask, xy, inner, 0, -1)
            cloud, fraction = sample(mask)
            signed = (cloud @ normal + offset) * 1000
            stats['rois'].append({'inner_radius_px': inner, 'outer_radius_px': outer,
                                  'valid_points': len(cloud), 'valid_fraction': fraction,
                                  'depth_below_rim_mm_percentiles_5_50_95': np.percentile(signed, [5, 50, 95]).tolist() if len(cloud) else None,
                                  'median_camera_xyz_m': np.median(cloud, axis=0).tolist() if len(cloud) else None})
            if inner == 0 and outer == 16 and len(cloud):
                threshold = float(np.percentile(signed, 5) + .6)
                selected = signed <= threshold
                stats['outward_cap_cluster'] = {
                    'method': 'Within r16 ROI, retain depths <= P5 + 0.6mm to separate tip/cap from deeper nut/recess.',
                    'threshold_mm': threshold, 'selected_points': int(selected.sum()),
                    'all_valid_roi_points': len(cloud),
                    'depth_mm_percentiles_5_50_95': np.percentile(signed[selected], [5, 50, 95]).tolist(),
                    'median_camera_xyz_m': np.median(cloud[selected], axis=0).tolist(),
                }
        features.append(stats)
    uv = np.asarray(definition['outer_corner_estimates'], float).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(uv, K, distortion)[:, 0]
    rays = np.c_[rays, np.ones(4)]
    corners = rays * (-offset / (rays @ normal))[:, None]
    edge_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1) * 1000
    result = {
        'model': fid, 'task': definition['task'], 'source': str(source),
        'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ['01_rgb_original.png', '03_depth_raw_u16.png', 'camera_intrinsics.json']},
        'npy_matches_png': npy_match, 'legacy_pose_result_used': False,
        'manual_outer_envelope_check': {'pixel_corners': definition['outer_corner_estimates'],
                                        'four_edge_lengths_mm': edge_lengths.tolist(),
                                        'use': 'Approximate visual model-size check, not dimensional acceptance.'},
        'rim_plane': {'normal': normal.tolist(), 'offset_m': float(offset), 'valid_points': len(points),
                      'inlier_points': int(keep.sum()), 'rmse_mm': float(np.sqrt(np.mean(residual[keep] ** 2)) * 1000)},
        'rim_rois': rim_stats, 'features': features,
        'limitations': ['Manual ROIs are specific to the named raw RGB image.',
                        'Raw RGB/depth alignment is checked geometrically; no hardware timestamp proof is available.',
                        'Old latest/result poses are not consumed and may represent another placement.',
                        'Plane RMSE is not absolute measurement accuracy; ROI quantiles are not error confidence intervals.',
                        'Ground centre pixels may include the screw recess; annular cap ROI is supplied separately.'],
    }
    if fid == 'JL132':
        result['excluded_archive'] = {
            'task': '20260818_111953_78daac',
            'reason': 'Raw depth PNG truncated at 98304 bytes; partial zlib stream contains only489 full rows. Ground at row806 is unavailable. No missing rows were fabricated.',
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / (fid + '_contacts.json')).write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    for name, polygon in definition['rim'].items():
        q = np.vstack([polygon, polygon[0]])
        ax.plot(q[:, 0], q[:, 1], 'c-', lw=1)
        ax.text(*polygon[0], name, color='cyan', fontsize=10)
    for i, xy in enumerate(definition['studs'] + [definition['ground']]):
        ax.add_patch(plt.Circle(xy, 12, fill=False, color='magenta'))
        ax.annotate(str(i + 1) if i < 6 else 'G', xy, xytext=(15, 10), textcoords='offset points', color='yellow')
    x0, x1, y0, y1 = definition['view']
    ax.set(xlim=(x0, x1), ylim=(y1, y0), title=f'{fid} saved raw frame: independent rim and contact ROIs')
    fig.tight_layout()
    fig.savefig(output / (fid + '_contact_rois.png'), dpi=170)
    plt.close(fig)
    print(json.dumps({'model': fid, 'plane': result['rim_plane'], 'rim': rim_stats,
                      'features': [{f['id']: f['rois'][:3]} for f in features]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('/home/wjr/code/r435/outputs/hik_web_pipeline'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=list(SCANS), default=list(SCANS))
    args = parser.parse_args()
    for fid in args.models:
        measure(fid, args.source_root, args.output)


if __name__ == '__main__':
    main()
