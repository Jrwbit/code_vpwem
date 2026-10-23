#!/usr/bin/env python3
"""Conservative height-field collision model from the saved 160 RGB-D scan.

Each XY column is occupied below its nearest observed surface. Missing depth
is occupied from the rim, not interpreted as free space. Clearly identified
terminal footprints are split into named cylinders for pair-specific mating.
This is a reproducible offline approximation, not full solid reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def build(scan, contacts_file, output, pitch=2.0, bottom=90.0):
    contacts = json.loads(contacts_file.read_text())
    rim_audit_file = scan.parent / 'cb9_audit/scan_audit.json'
    rim_audit = json.loads(rim_audit_file.read_text())
    T = np.asarray(contacts['camera_T_box'], float)
    intr = json.loads((scan / 'camera_intrinsics.json').read_text())
    depth = np.load(scan / '03_depth_raw.npy')
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    keys = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4']
    distortion = np.array([intr['dist_coeffs'][key] for key in keys])
    yy, xx = np.meshgrid(np.arange(70, 1540, 2), np.arange(850, 2450, 2), indexing='ij')
    xx, yy = xx.ravel(), yy.ravel()
    valid = depth[yy, xx] > 0
    xx, yy = xx[valid], yy[valid]
    ray = cv2.undistortPoints(np.c_[xx, yy].astype(float).reshape(-1, 1, 2), K, distortion)[:, 0]
    camera = np.c_[ray, np.ones(len(ray))] * (depth[yy, xx] * intr['depth_unit_raw'] / 1000)[:, None]
    points = (camera - T[:3, 3]) @ T[:3, :3] * 1000
    # Round the measured outer straight-edge envelope outward to whole cells.
    half = np.ceil(np.array([154.6, 149.4]) / (2 * pitch)) * pitch
    shape = np.rint(2 * half / pitch).astype(int)
    valid = np.all(abs(points[:, :2]) < half, axis=1) & (points[:, 2] >= -5) & (points[:, 2] < bottom)
    points = points[valid]
    features = contacts['features']
    terminals = []
    exclusion = np.zeros(len(points), bool)
    for i, feature in enumerate(features):
        xyz = np.asarray(feature['xyz_box_mm'], float)
        ground = feature['id'].startswith('ground')
        if ground:
            # A tight centre ROI includes the Phillips recess. Use the broad
            # cap ROI's outward P5 surface for collision, not recess depth.
            xyz[2] = rim_audit['ground_screw_cap_roi']['below_rim_plane_mm_percentiles_5_50_95'][0]
        else:
            xyz[2] = feature['depth_percentile_5_50_95_mm'][0]
        # M6 terminal shafts and the existing D8 ground-cap model. Radii are
        # retained modelling assumptions; ROI centres do not measure diameters.
        radius = 4.0 if ground else 3.0
        exclusion |= np.linalg.norm(points[:, :2] - xyz[:2], axis=1) < radius
        if ground:
            name, length = 'term_JM160_gnd', 8.0
        elif i < 3:
            name, length = f'term_JM160_stud_{i + 1}', 30.0
        else:
            name, length = f'term_JM160_other_{i - 2}', 30.0
        terminals.append({'name': name, 'a': (xyz / 1000).tolist(),
                          'b': ((xyz + [0, 0, length]) / 1000).tolist(), 'radius': radius / 1000,
                          'center_source': feature['id'], 'radius_source': 'inherited M6/D8 model, not ROI diameter'})
    surface = points[~exclusion]
    index = np.floor((surface[:, :2] + half) / pitch).astype(int)
    heights = np.full(shape, np.inf)
    np.minimum.at(heights, (index[:, 0], index[:, 1]), surface[:, 2])
    count = np.zeros(shape, int)
    np.add.at(count, (index[:, 0], index[:, 1]), 1)
    centres = np.stack(np.meshgrid((np.arange(shape[0]) + .5) * pitch - half[0],
                                   (np.arange(shape[1]) + .5) * pitch - half[1], indexing='ij'), axis=-1)
    terminal_interior = np.zeros(shape, bool)
    for terminal in terminals:
        # Only fully covered cells can be represented solely by the cylinder.
        distance = np.linalg.norm(centres - np.asarray(terminal['a'][:2]) * 1000, axis=-1)
        terminal_interior |= distance + pitch / np.sqrt(2) <= terminal['radius'] * 1000
    unknown = ~np.isfinite(heights) & ~terminal_interior
    heights[unknown] = 0.0
    heights[terminal_interior] = bottom
    # Round top outward in Z by <=1 mm, preserving the conservative direction.
    levels = np.floor(heights).astype(int)
    used = np.zeros(shape, bool)
    boxes = []
    for ix in range(shape[0]):
        for iy in range(shape[1]):
            if used[ix, iy] or levels[ix, iy] >= bottom:
                continue
            level, missing = levels[ix, iy], unknown[ix, iy]
            end_y = iy + 1
            while end_y < shape[1] and not used[ix, end_y] and levels[ix, end_y] == level and unknown[ix, end_y] == missing:
                end_y += 1
            end_x = ix + 1
            while (end_x < shape[0] and not np.any(used[end_x, iy:end_y]) and
                   np.all(levels[end_x, iy:end_y] == level) and np.all(unknown[end_x, iy:end_y] == missing)):
                end_x += 1
            used[ix:end_x, iy:end_y] = True
            lo = np.array([ix * pitch - half[0], iy * pitch - half[1], level])
            hi = np.array([end_x * pitch - half[0], end_y * pitch - half[1], bottom])
            boxes.append({'center': ((lo + hi) / 2000).tolist(), 'half_size': ((hi - lo) / 2000).tolist(),
                          'unknown': bool(missing)})
    doc = {
        'model': 'JM160', 'status': 'offline_conservative_saved_scan_height_field', 'units': 'metres',
        'source_scan': str(scan.resolve()), 'camera_T_box': T.tolist(),
        'source_sha256': {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [scan / '03_depth_raw.npy', scan / 'camera_intrinsics.json', contacts_file, rim_audit_file]},
        'frame': contacts['frame'], 'grid_pitch_mm': pitch, 'z_quantization_mm': 1,
        'grid_shape': shape.tolist(), 'retained_depth_points': len(points),
        'unknown_columns': int(unknown.sum()), 'total_columns': int(unknown.size),
        'terminal_only_columns': int(terminal_interior.sum()),
        'boxes': boxes, 'parts': [], 'measured_terminals': terminals,
        'limitations': [
            'Surface columns fill all space below the first observed depth; this can overestimate occluded solid.',
            'Missing depth columns start at the rim and may falsely obstruct an otherwise feasible insertion.',
            'Outer footprint is the rounded-out straight-edge envelope; exterior corner rounding is filled.',
            'Terminal XY centres and outward P5 top heights come from the scan; radii and lengths remain model assumptions.',
            'Ground collision top uses the broad cap ROI P5; centre-ROI median includes the screw recess.',
            'The 2-pixel sampling and 2mm XY cells do not establish submillimetre absolute accuracy.',
            'Only the terminal-box envelope is reconstructed; exterior motor/table/camera remain separate obstacles.',
            'No camera capture or robot motion was performed.',
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'JM160_collision.json').write_text(json.dumps(doc, indent=2) + '\n')
    np.savez_compressed(output / 'JM160_height_field.npz', heights=heights, counts=count,
                        unknown=unknown, half_extent=half, terminal_interior=terminal_interior)
    print(json.dumps({k: doc[k] for k in ['status', 'grid_shape', 'retained_depth_points', 'unknown_columns', 'total_columns',
                                         'terminal_only_columns']} | {'collision_boxes': len(boxes)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--contacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pitch-mm', type=float, default=2.0)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.scan.name != '20261022_110619_cb9e44':
        parser.error('The image crop is bound to the archived 20261022_110619_cb9e44 scan')
    build(args.scan, args.contacts, args.output, args.pitch_mm)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        values = np.load(args.output / 'JM160_height_field.npz')
        heights = values['heights'].copy()
        half = values['half_extent']
        heights[values['unknown']] = np.nan
        heights[values['terminal_interior']] = np.nan
        extent = [-half[0], half[0], -half[1], half[1]]
        fig, axis = plt.subplots(figsize=(9, 8))
        colour = axis.imshow(heights.T, origin='lower', extent=extent, cmap='viridis_r', vmin=0, vmax=70)
        mask = np.ma.masked_where(~values['unknown'].T, values['unknown'].T)
        axis.imshow(mask, origin='lower', extent=extent, cmap=ListedColormap(['#d95d66']), alpha=.85)
        for i, feature in enumerate(json.loads(args.contacts.read_text())['features']):
            xy = feature['xyz_box_mm'][:2]
            label = f'M{i + 1}' if i < 3 else f'U{i - 2}' if i < 6 else 'G'
            axis.plot(*xy, 'wo', mec='black', ms=6)
            axis.annotate(label, xy, xytext=(4, 4), textcoords='offset points', color='black', weight='bold')
        axis.set(xlabel='Box X (mm)', ylabel='Box Y (mm)',
                 title='JM160: nearest saved depth surface below independently fitted rim\n'
                       'Red = missing depth, conservatively occupied from rim')
        fig.colorbar(colour, ax=axis, label='Observed depth below rim (mm)')
        fig.text(.12, .025, 'M1-M3: mating studs; U1-U3: other studs; G: ground screw. '
                 '2 mm cells; no hardware execution.', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(args.output / 'JM160_height_field.png', dpi=170)
        plt.close(fig)


if __name__ == '__main__':
    main()
