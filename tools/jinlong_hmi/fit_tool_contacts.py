#!/usr/bin/env python3
"""Fit archived JM160 stud surfaces to the real tool STEP's mounting axes.

This is an offline design calculation. D7 in the STEP is a mounting hole, not
verified socket bore diameter. All radial fit limits below are sensitivity
parameters; depth-cap segmentation is not absolute metrology.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / 'outputs/diagnostics/full-solve-20261023'
STEP = Path('/home/wjr/code/jinlong-motor-arm-control/docs/tooling/flange_v2.STEP')
NOMINAL_SOCKETS = [[65.4, -37.8], [73.2, -66.7], [80.9, -95.7]]
NOMINAL_SLOT = [[89.6, -126.8], [96.5, -130.8]]


def extract_axes(source, output):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    reader = STEPControl_Reader(); reader.ReadFile(str(source)); reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_FACE); axes = []
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current()); explorer.Next()
        surf = BRepAdaptor_Surface(face)
        if surf.GetType() != GeomAbs_Cylinder:
            continue
        cylinder = surf.Cylinder(); p = cylinder.Location(); axis = cylinder.Axis().Direction()
        if cylinder.Radius() <= 6 and abs(axis.Z()) > .99:
            axes.append({'r': cylinder.Radius(), 'xyz': [p.X(), p.Y(), p.Z()], 'axis': [axis.X(), axis.Y(), axis.Z()]})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(axes, indent=2) + '\n')
    return axes


def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def fit_se2(source, target):
    U, _, Vt = np.linalg.svd((source-source.mean(0)).T @ (target-target.mean(0)))
    R = Vt.T @ np.diag([1., np.linalg.det(Vt.T @ U.T)]) @ U.T
    t = target.mean(0) - source.mean(0) @ R.T
    return np.r_[math.atan2(R[1, 0], R[0, 0]), t]


def apply_se2(points, fit):
    return np.asarray(points) @ rotation(fit[0]).T + fit[1:3]


def cap_points(contacts, tolerance=.4):
    scan = Path(contacts['source_scan']); intr = json.loads((scan / 'camera_intrinsics.json').read_text())
    depth = np.load(scan / '03_depth_raw.npy')
    rgb = cv2.imread(str(scan / '01_rgb_original.png'))
    T = np.asarray(contacts['camera_T_box'])
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    distortion = np.array([intr['dist_coeffs'][key] for key in ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4']])
    out, crops = [], []
    for feature in contacts['features'][:3]:
        cx, cy = feature['pixel_center']; yy, xx = np.mgrid[cy-50:cy+51, cx-50:cx+51]
        uv = np.c_[xx.ravel(), yy.ravel()]
        ray = cv2.undistortPoints(uv.astype(float).reshape(-1, 1, 2), K, distortion)[:, 0]
        camera = np.c_[ray, np.ones(len(ray))] * (depth[yy, xx].ravel() * intr['depth_unit_raw'] / 1000)[:, None]
        points = (camera - T[:3, 3]) @ T[:3, :3] * 1000
        z = points[:, 2].reshape(xx.shape)
        mask = ((np.abs(z-feature['xyz_box_mm'][2]) < tolerance) & ((xx-cx)**2+(yy-cy)**2 < 35**2) & (depth[yy, xx] > 0)).astype('uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if count < 2:
            raise ValueError('No valid cap component: ' + feature['id'])
        index = max(range(1, count), key=lambda k: stats[k, cv2.CC_STAT_AREA])
        if stats[index, cv2.CC_STAT_AREA] < 200:
            raise ValueError('Insufficient observed cap pixels')
        contours, _ = cv2.findContours((labels == index).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ellipse = cv2.fitEllipse(max(contours, key=cv2.contourArea))
        center = np.array(ellipse[0]) + [cx-50, cy-50]
        # Keep measured depth only; closing repairs image topology, not depth.
        real = (labels == index) & (depth[yy, xx] > 0) & (np.abs(z-feature['xyz_box_mm'][2]) < tolerance)
        cap_z = float(np.median(z[real]))
        camera_ray = np.r_[cv2.undistortPoints(center.reshape(1, 1, 2), K, distortion).ravel(), 1.]
        ray_scale = (cap_z / 1000 + T[:3, 3] @ T[:3, 2]) / (camera_ray @ T[:3, 2])
        xyz = (camera_ray * ray_scale - T[:3, 3]) @ T[:3, :3] * 1000
        row = {'id': feature['id'], 'pixel_center': center.tolist(), 'xyz_box_mm': xyz.tolist(),
               'component_area_px': int(stats[index, cv2.CC_STAT_AREA]),
               'ellipse_axes_px': list(ellipse[1]), 'ellipse_angle_deg': ellipse[2],
               'centroid_to_ellipse_px': float(np.linalg.norm(centroids[index] + [cx-50, cy-50] - center)),
               'depth_window_mm': tolerance}
        out.append(row)
        crop = rgb[cy-50:cy+51, cx-50:cx+51].copy()
        cv2.ellipse(crop, ellipse, (0, 220, 0), 1)
        cv2.drawMarker(crop, (50, 50), (140, 140, 140), cv2.MARKER_CROSS, 7, 1)
        cv2.drawMarker(crop, tuple(np.rint(ellipse[0]).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 7, 1)
        crops.append(cv2.resize(crop, None, fx=3, fy=3))
    return out, np.hstack(crops)


def tool_axes(axes, site):
    def nearest(point, radius):
        candidates = [np.asarray(a['xyz']) for a in axes if abs(a['r']-radius) < 1e-6]
        best = min(candidates, key=lambda p: np.linalg.norm(p[:2]-point))
        if np.linalg.norm(best[:2]-point) > .2:
            raise ValueError('STEP axis does not match the expected fixture mounting hole')
        return best[:2]
    socket_xy = np.array([nearest(p, 3.5) for p in NOMINAL_SOCKETS])
    slot_xy = np.array([nearest(p, 3.3) for p in NOMINAL_SLOT])
    R = np.eye(3); R[:2, :2] = rotation(math.radians(-150)); R = R @ np.diag([1., -1., -1.])
    B = np.linalg.inv(np.asarray(site['tcp_T_objects']['JM160']))
    B[2, 3] -= site.get('fixture_z_adjustments_mm', {}).get('JM160', 0) / 1000
    shift = np.asarray(site['box_center_adjustments_mm']['JM160'])
    linear = B[:3, :3] @ R
    def transform(points):
        return (np.asarray(points)-[0, 0, 6.]) @ linear.T + B[:3, 3]*1000 + shift
    sockets = transform(np.c_[socket_xy, np.full(3, -32.)])
    slot = transform(np.c_[slot_xy, np.full(2, -50.)])
    return {'socket_tip_box_mm': sockets, 'ground_slot_tip_box_mm': slot,
            'step_socket_axes_xy_mm': socket_xy, 'step_slot_axes_xy_mm': slot_xy,
            'box_from_step_rotation': linear}


def analyse(source, target, ground, slot, fit, radius_limit):
    center = slot.mean(0)[:2]; direction = slot[1, :2]-slot[0, :2]
    length = float(np.linalg.norm(direction)); direction /= length
    def residual(x):
        return np.linalg.norm(apply_se2(source, x)-target, axis=1)
    def distance(x):
        return float(np.sum((apply_se2(center+direction*x[3], x)-ground)**2))
    init = np.r_[fit, 0.]
    if np.max(residual(init)) > radius_limit:
        # A feasibility search is essential; the least-squares fit is not a
        # minimax fit and cannot alone prove that a requested radius is infeasible.
        mm = minimize(lambda x: x[3], np.r_[fit, np.max(residual(init))], method='SLSQP',
                      constraints={'type': 'ineq', 'fun': lambda x: x[3]-residual(x)},
                      options={'ftol': 1e-12, 'maxiter': 500})
        if not mm.success or mm.x[3] > radius_limit+1e-6:
            return {'socket_radial_limit_mm': radius_limit, 'feasible': False, 'minimax_required_mm': float(mm.x[3])}
        init[:3] = mm.x[:3]
    fit_opt = minimize(distance, init, method='SLSQP',
                       bounds=[(fit[0]-.1, fit[0]+.1), (fit[1]-5, fit[1]+5), (fit[2]-5, fit[2]+5), (-length/2, length/2)],
                       constraints={'type': 'ineq', 'fun': lambda x: radius_limit-residual(x)},
                       options={'ftol': 1e-9, 'maxiter': 1000})
    x = fit_opt.x; delta = ground-apply_se2(center+direction*x[3], x); dist = float(np.linalg.norm(delta))
    required = delta*max(0., dist-1.5)/dist if dist else np.zeros(2)
    return {'socket_radial_limit_mm': radius_limit, 'feasible': bool(np.max(residual(x)) <= radius_limit+1e-6), 'optimizer_converged': bool(fit_opt.success),
            'se2_yaw_deg': math.degrees(x[0]), 'se2_translation_mm': x[1:3].tolist(),
            'socket_residuals_mm': residual(x).tolist(), 'slot_shift_mm': float(x[3]),
            'ground_center_error_mm': delta.tolist(), 'ground_center_distance_mm': dist,
            'minimum_additional_shift_for_assumed_1p5mm_landing_mm': required.tolist(),
            'minimum_additional_shift_norm_mm': max(0., dist-1.5)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contacts', type=Path, default=DEFAULT_BASE/'scan_contacts.json')
    p.add_argument('--site-bundle', type=Path, default=ROOT/'outputs/diagnostics/front-fixed-waist-20261023/input.json')
    p.add_argument('--axes', type=Path, default=DEFAULT_BASE/'tool_geometry/hole_axes_step_mm.json')
    p.add_argument('--output', type=Path, default=DEFAULT_BASE/'contact_fit')
    p.add_argument('--extract-cad-only', action='store_true')
    args = p.parse_args()
    if args.extract_cad_only:
        extract_axes(STEP, args.axes); return
    args.output.mkdir(parents=True, exist_ok=True)
    contacts = json.loads(args.contacts.read_text()); site = json.loads(args.site_bundle.read_text())['site']
    axes = json.loads(args.axes.read_text()); tool = tool_axes(axes, site)
    refined, preview = cap_points(contacts)
    cv2.imwrite(str(args.output/'stud_cap_centres.png'), preview)
    source = tool['socket_tip_box_mm'][:, :2]; target = np.array([r['xyz_box_mm'][:2] for r in refined])
    ground = np.asarray(contacts['features'][-1]['xyz_box_mm'][:2]); slot = tool['ground_slot_tip_box_mm']
    fit = fit_se2(source, target); residual = target-apply_se2(source, fit)
    direction = slot[1, :2]-slot[0, :2]; length = np.linalg.norm(direction); direction /= length
    center = slot.mean(0)[:2]; d = ground-apply_se2(center, fit)
    adjustment = float(np.clip(d @ (rotation(fit[0]) @ direction), -length/2, length/2))
    after_slot = ground-apply_se2(center+adjustment*direction, fit)
    corrected_box_delta = rotation(fit[0]).T @ after_slot
    manufacturing = tool['box_from_step_rotation'].T @ np.r_[corrected_box_delta, 0.]
    inward_step = tool['box_from_step_rotation'].T @ np.array([0., 5., 0.])
    proposed = center + np.array([0., 5.])
    d_proposed = ground-apply_se2(proposed, fit)
    proposed_slot = float(np.clip(d_proposed @ (rotation(fit[0]) @ direction), -length/2, length/2))
    proposed_error = ground-apply_se2(proposed+proposed_slot*direction, fit)
    cap_depth = float(contacts['ground_cap_wide_roi_depth_median_mm'])
    z_ground = float(slot[:, 2].mean()); stud_z = np.array([r['xyz_box_mm'][2] for r in refined])
    depth = []
    for compression in [0., 2.]:
        extra = cap_depth+compression-z_ground
        depth.append({'ground_compression_mm': compression, 'additional_box_Z_motion_mm': extra,
                      'socket_tip_depth_mm': (tool['socket_tip_box_mm'][:, 2]+extra).tolist(),
                      'socket_overlap_from_scanned_stud_tip_mm': (tool['socket_tip_box_mm'][:, 2]+extra-stud_z).tolist()})
    sensitivity = []
    for tol in [.25, .4, .6, 1.]:
        caps, _ = cap_points(contacts, tol); points = np.array([r['xyz_box_mm'][:2] for r in caps]); sf = fit_se2(source, points)
        sensitivity.append({'cap_depth_window_mm': tol, 'yaw_deg': math.degrees(sf[0]), 'translation_mm': sf[1:].tolist(),
                            'max_socket_residual_mm': float(np.max(np.linalg.norm(points-apply_se2(source, sf), axis=1))),
                            'ground_error_before_slot_mm': (ground-apply_se2(center, sf)).tolist()})
    result = {'model': 'JM160', 'status': 'offline_nominal_fit_not_metrology_or_executed_plan',
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [STEP, args.contacts, args.site_bundle, args.axes, Path(__file__)]},
              'refined_stud_caps': refined, 'tool': {k: v.tolist() for k, v in tool.items()},
              'least_squares': {'yaw_deg': math.degrees(fit[0]), 'translation_mm': fit[1:].tolist(), 'socket_residual_vectors_mm': residual.tolist(),
                                'socket_residuals_mm': np.linalg.norm(residual, axis=1).tolist(),
                                'ground_error_before_slot_mm': d.tolist(), 'slot_shift_mm': adjustment,
                                'ground_error_after_slot_mm': after_slot.tolist(), 'ground_distance_after_slot_mm': float(np.linalg.norm(after_slot)),
                                'additional_seat_shift_nominal_box_mm': corrected_box_delta.tolist(),
                                'additional_seat_shift_STEP_mm': manufacturing.tolist()},
              'socket_limit_sensitivity': [analyse(source, target, ground, slot, fit, lim) for lim in [.3, .5, .75, 1.]],
              'five_mm_inward_seat_candidate': {'shift_nominal_box_mm': [0., 5., 0.], 'shift_STEP_mm': inward_step.tolist(),
                                               'slot_shift_mm': proposed_slot, 'ground_error_after_shift_mm': proposed_error.tolist(),
                                               'ground_distance_mm': float(np.linalg.norm(proposed_error)),
                                               'fits_assumed_1p5mm_landing_radius': bool(np.linalg.norm(proposed_error) <= 1.5)},
              'depth': depth, 'segmentation_sensitivity': sensitivity,
              'limitations': ['D7 is the tool mounting hole, not verified socket bore. Radial limits are sensitivity assumptions.',
                              'Top-face ellipse fit and threshold variation do not establish absolute 0.5mm metrology accuracy.',
                              'Ground XY remains manual ROI centre; its cap radius4mm and pin radius2.5mm are model assumptions.',
                              'Eight-mm slot range is geometric centre travel; fitted mounting shaft/nut clearance has not been measured.',
                              'Socket extension32mm and ground pin22mm are photo estimates. Depth overlaps are geometric, not electrical engagement acceptance.',
                              'Five-mm seat shift requires complete tool/box collision recheck and manufacturability review.',
                              'Whole-tool 90degree adapter cannot alter these internal socket/ground relationships at mating.']}
    (args.output/'fit_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['least_squares', 'five_mm_inward_seat_candidate', 'depth', 'socket_limit_sensitivity']}, indent=2))


if __name__ == '__main__':
    main()
