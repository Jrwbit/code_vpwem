#!/usr/bin/env python3
"""Fit offline XY/yaw target corrections to independently measured RGB-D studs.

The tool supplies predicted socket axes, never motor-terminal measurements.
Current site mounting, object offsets, box shifts and all Z values are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/diagnostics/actual-terminal-alignment-20261023'
SOCKET_SEEDS = {'JL100': [[-97.4, -31.4], [-86.6, -50], [-75.9, -68.6]],
                'JL132': [[-25, 100], [0, 100], [25, 100]],
                'JM160': [[65.4, -37.8], [73.2, -66.7], [80.9, -95.7]]}
GROUND_CENTRES = {'JL100': [-57.8, -78.4], 'JL132': [-45.8, 87.5], 'JM160': [93.05, -128.8]}
CORRESPONDENCE = {'JL100': ['U1', 'V1', 'W1'], 'JL132': ['W1', 'V1', 'U1'], 'JM160': ['U1', 'V1', 'W1']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_step_axes(path):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    reader = STEPControl_Reader()
    if int(reader.ReadFile(str(path))) != 1:
        raise ValueError('STEP load failed')
    reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_FACE)
    centres = {3.3: [], 3.5: []}
    while explorer.More():
        surface = BRepAdaptor_Surface(TopoDS.Face_s(explorer.Current())); explorer.Next()
        if surface.GetType() != GeomAbs_Cylinder:
            continue
        cylinder = surface.Cylinder()
        radius = min(centres, key=lambda r: abs(cylinder.Radius() - r))
        if abs(cylinder.Radius() - radius) > 1e-6:
            continue
        direction = cylinder.Axis().Direction()
        if abs(abs(direction.Z()) - 1) > 1e-8:
            raise ValueError('Expected socket/ground bores normal to the plate')
        p = cylinder.Location(); xy = np.array([p.X(), p.Y()])
        if not any(np.linalg.norm(xy - old) < 1e-6 for old in centres[radius]):
            centres[radius].append(xy)
    parts = {}
    for fid, seeds in SOCKET_SEEDS.items():
        holes = np.asarray(centres[3.5])
        ids = [int(np.argmin(np.linalg.norm(holes - seed, axis=1))) for seed in seeds]
        if max(np.linalg.norm(holes[i] - seed) for i, seed in zip(ids, seeds)) > .1:
            raise ValueError('Socket CAD axes no longer match known fixture sectors')
        slot = np.asarray(centres[3.3])
        slot = slot[np.linalg.norm(slot - GROUND_CENTRES[fid], axis=1) < 6]
        if slot.shape != (2, 2) or abs(np.linalg.norm(slot[1]-slot[0]) - 8) > 1e-6:
            raise ValueError('Expected two ground-slot cap centres spaced 8mm')
        parts[fid] = {'socket_xy_step_mm': holes[ids].tolist(), 'ground_slot_cap_centres_xy_step_mm': slot.tolist(),
                      'ground_slot_radius_mm': 3.3}
    return {'source_step': str(path), 'source_step_sha256': sha(path), 'parts': parts,
            'method': 'OpenCascade analytic cylinder axes: D7 socket bores and R3.3 ground-slot cap centres. No measured motor point comes from the tool.'}


def fit_rigid_xy(predicted, measured):
    p, q = np.asarray(predicted), np.asarray(measured)
    pc, qc = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    fix = np.diag([1., np.sign(np.linalg.det(vt.T @ u.T))])
    R = vt.T @ fix @ u.T
    t = q.mean(0) - R @ p.mean(0)
    return R, t


def transform_mm(T, points):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3] * 1000


def residuals(predicted, observed):
    residual = np.asarray(predicted) - observed
    norms = np.linalg.norm(residual, axis=1)
    return {'predicted_minus_observed_xy_mm': residual.tolist(), 'per_point_norm_mm': norms.tolist(),
            'rms_mm': float(np.sqrt(np.mean(norms ** 2))), 'max_mm': float(norms.max())}


def ground_slot_error(slot_xy, ground_xy):
    a, b = np.asarray(slot_xy)
    u = (b-a) / np.linalg.norm(b-a)
    center = (a+b) / 2
    along = float((ground_xy-center) @ u)
    cross = float((ground_xy-center) @ np.array([-u[1], u[0]]))
    nearest = center + np.clip(along, -np.linalg.norm(b-a)/2, np.linalg.norm(b-a)/2) * u
    return {'slot_cap_centres_xy_mm': [a.tolist(), b.tolist()], 'slot_midpoint_xy_mm': center.tolist(),
            'observed_ground_xy_mm': ground_xy.tolist(), 'along_slot_from_midpoint_mm': along,
            'signed_cross_slot_offset_mm': cross, 'cross_slot_offset_mm': abs(cross),
            'distance_to_slot_centreline_segment_mm': float(np.linalg.norm(ground_xy-nearest)),
            'nearest_centreline_position_mm': nearest.tolist(),
            'scope': 'XY centreline geometry only. Slot radius, pin diameter, assembly tolerances and pin seating are not a manufactured adjustment acceptance.'}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.extract_step:
        args.step_axes.write_text(json.dumps(extract_step_axes(args.step), ensure_ascii=False, indent=2)+'\n')
        return
    correspondence_audit = BASE / 'archive/mating-audit/recorded_fixture_mating.json'
    sources = [Path(__file__), args.site_input, args.step, args.step_axes, correspondence_audit,
               *[args.templates / (fid+'_observed_template.json') for fid in CORRESPONDENCE]]
    hashes = {str(path.resolve()): sha(path) for path in sources}
    bundle = json.loads(args.site_input.read_text()); site = bundle.get('site', bundle)
    mount = np.asarray(site['flange_T_tcp'], float)
    expected = np.eye(4); expected[:3, :3] = Rotation.from_euler('z', -90, degrees=True).as_matrix()
    if not np.allclose(mount, expected, atol=1e-10):
        raise ValueError('This offline audit requires the current verified Rz(-90) mount')
    axes = json.loads(args.step_axes.read_text())
    if axes['source_step_sha256'] != sha(args.step):
        raise ValueError('STEP changed since analytic axes extraction')
    tcp_T_step = np.eye(4)
    tcp_T_step[:3, :3] = Rotation.from_euler('z', -150, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
    tcp_T_step[:3, 3] = -tcp_T_step[:3, :3] @ [0, 0, .006]
    flange_T_step = mount @ tcp_T_step
    corrections, details = {}, {}
    for fid, names in CORRESPONDENCE.items():
        template_path = args.templates / (fid+'_observed_template.json')
        template = json.loads(template_path.read_text())
        if template['model'] != fid or template['tool_geometry_used_for_measurement'] is not False:
            raise ValueError('Expected an independently observed template of the selected motor')
        features = {x['id']: x for x in template['features']}
        measured = np.asarray([features[name]['xyz_box_mm'] for name in names])
        ground = np.asarray(features['ground']['xyz_box_mm'])
        tool = np.asarray(site['tcp_T_objects'][fid], float)
        tool[2, 3] += site.get('fixture_z_adjustments_mm', {}).get(fid, 0) / 1000
        shift = np.eye(4)
        shift[:3, 3] = np.asarray(site.get('box_center_adjustments_mm', {}).get(fid, [0, 0, 0])) / 1000
        nominal = shift @ np.linalg.inv(tool) @ np.linalg.inv(mount)
        part = axes['parts'][fid]
        # z=0 is an axis reference plane, never an assertion about socket length.
        socket_step = np.c_[part['socket_xy_step_mm'], np.zeros(3)]
        slot_step = np.c_[part['ground_slot_cap_centres_xy_step_mm'], np.zeros(2)]
        socket_flange = transform_mm(flange_T_step, socket_step)
        predicted = transform_mm(nominal, socket_flange)
        slot = transform_mm(nominal, transform_mm(flange_T_step, slot_step))
        R, t = fit_rigid_xy(predicted[:, :2], measured[:, :2])
        yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        C = np.eye(4); C[:2, :2] = R; C[:2, 3] = t / 1000
        corrected = C @ nominal
        corrected_points = transform_mm(corrected, socket_flange)
        corrected_slot = slot[:, :2] @ R.T + t
        if not np.array_equal(corrected[2], nominal[2]) or not np.allclose(corrected_points[:, 2], predicted[:, 2], atol=1e-12):
            raise ValueError('XY fitting changed nominal Z')
        corrections[fid] = {'yaw_deg': yaw, 'translation_mm': [float(t[0]), float(t[1]), 0.]}
        details[fid] = {
            'template_source': str(template_path), 'source_task': template['source_task'],
            'frame': template['frame'], 'socket_1_2_3_target_feature_ids': names,
            'column_selection_basis': str(correspondence_audit),
            'independent_observed_studs_box_xyz_mm': measured.tolist(),
            'nominal_box_T_flange': nominal.tolist(), 'correction_box_T_nominal_box': C.tolist(),
            'corrected_box_T_flange': corrected.tolist(),
            'nominal_socket_axis_reference_points_box_mm': predicted.tolist(),
            'corrected_socket_axis_reference_points_box_mm': corrected_points.tolist(),
            'before_fit': residuals(predicted[:, :2], measured[:, :2]),
            'after_fit': residuals(corrected_points[:, :2], measured[:, :2]),
            'ground_slot_before': ground_slot_error(slot[:, :2], ground[:2]),
            'ground_slot_after': ground_slot_error(corrected_slot, ground[:2]),
            'original_site_fixture_z_adjustment_mm': site.get('fixture_z_adjustments_mm', {}).get(fid, 0),
            'original_site_box_center_adjustment_mm': site.get('box_center_adjustments_mm', {}).get(fid, [0, 0, 0]),
            'flange_box_z_before_after_mm': [float(nominal[2, 3]*1000), float(corrected[2, 3]*1000)],
            'Z_unchanged': True, 'scaling_allowed': False, 'reflection_allowed': False,
            'measurement_sensitivities_mm': {name: features[name].get('pixel_center_plus_minus_3px_sensitivity_mm') for name in names},
        }
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('An input changed during the fit')
    report = {'status': 'offline_noncontact_target_candidate_only', 'controller_writes': False,
              'flange_T_tcp_unchanged': mount.tolist(), 'correction_semantics': 'solver local_flange_target = correction @ nominal_box_T_flange; all translations are mm and correction Z is exactly zero',
              'correspondence_basis': 'Independent same-task historical real-path overlays select U1/V1/W1; JL132 socket numbering is reversed (W1,V1,U1). No terminal coordinates are inferred from tool geometry.',
              'source_sha256': hashes, 'corrections': corrections, 'fixtures': details,
              'limitations': ['Single historical RGB-D manual feature measurements; residual fit is not an independent insertion accuracy acceptance.',
                             'Only in-plane rotation and XY translation are fitted; all current site Z, press, mounting and tool shape remain unchanged.',
                             'Ground error is reported independently and is not added to the three-stud objective or silently corrected.',
                             'Socket/pin lengths, sleeve bores, forces and electrical tests are not validated.',
                             'Motor-foot/shaft-to-box registration is separate from this local mating fit.']}
    (args.output / 'target_corrections.json').write_text(json.dumps(corrections, indent=2)+'\n')
    (args.output / 'fit_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({fid: {'correction': corrections[fid], 'after_fit': item['after_fit'],
                           'ground_cross_slot_mm': item['ground_slot_after']['cross_slot_offset_mm']}
                      for fid, item in details.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-input', type=Path, default=ROOT / 'outputs/diagnostics/front-fixed-waist-20261023/input.json')
    parser.add_argument('--templates', type=Path, default=BASE / 'direction-audit/templates')
    parser.add_argument('--step', type=Path, default=Path('/home/wjr/code/jinlong-motor-arm-control/docs/tooling/flange_v2.STEP'))
    parser.add_argument('--output', type=Path, default=BASE / 'target-fit')
    parser.add_argument('--step-axes', type=Path, default=BASE / 'target-fit/step_socket_and_slot_axes.json')
    parser.add_argument('--extract-step', action='store_true')
    run(parser.parse_args())
