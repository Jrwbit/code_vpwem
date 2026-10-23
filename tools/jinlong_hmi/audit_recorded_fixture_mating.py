#!/usr/bin/env python3
"""Compare recorded real TCP poses with independent pixels of motor studs.

This is an offline direction audit, not a production calibration. The manually
reviewed pixels below come from the named, same-task 3200x1944 annotated images;
none of the motor observations are generated from the tool. Unknown stud depth
is explicitly swept. Socket axes are analytic D7 bore centres from the STEP.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'outputs/diagnostics/actual-terminal-alignment-20261023/archive'
STEP = Path('/home/wjr/code/jinlong-motor-arm-control/docs/tooling/flange_v2.STEP')
CASES = {
    'JL100': {'task': '20260818_110510_b0c41b',
              'stud_pixels': [[1219, 443], [1224, 555], [1227, 674],
                              [1335, 440], [1342, 554], [1343, 669]],
              'ground_pixel': [1290, 765],
              'step_xy_selection': [[-97.4, -31.4], [-86.6, -50], [-75.9, -68.6]]},
    'JL132': {'task': '20260818_115509_7f6c5a',
              'stud_pixels': [[1392, 595], [1396, 740], [1403, 882],
                              [1537, 589], [1542, 734], [1548, 878]],
              'ground_pixel': [1471, 984],
              'step_xy_selection': [[-25, 100], [0, 100], [25, 100]]},
    'JM160': {'task': '20260818_115411_244623',
              'stud_pixels': [[1579, 678], [1743, 821], [1897, 962],
                              [1873, 374], [2024, 510], [2172, 650]],
              'ground_pixel': [2058, 1061],
              'step_xy_selection': [[65.4, -37.8], [73.2, -66.7], [80.9, -95.7]]},
}


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
    centres = []
    while explorer.More():
        surface = BRepAdaptor_Surface(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
        if surface.GetType() != GeomAbs_Cylinder:
            continue
        cylinder = surface.Cylinder()
        if abs(cylinder.Radius() - 3.5) > 1e-6:
            continue
        p = cylinder.Location()
        xy = np.array([p.X(), p.Y()])
        if not any(np.linalg.norm(xy - old) < 1e-6 for old in centres):
            centres.append(xy)
    return np.asarray(centres)


def ray_box_points(pixels, intrinsic, camera_T_box, depth_mm):
    """Undistorted camera rays intersect a specified plane in the saved box frame."""
    import cv2
    K = np.asarray(intrinsic['rgb_intrinsic']).reshape(3, 3)
    d = intrinsic['dist_coeffs']
    dist = np.array([d[k] for k in ('k1', 'k2', 'p1', 'p2', 'k3')])
    xy = cv2.undistortPoints(np.asarray(pixels, float).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rays = np.c_[xy, np.ones(len(xy))]
    inv = np.linalg.inv(camera_T_box)
    origins = inv[:3, 3] * 1000
    directions = rays @ inv[:3, :3].T
    t = (depth_mm - origins[2]) / directions[:, 2]
    return origins + t[:, None] * directions


def nearest_assignment(predicted, observed):
    errors = np.linalg.norm(predicted[:, None, :2] - observed[None, :, :2], axis=2)
    order = min(itertools.permutations(range(len(observed)), len(predicted)),
                key=lambda ids: sum(errors[i, j] ** 2 for i, j in enumerate(ids)))
    residual = observed[list(order), :2] - predicted[:, :2]
    a = predicted[-1, :2] - predicted[0, :2]
    b = observed[order[-1], :2] - observed[order[0], :2]
    angle = np.degrees(np.arctan2(a[0]*b[1] - a[1]*b[0], np.dot(a, b)))
    return {'observed_indices': list(order), 'xy_residual_mm': residual.tolist(),
            'rms_mm': float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
            'max_mm': float(np.max(np.linalg.norm(residual, axis=1))),
            'row_angle_difference_deg': float(angle)}


def check_jl100_template(report, source):
    spec = importlib.util.spec_from_file_location('archived_fixture_primitives', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [(name, (np.asarray(a) + b) * 500)
            for name, a, b, _ in module.terminal_primitives('JL100') if '_gnd' not in name]
    observed = np.asarray(report['cases']['JL100']['sensitivity'][1]['observed_box_xyz_mm'])
    return {'source': str(source), 'sha256': sha(source),
            'terminal_count_excluding_ground': len(rows),
            'names': [x[0] for x in rows], 'xy_mm': [x[1][:2].tolist() for x in rows],
            'match_against_actual_aug18_observations_at_assumed8mm': nearest_assignment(np.asarray([x[1] for x in rows]), observed),
            'note': 'Template geometry is fixture-derived. It is compared here against independently observed image points, not accepted as measured geometry.'}


def check_rokae132(report, chain_path, vision_path):
    chain = json.loads(chain_path.read_text())
    vision = json.loads(vision_path.read_text())
    if not np.array_equal(vision['camera_T_box'], chain['camera_T_box']):
        raise ValueError('ROKAE image/pose chain does not share the same observation')
    B = np.asarray(chain['observed_box_T_flange'])
    mount = Rotation.from_euler('z', -90, degrees=True).as_matrix()
    R = mount @ Rotation.from_euler('z', -150, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
    xy = np.asarray(report['cases']['JL132']['socket_axis_step_xy_mm'])
    axes = (np.c_[xy, np.zeros(3)] - [0, 0, 6]) @ R.T
    origins = axes @ B[:3, :3].T + B[:3, 3] * 1000
    direction = B[:3, :3] @ R @ np.array([0., 0., -1.])
    pixels = [[1448, 333], [1459, 505], [1472, 677], [1617, 330], [1631, 502], [1648, 664]]
    checks = []
    for depth in [0., 8., 10., 15.]:
        observed = ray_box_points(pixels, vision['live']['intrinsics'], vision['camera_T_box'], depth)
        predicted = origins + ((depth - origins[:, 2]) / direction[2])[:, None] * direction
        checks.append({'assumed_stud_depth_mm': depth,
                       'observed_box_xyz_mm': observed.tolist(),
                       'socket_axes_at_plane_mm': predicted.tolist(),
                       'matching': nearest_assignment(predicted, observed)})
    return {'result_source': str(vision_path), 'result_sha256': sha(vision_path),
            'chain_source': str(chain_path), 'chain_sha256': sha(chain_path),
            'task_id': vision['task_id'], 'capture': vision['live']['captured_at'],
            'observed_tip_pixels': pixels, 'sensitivity': checks,
            'note': 'Real observed box_T_flange plus current Rz(-90) flange_T_TCP; independently handpicked same-task RGB pixels. Pixel/depth uncertainty applies.'}


def plot_report(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    for ax, (fid, item) in zip(axes, report['cases'].items()):
        check = item['sensitivity'][1]
        observed = np.asarray(check['observed_box_xyz_mm'])
        predicted = np.asarray(check['socket_axes_at_plane_mm'])
        ground = item['ground_assumed_depth20_box_xyz_mm']
        ax.scatter(*observed[:, :2].T, color='#1865af', s=48, label='Observed six stud tips')
        ax.scatter(*predicted[:, :2].T, marker='x', color='#d64b26', s=85, label='STEP socket axes / real TCP')
        ax.scatter(*ground[:2], marker='s', color='#3b8a3e', s=40, label='Observed ground side')
        for index, xy in enumerate(observed):
            ax.annotate(str(index), xy[:2], xytext=(5, 4), textcoords='offset points')
        ax.axhline(0, color='grey', linewidth=.5)
        ax.axvline(0, color='grey', linewidth=.5)
        ax.set(xlabel='Saved box X (mm)', ylabel='Saved box Y (mm)', aspect='equal',
               title=f"{fid}: row difference {check['matching']['row_angle_difference_deg']:.2f} deg")
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc='outside lower center', ncol=3)
    fig.suptitle('Historical real pose + independently reviewed pixels; assumed stud depth 8 mm\nDirection audit only: manual pixels and unknown depth do not establish insertion accuracy', fontsize=10)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    # Cache may be extracted with --extract-step-axes in the OCP environment.
    if args.extract_step_axes:
        axes = extract_step_axes(args.step)
        doc = {'source': str(args.step), 'sha256': sha(args.step),
               'axis_xy_step_mm': axes.tolist(), 'extractor_sha256': sha(__file__)}
        args.axis_cache.write_text(json.dumps(doc, indent=2) + '\n')
        return doc
    axes_doc = json.loads(args.axis_cache.read_text())
    if axes_doc['sha256'] != sha(args.step):
        raise ValueError('STEP axis source changed')
    axes = np.asarray(axes_doc['axis_xy_step_mm'])
    R = Rotation.from_euler('z', -150, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
    report = {'status': 'offline_direction_audit_only', 'source_sha256': sha(__file__),
              'step_source': axes_doc, 'cases': {}, 'controller_writes': False,
              'limitations': [
                  'Manually reviewed stud-tip pixels have roughly 3px uncertainty; JM160 upper stud is partly covered by the axis overlay.',
                  'No raw depth exists in the selected August18 task archives; depth sweep is sensitivity analysis, not measured height.',
                  'Plate D7 bores locate nominal socket axes; actual socket bodies/tip lengths were not independently measured.',
                  'Same-task recorded TCP feedback is compared with observed motor pixels; no terminal is generated from the fixture.',
                  'A matched row direction does not prove precision insertion, contact force, electrical acceptance or motor-to-foot-datum registration.',
              ]}
    for fid, info in CASES.items():
        folder = args.archive / info['task']
        evidence_path = folder / 'evidence.json'
        evidence = json.loads(evidence_path.read_text())
        intrinsic = json.loads((folder / 'camera_intrinsics.json').read_text())
        indices = [int(np.argmin(np.linalg.norm(axes - seed, axis=1))) for seed in info['step_xy_selection']]
        step_xy = axes[indices]
        if max(np.min(np.linalg.norm(axes - seed, axis=1)) for seed in info['step_xy_selection']) > .1:
            raise ValueError('Socket axis could not be identified in STEP')
        # Axis origin at plate z=0. Comparison uses axis-plane intersection,
        # hence deliberately avoids the unmeasured 32mm socket length.
        tcp_axes_mm = (np.c_[step_xy, np.zeros(3)] - [0, 0, 6]) @ R.T
        B = np.asarray(evidence['actual_return_before_box_T_tcp'])
        box_axis_origins = tcp_axes_mm @ B[:3, :3].T + B[:3, 3] * 1000
        box_axis_direction = B[:3, :3] @ R @ np.array([0., 0., -1.])
        results = []
        for depth in [0., 8., 10., 15.]:
            observed = ray_box_points(info['stud_pixels'], intrinsic, evidence['camera_T_box'], depth)
            predicted = box_axis_origins + ((depth - box_axis_origins[:, 2]) / box_axis_direction[2])[:, None] * box_axis_direction
            results.append({'assumed_stud_depth_mm': depth, 'observed_box_xyz_mm': observed.tolist(),
                            'socket_axes_at_plane_mm': predicted.tolist(),
                            'matching': nearest_assignment(predicted, observed)})
        ground = ray_box_points([info['ground_pixel']], intrinsic, evidence['camera_T_box'], 20.)[0]
        report['cases'][fid] = {'task_id': info['task'], 'evidence_path': str(evidence_path),
            'evidence_sha256': sha(evidence_path), 'observed_stud_pixels': info['stud_pixels'],
            'pixel_index_semantics': '0..2 unbridged U1/V1/W1 row,3..5 bridged W2/U2/V2 row; hand reviewed independently of predicted socket positions',
            'ground_pixel': info['ground_pixel'], 'ground_assumed_depth20_box_xyz_mm': ground.tolist(),
            'ground_is_on_minus_y_side': bool(ground[1] < 0),
            'socket_axis_step_xy_mm': step_xy.tolist(), 'sensitivity': results}
    report['current_JL100_template_check'] = check_jl100_template(report, args.simulation_fixtures)
    report['ROKAE132_same_task_direction_check'] = check_rokae132(report, args.rokae_chain, args.rokae_vision)
    output = args.output / 'recorded_fixture_mating.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    plot_report(report, args.output / 'recorded_fixture_direction.png')
    print(json.dumps({fid: x['sensitivity'][1]['matching'] for fid, x in report['cases'].items()}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--step', type=Path, default=STEP)
    parser.add_argument('--output', type=Path, default=ARCHIVE / 'mating-audit')
    parser.add_argument('--axis-cache', type=Path, default=ARCHIVE / 'step_socket_axis_cache.json')
    parser.add_argument('--simulation-fixtures', type=Path, default=Path('/home/wjr/.local/share/jinlong-gpu-planner/ipc/sim-repo/sim/helios_insert/fixtures.py'))
    parser.add_argument('--rokae-chain', type=Path, default=ARCHIVE / 'rokae_132_executed_chain.json')
    parser.add_argument('--rokae-vision', type=Path, default=ARCHIVE.parent / 'ipc/snapshot/outputs/hik_web_pipeline/20261021_174614_e1e6bb/result.json')
    parser.add_argument('--extract-step-axes', action='store_true')
    run(parser.parse_args())
