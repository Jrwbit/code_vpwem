#!/usr/bin/env python3
"""Audit JL100's far outer edge in actual STEP/TCP/flange frames, offline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/diagnostics/actual-terminal-alignment-20261023/archive'
STEP = Path('/home/wjr/code/jinlong-motor-arm-control/docs/tooling/flange_v2.STEP')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(path):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Line
    reader = STEPControl_Reader()
    if int(reader.ReadFile(str(path))) != 1:
        raise ValueError('STEP read failed')
    reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_EDGE)
    curves, lines, seen = [], [], set()
    while explorer.More():
        c = BRepAdaptor_Curve(TopoDS.Edge_s(explorer.Current())); explorer.Next()
        n = 2 if c.GetType() == GeomAbs_Line else 80
        points = []
        for u in np.linspace(c.FirstParameter(), c.LastParameter(), n):
            p = c.Value(float(u)); points.append([p.X(), p.Y(), p.Z()])
        points = np.asarray(points)
        if np.ptp(points[:, 2]) > 1e-5 or min(abs(points[0, 2]), abs(points[0, 2] - 6)) > 1e-5:
            continue
        # XY outlines on either plate face; keep their original STEP directions.
        key = tuple(np.round(np.sort(points[:, :2], axis=0).ravel(), 5))
        if key not in seen:
            curves.append(points.tolist()); seen.add(key)
        if (n == 2 and np.linalg.norm(points[1] - points[0]) > 100 and abs(points[0, 2]) < 1e-5
                and not any(np.allclose(points, old) or np.allclose(points[::-1], old) for old in lines)):
            lines.append(points)
    candidates = [x for x in lines if np.all(np.mean(x, axis=0)[:2] < 0)]
    if len(candidates) != 1:
        raise ValueError('Cannot identify a unique JL100 outer long edge')
    return {'source': str(path), 'sha256': sha(path), 'curves_step_mm': curves,
            'jl100_outer_edge_step_mm': candidates[0].tolist(),
            'selection': 'Unique straight edge >100mm on plate z=0 whose XY midpoint is in the JL100 negative-X/negative-Y sector.'}


def apply(T, points_mm):
    return np.asarray(points_mm) @ T[:3, :3].T + T[:3, 3] * 1000


def parallel_angle(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b), 0., 1.))))


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.extract:
        args.cache.write_text(json.dumps(extract(args.step), indent=2) + '\n')
        return
    geometry = json.loads(args.cache.read_text())
    if geometry['sha256'] != sha(args.step):
        raise ValueError('STEP source changed')
    evidence_path = BASE / '20260818_110510_b0c41b/evidence.json'
    recorded = json.loads(evidence_path.read_text())
    mating = json.loads((BASE / 'mating-audit/recorded_fixture_mating.json').read_text())['cases']['JL100']
    tcp_T_step = np.eye(4)
    tcp_T_step[:3, :3] = Rotation.from_euler('z', -150, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
    tcp_T_step[:3, 3] = -tcp_T_step[:3, :3] @ np.array([0., 0., .006])
    flange_T_tcp = np.eye(4)
    flange_T_tcp[:3, :3] = Rotation.from_euler('z', -90, degrees=True).as_matrix()
    flange_T_step = flange_T_tcp @ tcp_T_step
    box_T_tcp_recorded = np.asarray(recorded['actual_return_before_box_T_tcp'])
    derived_box_T_flange = box_T_tcp_recorded @ np.linalg.inv(flange_T_tcp)
    edge = np.asarray(geometry['jl100_outer_edge_step_mm'])
    if edge[1, 0] < edge[0, 0]:
        edge = edge[::-1]
    edge_tcp, edge_flange = apply(tcp_T_step, edge), apply(flange_T_step, edge)
    edge_box = apply(box_T_tcp_recorded, edge_tcp)
    socket_step = np.c_[mating['socket_axis_step_xy_mm'], np.zeros(3)]
    socket_flange = apply(flange_T_step, socket_step)
    socket_box = apply(box_T_tcp_recorded, apply(tcp_T_step, socket_step))
    nominal_tcp_T_box = np.eye(4); nominal_tcp_T_box[:3, 3] = [.08925, 0., .01123]
    current_nominal_box_T_flange = np.linalg.inv(nominal_tcp_T_box) @ np.linalg.inv(flange_T_tcp)
    direction = edge_flange[1] - edge_flange[0]
    current_edge_direction_box = current_nominal_box_T_flange[:3, :3] @ direction
    report = {
        'source_step': str(args.step), 'source_step_sha256': sha(args.step),
        'recorded_evidence_path': str(evidence_path), 'recorded_evidence_sha256': sha(evidence_path),
        'photo_path': str(evidence_path.parent / 'actual_box_annotated.jpg'),
        'photo_sha256': sha(evidence_path.parent / 'actual_box_annotated.jpg'),
        'audit_sha256': sha(__file__), 'recorded_task': recorded['task_id'],
        'controller_writes': False, 'edge_selection': geometry['selection'],
        'edge_step_mm': edge.tolist(), 'edge_tcp_mm': edge_tcp.tolist(),
        'edge_current_flange_mm': edge_flange.tolist(), 'edge_box_from_recorded_UR_TCP_mm': edge_box.tolist(),
        'edge_length_mm': float(np.linalg.norm(direction)),
        'edge_midpoint_current_flange_mm': edge_flange.mean(0).tolist(),
        'edge_line_planar_distance_from_flange_origin_mm': float(np.linalg.norm(edge_flange.mean(0)[:2])),
        'edge_direction_current_flange_unit': (direction / np.linalg.norm(direction)).tolist(),
        'socket_axis_centres_current_flange_mm': socket_flange.tolist(),
        'parallel_to_three_socket_row_deg': parallel_angle(direction, socket_flange[-1] - socket_flange[0]),
        'recorded_edge_parallel_to_box_Y_deg': parallel_angle(edge_box[1] - edge_box[0], [0., 1., 0.]),
        'current_minus90_nominal_edge_parallel_to_box_Y_deg': parallel_angle(current_edge_direction_box, [0., 1., 0.]),
        'historical_box_T_tcp_observed': box_T_tcp_recorded.tolist(),
        'box_T_flange_derived_for_current_minus90_mount': derived_box_T_flange.tolist(),
        'current_minus90_nominal_box_T_flange': current_nominal_box_T_flange.tolist(),
        'flange_T_tcp': flange_T_tcp.tolist(), 'tcp_T_step': tcp_T_step.tolist(),
        'user_confirmed_constraint': 'JL100 has 2x3 terminals; the direction of the three consecutive terminals is parallel to the motor shaft. The far JL100 plate edge must be parallel to the requested motor-box direction.',
        'limits': ['No current ROKAE JL100 execution was found. box_T_flange here is transferred from historical UR box_T_TCP using the current verified mount, not a measured current JL100 flange pose.',
                   'The photo precedes insertion. Tool edge and flange centre are projections of the real historical TCP record, not tool pixels visible in the photo.',
                   'Motor shaft parallelism is the explicit user constraint; its drawn location is schematic, not a measured shaft line.',
                   'Absolute robot-left orientation must be set by the common workcell transform; this local-frame audit does not infer it from the UR image.'],
    }
    (args.output / 'jl100_edge_frame_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    render(report, geometry, recorded, mating, args.output)
    print(json.dumps({k: report[k] for k in ['edge_direction_current_flange_unit', 'edge_midpoint_current_flange_mm', 'parallel_to_three_socket_row_deg', 'recorded_edge_parallel_to_box_Y_deg', 'current_minus90_nominal_edge_parallel_to_box_Y_deg']}, indent=2))


def render(report, geometry, recorded, mating, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from PIL import Image
    from render_recorded_fixture_overlay import project_box_points
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    intrinsic = json.loads((BASE / '20260818_110510_b0c41b/camera_intrinsics.json').read_text())
    rgb = np.asarray(Image.open(BASE / '20260818_110510_b0c41b/actual_box_annotated.jpg').convert('RGB'))
    camera = recorded['camera_T_box']
    edge_image = project_box_points(report['edge_box_from_recorded_UR_TCP_mm'], camera, intrinsic)
    centre_image = project_box_points([np.asarray(report['historical_box_T_tcp_observed'])[:3, 3] * 1000], camera, intrinsic)[0]
    pixels = np.asarray(mating['observed_stud_pixels'])
    fig, axes = plt.subplots(1, 2, figsize=(17, 10))
    fig.subplots_adjust(top=.82, left=.02, right=.98, bottom=.06, wspace=.12)
    photo, cad = axes
    photo.imshow(rgb)
    photo.scatter(*pixels.T, s=80, facecolors='none', edgecolors='#00dcff', linewidths=1.6)
    photo.plot(*edge_image.T, color='#ff50cf', linewidth=3)
    photo.scatter(*centre_image, marker='*', s=240, color='#ffe13b', edgecolors='black', zorder=5)
    photo.annotate('法兰中心\n（按当前安装换算）', centre_image, xytext=(1800, 400), color='#ffe13b',
                   fontproperties=font, fontsize=11, arrowprops={'arrowstyle': '->', 'color': '#ffe13b'})
    photo.annotate('100最远长边\n（真实TCP回投）', edge_image.mean(0), xytext=(400, 950), color='#ff70df',
                   fontproperties=font, fontsize=11, arrowprops={'arrowstyle': '->', 'color': '#ff70df'})
    photo.plot(*pixels[[0, 2]].T, color='#00dcff', linewidth=1.5)
    photo.annotate('实物2×3；连续三柱方向', pixels[2], xytext=(1500, 1050), color='#00dcff',
                   fontproperties=font, fontsize=11, arrowprops={'arrowstyle': '->', 'color': '#00dcff'})
    # Label a shaft-direction constraint, without pretending to fit a measured shaft axis.
    axis = project_box_points([[0., -165., 0.], [0., 110., 0.]], camera, intrinsic)
    photo.plot(*axis.T, '--', color='#75ff7c', linewidth=1.5)
    photo.annotate('电机轴方向（用户约束：∥三柱）\n虚线位置为示意', axis[0], xytext=(1750, 1550),
                   color='#75ff7c', fontproperties=font, fontsize=11,
                   arrowprops={'arrowstyle': '->', 'color': '#75ff7c'})
    photo.set_title('真实100原图＋2026-08-18实际TCP记录\n照片内没有工装；粉线和星号为回投，法兰系按当前安装换算', fontproperties=font, fontsize=12)
    photo.set_xlim(0, rgb.shape[1]); photo.set_ylim(rgb.shape[0], 0)
    photo.set_xticks([]); photo.set_yticks([])
    T = np.asarray(report['flange_T_tcp']) @ np.asarray(report['tcp_T_step'])
    for curve in geometry['curves_step_mm']:
        points = apply(T, curve)
        cad.plot(points[:, 0], points[:, 1], color='#8b8e94', linewidth=.65, alpha=.8)
    e = np.asarray(report['edge_current_flange_mm'])
    s = np.asarray(report['socket_axis_centres_current_flange_mm'])
    cad.plot(*e[:, :2].T, color='#cb178a', linewidth=4, label='100最远长边')
    cad.plot(*s[:, :2].T, 'o-', color='#087db4', linewidth=1.6, label='100三个套筒轴排列')
    cad.scatter(0, 0, marker='*', s=230, color='#efb510', edgecolors='black', zorder=5)
    cad.annotate('法兰中心 F=(0,0)', [0, 0], xytext=(25, 15), fontproperties=font, fontsize=11)
    cad.annotate('长边中点 F=(0,−125,6) mm\n边长131.987 mm，方向∥F的X轴', e.mean(0)[:2], xytext=(-165, -170),
                 fontproperties=font, fontsize=11, arrowprops={'arrowstyle': '->', 'color': '#cb178a'})
    cad.annotate('套筒三轴与长边平行', s.mean(0)[:2], xytext=(10, -65), fontproperties=font,
                 fontsize=11, arrowprops={'arrowstyle': '->', 'color': '#087db4'})
    cad.arrow(0, 0, 40, 0, width=.5, head_width=5, color='#d13d34')
    cad.arrow(0, 0, 0, 40, width=.5, head_width=5, color='#358d41')
    cad.text(43, -3, '+X_F', color='#d13d34'); cad.text(3, 43, '+Y_F', color='#358d41')
    cad.axhline(0, color='grey', linewidth=.5); cad.axvline(0, color='grey', linewidth=.5)
    cad.set(xlabel='Flange X (mm)', ylabel='Flange Y (mm)', aspect='equal', xlim=(-180, 190), ylim=(-195, 170))
    cad.grid(alpha=.15)
    cad.set_title('真实STEP在当前法兰坐标系中的俯视投影\nF_T_TCP=Rz(−90°)，+X向右、+Y向上；没有旋转凑对齐', fontproperties=font, fontsize=12)
    cad.legend(prop=font, loc='upper left')
    fig.suptitle('100长边、三柱排列与电机轴约束\n当前安装下长边沿法兰±X；在盒坐标中沿±Y，与三柱排列平行。机器人左侧的绝对朝向另由共同工位确定。',
                 fontproperties=font, fontsize=14)
    fig.savefig(output / '100长边与法兰坐标核对.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=Path, default=STEP)
    parser.add_argument('--output', type=Path, default=BASE / 'jl100-edge-audit')
    parser.add_argument('--cache', type=Path, default=BASE / 'jl100-edge-audit/step_edge_cache.json')
    parser.add_argument('--extract', action='store_true')
    main(parser.parse_args())
