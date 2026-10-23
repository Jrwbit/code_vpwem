#!/usr/bin/env python3
"""Inspect saved-video geometry at its saved endpoint; never plan a new route.

Rebuild the archived scene using its pinned collision sources and actual fixture
parameters, then reproduce the visual-only additions from the video renderer.
This audits internal display/collision consistency, not actual motor geometry.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'outputs/diagnostics/shared-station-20261023'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def line_direction(points):
    points = np.asarray(points)
    direction = points[-1] - points[0]
    return direction / np.linalg.norm(direction)


def parallel_error_degrees(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 1))))


def binary_stl_vertices(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError('Expected exact binary STL')
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    return np.unique(np.frombuffer(raw, dtype=dtype, offset=84, count=count)['vertices'].reshape(-1, 3), axis=0)


def fitted_visual_bores(flange_vertices, expected_xy):
    """Fit the D7 visual hole rims near independently named socket axes.

    The seed only selects the three holes. Circle centres are solved from the
    exported visual STL vertices; selecting vertices does not move the mesh.
    """
    vertices = np.asarray(flange_vertices)
    plate = vertices[(vertices[:, 2] >= -1e-6) & (vertices[:, 2] <= .006001)]
    circles = []
    for seed in expected_xy:
        distance = np.linalg.norm(plate[:, :2] - seed, axis=1)
        points = np.unique(plate[np.abs(distance - .0035) < .00012, :2], axis=0)
        if len(points) < 10:
            raise ValueError('Not enough visual D7 bore vertices to inspect the chosen socket')
        # x²+y² = 2*cx*x + 2*cy*y + (r²-cx²-cy²).
        solution, _, _, _ = np.linalg.lstsq(np.c_[2*points, np.ones(len(points))],
                                          np.sum(points**2, axis=1), rcond=None)
        center = solution[:2]
        radius = float(np.sqrt(solution[2] + center @ center))
        residual = np.linalg.norm(points - center, axis=1) - radius
        circles.append({'center_flange_m': [*center, .003], 'radius_mm': radius * 1000,
                        'vertices': len(points), 'rms_circle_residual_mm': float(np.sqrt(np.mean(residual**2))) * 1000,
                        'offset_from_rounded_collision_socket_xy_mm': ((center - seed) * 1000).tolist()})
    return circles


def run(args):
    result_path = args.old / 'root-final-reproduce/result.json'
    scene_path = args.old / 'video/scenes' / args.model / 'scene_used.json'
    manifest_path = args.old / 'video/video_manifest.json'
    station_path = args.old / 'geometry/shared_station_geometry.json'
    visual_path = args.old / 'geometry/motor_visuals.json'
    tool_visual_path = args.old / 'geometry/tool_visual.json'
    result, scene_doc, manifest, geometry = [json.loads(p.read_text()) for p in
                                            (result_path, scene_path, manifest_path, station_path)]
    if sha(result_path) != manifest['source_result_sha256']:
        raise ValueError('The saved result differs from the published-video manifest')
    values = result['run_arguments']
    sim, control, vision = [Path(values[k]) for k in ('sim_root', 'control_root', 'vision_root')]
    sys.path[:0] = list(map(str, (control, sim, vision)))
    os.environ.setdefault('MUJOCO_GL', 'egl')
    # These sources directly build geometry. The modified station solver is not
    # imported or executed: it is unnecessary to inspect saved q and scene.
    pinned = {}
    for name in ('scene.py', 'fixtures.py'):
        path = sim / 'sim/helios_insert' / name
        expected = manifest['sources_sha256_at_render'][str(path)]
        if sha(path) != expected:
            raise ValueError('Pinned simulator builder changed: ' + str(path))
        pinned[str(path)] = expected
    for name in ('build_measured_box_geometry.py', 'build_tool_geometry.py', 'solve_fixed_waist_cycle.py'):
        path = Path(__file__).with_name(name)
        expected = manifest['sources_sha256_at_render'][str(path)]
        if sha(path) != expected:
            raise ValueError('Pinned geometry source changed: ' + str(path))
        pinned[str(path)] = expected
    import mujoco
    from sim.helios_insert import scene, fixtures
    from helios_hmi.controller_kinematics import apply_collision_model
    from hk_pointcloud_registration.robot.corridor_worker import configure_geometry
    from hk_pointcloud_registration.robot.tilted_search import selected_terminal_model
    import build_measured_box_geometry as box_geometry
    from solve_fixed_waist_cycle import bounded_contact_horizon
    from render_shared_station_video import display_model
    renderer = Path(__file__).with_name('render_shared_station_video.py')
    if sha(renderer) != manifest['renderer_sha256']:
        raise ValueError('Video display-model construction source changed')
    pinned[str(renderer)] = sha(renderer)
    bundle_path = ROOT / values['bundle']
    bundle = json.loads(bundle_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(scene_doc)
    configure_geometry(cfg, fixtures, bundle['site'], sim, args.output)
    if sha(cfg['tool_mesh']) != manifest['geometry_sha256_at_build'][scene_doc['tool_mesh']]:
        raise ValueError('Reconstructed actual-mount tool hulls differ from video')
    for key in ('model_specific_box_collision', 'tool_cad_collision'):
        path = cfg[key]
        if sha(path) != manifest['geometry_sha256_at_build'][path]:
            raise ValueError('Saved collision input changed: ' + path)
        pinned[path] = sha(path)
    scene.VENDOR = control / 'helios_hmi/model/Wheel_Robot'
    captured = []
    old_modify = box_geometry.modify_spec

    def capture(spec, cfg, fid):
        spec = old_modify(spec, cfg, fid)
        captured.append(spec)
        return spec

    box_geometry.modify_spec = capture
    try:
        collision = box_geometry.build_model(cfg, args.model, scene.build)
    finally:
        box_geometry.modify_spec = old_modify
    collision = selected_terminal_model(apply_collision_model(collision, bundle['site']), args.model)
    horizon = bounded_contact_horizon(collision) if values['bounded_contact_horizon'] else None
    captured[-1].copy_back(collision)
    station = np.asarray(result['candidate']['world_T_station'])
    visuals, tool_visual = json.loads(visual_path.read_text()), json.loads(tool_visual_path.read_text())
    visual_entry = visuals['models'][args.model] if 'models' in visuals else visuals[args.model]
    display, display_info = display_model(mujoco, collision, captured[-1], station, args.model,
                                         visual_entry, tool_visual, args.output)
    q = np.asarray(result['candidate']['paths'][args.model]['outward_arm_rad'][0])
    waist = np.radians(result['candidate']['waist_model_deg'])
    data = mujoco.MjData(display)
    for names, positions in [(scene.WAIST, waist), (scene.LEFT, q),
                             (scene.HEAD, cfg['static_pose']['head']),
                             (scene.RIGHT, cfg['static_pose']['right_arm'])]:
        for name, position in zip(names, positions):
            data.qpos[display.joint(name).qposadr[0]] = position
    mujoco.mj_forward(display, data)
    flange_id = display.body('AR5-5_07L_tcp').id
    F = np.eye(4)
    F[:3, :3] = data.xmat[flange_id].reshape(3, 3)
    F[:3, 3] = data.xpos[flange_id]

    def centres(prefix):
        return np.asarray([data.geom_xpos[display.geom(f'{prefix}_{i}').id] for i in range(1, 4)])

    studs = centres(f'term_{args.model}_stud')
    sockets = centres(f'tool_sock_{args.model}')
    shaft = station[:3, :3] @ geometry['model_datums'][args.model]['shaft_direction_station']
    box = np.asarray(scene_doc['box_T_base_link'])
    actual_visual = display.geom('actual_mount_tool_visual')
    mesh_id = display.geom_dataid[actual_visual.id]
    mesh_vertices = display.mesh_vert[display.mesh_vertadr[mesh_id]:display.mesh_vertadr[mesh_id]+display.mesh_vertnum[mesh_id]]
    world_vertices = mesh_vertices @ data.geom_xmat[actual_visual.id].reshape(3, 3).T + data.geom_xpos[actual_visual.id]
    flange_vertices = (world_vertices - F[:3, 3]) @ F[:3, :3]
    expected_xy = np.asarray([a[:2] for a, b in fixtures.tool_parts_flange()[args.model]['sockets']])
    bores = fitted_visual_bores(flange_vertices, expected_xy)
    bore_centers = np.asarray([x['center_flange_m'] for x in bores])
    bore_world = bore_centers @ F[:3, :3].T + F[:3, 3]
    from scipy.spatial import cKDTree
    raw_visual = binary_stl_vertices(tool_visual['path'])
    mesh_error = float(cKDTree(raw_visual).query(flange_vertices)[0].max())
    target = np.asarray(scene_doc['targets'][0]['T'])
    row_dirs = {'terminal_row_world': line_direction(studs).tolist(),
                'socket_row_world': line_direction(sockets).tolist(),
                'visual_bore_row_world': line_direction(bore_world).tolist(), 'shaft_world': shaft.tolist()}
    angles = {key.replace('_world', '_vs_shaft_deg'): parallel_error_degrees(direction, shaft)
              for key, direction in row_dirs.items() if key != 'shaft_world'}
    angles['visual_bores_vs_collision_sockets_deg'] = parallel_error_degrees(line_direction(bore_world), line_direction(sockets))
    angles['socket_vs_terminal_row_deg'] = parallel_error_degrees(line_direction(sockets), line_direction(studs))
    outer_edge = None
    if args.model == 'JL100':
        plate = flange_vertices[(flange_vertices[:, 2] >= -1e-6) & (flange_vertices[:, 2] <= .006001)]
        outer_vertices = plate[np.abs(plate[:, 1] - plate[:, 1].min()) < 1e-7]
        endpoints = outer_vertices[[np.argmin(outer_vertices[:, 0]), np.argmax(outer_vertices[:, 0])]].copy()
        endpoints[:, 2] = 0
        world_endpoints = endpoints @ F[:3, :3].T + F[:3, 3]
        outer_edge = {'selection': 'Actual STL plate-face outer straight edge at minimum flange Y',
                      'flange_endpoints_mm': (endpoints * 1000).tolist(),
                      'world_endpoints_mm': (world_endpoints * 1000).tolist(),
                      'length_mm': float(np.linalg.norm(endpoints[1] - endpoints[0]) * 1000),
                      'vs_shaft_parallel_error_deg': parallel_error_degrees(line_direction(world_endpoints), shaft)}
    output = {
        'status': 'saved_video_internal_geometry_audit_not_real_motor_registration', 'model': args.model,
        'source_result': str(result_path), 'source_result_sha256': sha(result_path),
        'saved_scene': str(scene_path), 'saved_scene_sha256': sha(scene_path),
        'source_video_manifest_sha256': sha(manifest_path), 'adapter_sha256': sha(__file__),
        'geometry_sources_verified_sha256': pinned, 'new_planning_or_hardware_commands': False,
        'saved_endpoint_left_rad': q.tolist(), 'saved_waist_deg': result['candidate']['waist_model_deg'],
        'actual_flange_world_T': F.tolist(), 'saved_target_flange_world_T': target.tolist(),
        'endpoint_position_error_mm': float(np.linalg.norm(F[:3, 3] - target[:3, 3]) * 1000),
        'box_frame_world_T': box.tolist(), 'station_frame_world_T': station.tolist(),
        'terminal_centers_world_mm': (studs * 1000).tolist(),
        'socket_centers_world_mm': (sockets * 1000).tolist(),
        'visual_bore_centers_world_mm': (bore_world * 1000).tolist(),
        'directions': row_dirs, 'parallel_errors': angles,
        'visual_bore_fit': bores, 'compiled_visual_to_original_STL_max_vertex_distance_mm': mesh_error * 1000,
        'JL100_actual_visual_outer_long_edge': outer_edge,
        'same_saved_contact_horizon_applied': horizon,
        'display_motor': display_info,
        'limitations': [
            'This inspects the saved 57-second video model, not an independently registered physical motor.',
            'Old JL100/JL132 terminal positions are generated by fixtures.terminals_box from the tool; internal consistency is not physical evidence.',
            'Bore centres are fitted from the actual visual STL; they describe tool-hole alignment, not unmeasured socket tip geometry.',
            'The same saved endpoint and DH parameters are used. No route is replanned and no prior artifacts are changed.',
        ],
    }
    destination = args.output / f'{args.model}_saved_video_direction_audit.json'
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    make_figure(args, output, binary_stl_vertices(visual_entry['path']), station)
    print(json.dumps({'output': str(destination), 'parallel_errors': angles,
                      'compiled_visual_max_error_mm': mesh_error * 1000}, ensure_ascii=False, indent=2))
    return output


def make_figure(args, audit, motor_vertices, station):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/jinlong-direction-audit-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from scipy.spatial import ConvexHull
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    preview = args.old / f'video/{args.model}_endpoint_preview.png'
    from PIL import Image
    axes[0].imshow(np.asarray(Image.open(preview))[112:752, 720:1440])
    axes[0].axis('off')
    axes[0].set_title('原57秒视频保存终点（历史方案）', fontproperties=font)
    xy = (motor_vertices @ station[:3, :3].T + station[:3, 3])[:, :2] * 1000
    outline = xy[ConvexHull(xy).vertices]
    axis = axes[1]
    axis.fill(outline[:, 0], outline[:, 1], color='#d8e3e8', label='视频电机网格俯视包络')
    shaft = np.asarray(audit['directions']['shaft_world'])[:2]
    start = station[:2, 3] * 1000
    axis.arrow(*start, *(shaft * 100), width=.7, head_width=5, color='#34495e', length_includes_head=True)
    axis.text(*(start + shaft * 105), '轴向', fontproperties=font, fontsize=10)
    for key, label, style in [('terminal_centers_world_mm', '旧碰撞端子列', 'o-'),
                              ('visual_bore_centers_world_mm', '实际显示STL三孔', 'x--')]:
        p = np.asarray(audit[key])
        axis.plot(p[:, 0], p[:, 1], style, label=label, ms=8, lw=1.5)
    edge = audit['JL100_actual_visual_outer_long_edge']
    if edge:
        p = np.asarray(edge['world_endpoints_mm'])
        axis.plot(p[:, 0], p[:, 1], '-', color='#d6a127', lw=3, label='实际显示工装外长边')
    axis.set_aspect('equal')
    axis.grid(alpha=.25)
    axis.set_xlabel('World X (mm)'); axis.set_ylabel('World Y (mm)')
    axis.legend(prop=font, fontsize=9)
    axis.set_title('保存关节位置下的真实模型投影\n三孔/外长边平行轴；端子列差约0.026°', fontproperties=font)
    figure.suptitle(f'{args.model} 旧视频内部方向审计', fontproperties=font, fontsize=17)
    figure.text(.5, .035, '该图只核对旧视频内部一致性。旧端子由工装反推，不能证明真实电机配准或插接成功。',
                fontproperties=font, ha='center', fontsize=11)
    figure.tight_layout(rect=(0, .08, 1, .95))
    figure.savefig(args.output / f'{args.model}_saved_video_direction_audit.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, default=OLD)
    parser.add_argument('--model', choices=['JL100', 'JL132', 'JM160'], default='JL100')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/diagnostics/actual-terminal-alignment-20261023/saved-video-audit')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
