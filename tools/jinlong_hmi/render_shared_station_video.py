#!/usr/bin/env python3
"""Render saved shared-station cycles and actual horizontal-motor CAD visuals.

The source result must validate all three cycles from the same home and waist.
Only saved, checked joint edges are interpolated; no new IK or hardware commands
are used. Motor meshes are visual-only additions to a separate display model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_fixed_waist_video import WIDTH, HEIGHT, PANEL_HEIGHT, FONT, COLOURS, camera, card

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ('JM160', 'JL132', 'JL100')
DESIGN_CANDIDATE_LABEL = '实扫盒体／CAD底座候选，未实物配准'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_lift_label(result):
    """Describe only a verified laboratory-Z camera relocation in the result."""
    meta = result.get('frame_metadata') or {}
    lift_mm = float(meta.get('camera_lift_design_mm', 0))
    if not np.isfinite(lift_mm) or lift_mm < 0:
        raise ValueError('Camera lift metadata must be finite and nonnegative')
    if lift_mm == 0:
        return None
    calibrated = np.asarray(meta['calibration_lab_T_camera'], float)
    candidate = np.asarray(meta['lab_T_camera'], float)
    if (calibrated.shape != (4, 4) or candidate.shape != (4, 4)
            or not np.isfinite(calibrated).all() or not np.isfinite(candidate).all()
            or not np.allclose(calibrated[3], [0, 0, 0, 1], atol=1e-10, rtol=0)):
        raise ValueError('Camera lift requires finite homogeneous laboratory transforms')
    expected = calibrated.copy()
    expected[2, 3] += lift_mm / 1000
    if not np.allclose(candidate, expected, atol=1e-10, rtol=0):
        raise ValueError('Camera lift changed XY or orientation, or disagrees with saved height')
    return f'相机仅竖直抬高 {lift_mm:g} mm，水平位置与朝向不变'


def study_args(result, output):
    """Restore typed arguments from the validated run, changing only output."""
    from solve_shared_motor_station import argument_parser
    values = result['run_arguments'].copy()
    for action in argument_parser()._actions:
        if action.type is Path and values.get(action.dest) is not None:
            value = values[action.dest]
            values[action.dest] = [Path(v) for v in value] if isinstance(value, list) else Path(value)
    values['output'] = output
    return SimpleNamespace(**values)


def check_result(result, *, design_candidate=False):
    if not result.get('full_common_home_cycles_verified') or not result.get('full_noncontact_cycles_verified'):
        raise ValueError('All three validated common-home noncontact cycles are required')
    if design_candidate:
        if result.get('station_geometry_confirmed') is not False or result.get('physical_motor_registration_confirmed') is not False:
            raise ValueError('Design candidate mode requires explicitly unconfirmed station geometry and physical registration')
    elif not result.get('station_geometry_confirmed'):
        raise ValueError('Provisional intrinsic motor geometry cannot be presented as a solution')
    c = result['candidate']
    retract = c.get('endpoint_retract_mm')
    if not isinstance(retract, (int, float)) or not np.isfinite(retract) or retract <= 0:
        raise ValueError('A positive, finite saved noncontact endpoint standoff is required')
    if not c['common_home_validation']['passed']:
        raise ValueError('Common-home validation did not pass')
    source_hashes = result['sources_sha256_at_load']
    if not source_hashes or source_hashes != result['sources_sha256_at_finish']:
        raise ValueError('The original validation changed its input sources')
    for path, expected in source_hashes.items():
        if sha(path) != expected:
            raise ValueError('Source changed after validation: ' + path)
    return c


def sample_edges(q, positions, fraction):
    q, positions = np.asarray(q), np.asarray(positions)
    if len(q) == 1:
        return q[0]
    lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    # A connector may rotate in place. Use joint distance for that edge.
    joint_lengths = np.linalg.norm(np.diff(q, axis=0), axis=1)
    lengths = np.where(lengths > 1e-12, lengths, joint_lengths)
    cumulative = np.r_[0., np.cumsum(lengths)]
    if cumulative[-1] <= 1e-12:
        return q[0]
    distance = np.clip(fraction, 0., 1.) * cumulative[-1]
    edge = int(np.clip(np.searchsorted(cumulative, distance, side='right') - 1, 0, len(q)-2))
    u = (distance-cumulative[edge])/lengths[edge] if lengths[edge] else 0.
    return q[edge] + np.clip(u, 0., 1.) * (q[edge+1]-q[edge])


def schedule(candidate, fid, fps):
    path = candidate['paths'][fid]
    q = np.asarray(path['outward_arm_rad'])
    xyz = np.asarray(path['outward_flange_poses'])[:, :3, 3]
    all_connector = np.asarray(path['common_home_connector_waist_arm_rad'])
    waist = np.radians(candidate['waist_model_deg'])
    if not np.allclose(all_connector[:, :4], waist, atol=1e-10, rtol=0):
        raise ValueError('Saved connector moved the waist')
    connector = all_connector[:, 4:]
    home = np.asarray(candidate['common_home_validation']['common_home_arm_rad'])
    if not np.allclose(connector[0], home) or not np.allclose(connector[-1], q[-1]):
        raise ValueError('Saved home connector does not join this model route')
    away = np.flatnonzero(np.linalg.norm(xyz[:, :2]-xyz[0, :2], axis=1) > 1e-7)
    split = int(away[0]-1) if len(away) else len(q)-1
    if split < 1 or split >= len(q)-1:
        raise ValueError('Expected vertical approach followed by a lateral corridor')
    phases = [('共同等待 / 拍照位置', 1., home[None], home[None])]
    if np.max(np.abs(connector-home)) > 1e-10:
        phases.append(('转入本型号接近路线', 2., connector, connector))
    phases.extend([
        ('水平接近', 3.5, q[split:][::-1], xyz[split:][::-1]),
        ('垂直接近', 1.5, q[:split+1][::-1], xyz[:split+1][::-1]),
        ('非接触终点停留', 2., q[:1], xyz[:1]),
        ('垂直撤离', 1.5, q[:split+1], xyz[:split+1]),
        ('退出本型号接近路线', 3.5, q[split:], xyz[split:]),
    ])
    if np.max(np.abs(connector-home)) > 1e-10:
        phases.append(('返回共同等待位', 2., connector[::-1], connector[::-1]))
    phases.append(('共同等待位 / 本型结束', 1., home[None], home[None]))
    frames = []
    for label, seconds, joints, positions in phases:
        count = round(seconds*fps)
        for index in range(count):
            frames.append((label, sample_edges(joints, positions, index/max(1, count-1))))
    if not np.array_equal(frames[0][1], frames[-1][1]):
        raise ValueError('The visualized cycle does not close at the shared home')
    return frames


def check_geometry(result, study):
    expected_hashes = result['geometry_sha256_at_build']
    for path, value in study.geometry_hashes.items():
        expected = expected_hashes.get(path)
        if expected is None and Path(path).name == 'rotated_tool_hulls.json':
            suffix = '/' + '/'.join(Path(path).parts[-2:])
            matches = [h for p, h in expected_hashes.items() if p.endswith(suffix)]
            expected = matches[0] if len(set(matches)) == 1 else None
        if expected != value:
            raise ValueError('Solid geometry differs from the validated result: ' + path)


def build_with_spec(study, fid, boxes, world_camera, folder):
    """Retain the builder's MjSpec without changing its compiled model."""
    captured = []
    geometry_hook = study.geometry
    if not hasattr(geometry_hook, 'modify_spec'):
        # The observed-frame adapter delegates compilation to this exact hook.
        # Capture that delegated MjSpec without bypassing frame/SHA validation.
        if getattr(geometry_hook, 'SCHEMA_VERSION', None) != 'observed-terminal-geometry/v1':
            raise ValueError('Unknown geometry adapter: cannot capture its verified compilation')
        import build_measured_box_geometry
        geometry_hook = build_measured_box_geometry
    original = geometry_hook.modify_spec
    def capture(spec, scene, model_id):
        result = original(spec, scene, model_id)
        captured.append(result)
        return result
    geometry_hook.modify_spec = capture
    try:
        ctx = study.build(fid, boxes, world_camera, folder)
    finally:
        geometry_hook.modify_spec = original
    if len(captured) != 2:
        raise ValueError('Expected zone-model and robot-model compilations')
    # Controller DH frames, terminal selection and contact horizons are applied
    # to the compiled model. Preserve those exact changes before adding visuals.
    captured[-1].copy_back(ctx.robot.m)
    return ctx, captured[-1]


def display_model(mujoco, original, spec, station, fid, visual, tool_visual, folder, *, observed_box=False):
    """Add a visual motor without editing any original contact property."""
    source = Path(visual['path'])
    if (visual['units'], visual['frame'], visual['visual_only']) != ('metres', 'station', True):
        raise ValueError('Expected visual-only motor mesh in station metres')
    if sha(source) != visual['sha256']:
        raise ValueError('Motor visual mesh hash changed: ' + str(source))
    # OBJ preserves every face and avoids MuJoCo's binary STL face-count limit.
    obj = folder / (fid + '_display.obj')
    raw = source.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84 + 50*count:
        raise ValueError('Expected a binary STL motor export')
    triangle_dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    triangles = np.frombuffer(raw, dtype=triangle_dtype, offset=84, count=count)
    vertices, inverse = np.unique(triangles['vertices'].reshape(-1, 3), axis=0, return_inverse=True)
    with obj.open('w') as handle:
        np.savetxt(handle, vertices, fmt='v %.9g %.9g %.9g')
        np.savetxt(handle, inverse.reshape(-1, 3)+1, fmt='f %d %d %d')
    spec = spec.copy()
    mesh_name = fid + '_motor_visual_mesh'
    spec.add_mesh(name=mesh_name, file=str(obj.resolve()), maxhullvert=64)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(station[:3, :3]).reshape(9))
    spec.worldbody.add_geom(name=fid+'_motor_visual', type=mujoco.mjtGeom.mjGEOM_MESH,
                            meshname=mesh_name, pos=station[:3, 3], quat=quat,
                            contype=0, conaffinity=0, mass=0, group=0,
                            rgba=[.20, .34, .42, 1])
    if (tool_visual['units'], tool_visual['frame'], tool_visual['visual_only']) != ('metres', 'flange', True):
        raise ValueError('Expected visual-only tool mesh in actual flange metres')
    if sha(tool_visual['path']) != tool_visual['sha256']:
        raise ValueError('Actual-mounting tool visual mesh changed')
    spec.add_mesh(name='actual_mount_tool_visual_mesh', file=tool_visual['path'], maxhullvert=64)
    spec.body('AR5-5_07L_tcp').add_geom(
        name='actual_mount_tool_visual', type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname='actual_mount_tool_visual_mesh', contype=0, conaffinity=0, mass=0,
        group=0, rgba=[.85, .62, .25, 1])
    display = spec.compile()
    if (display.nq, display.nv) != (original.nq, original.nv):
        raise ValueError('Visual mesh unexpectedly changed joint state dimensions')
    # Recompilation may reorder world geoms. Names identify named geometry;
    # body and ordinal identify anonymous vendor meshes and the floor plane.
    anonymous = {}
    for i in range(display.ngeom):
        if not display.geom(i).name:
            body = display.body(display.geom_bodyid[i]).name
            anonymous.setdefault(body, []).append(i)
    original_unnamed_counts = {}
    for i in range(original.ngeom):
        name = original.geom(i).name
        if name:
            j = mujoco.mj_name2id(display, mujoco.mjtObj.mjOBJ_GEOM, name)
        else:
            body = original.body(original.geom_bodyid[i]).name
            ordinal = original_unnamed_counts.get(body, 0)
            j = anonymous.get(body, [])[ordinal]
            original_unnamed_counts[body] = ordinal+1
        if j < 0:
            raise ValueError('Original geometry missing from display model: ' + name)
        # MjSpec.copy_back preserves positions and margins but not the runtime
        # terminal-selection masks. Restore those exact validated mask values.
        display.geom_contype[j] = original.geom_contype[i]
        display.geom_conaffinity[j] = original.geom_conaffinity[i]
        for field in ('geom_contype', 'geom_conaffinity', 'geom_margin', 'geom_gap', 'geom_size', 'geom_pos', 'geom_quat'):
            if not np.allclose(getattr(display, field)[j], getattr(original, field)[i], atol=1e-6, rtol=1e-6):
                raise ValueError('Visual construction changed original collision property: '+name+'/'+field)
        if name == 'motor_body' or name.startswith('motor_fitting_'):
            display.geom_rgba[j, 3] = 0
        if observed_box and name.startswith('box_' + fid + '_'):
            # Reveal the independently measured six studs inside the occupied
            # scan columns. This changes appearance only, never collision.
            display.geom_rgba[j, 3] = .18
        if observed_box and name.startswith('term_' + fid + '_'):
            display.geom_rgba[j] = ([.15, .85, .35, 1] if name.endswith('_gnd')
                                    else [.92, .35, .15, 1] if '_stud_' in name
                                    else [.95, .75, .20, 1])
        if any(name.startswith('term_'+other+'_') for other in FIXTURES if other != fid):
            display.geom_rgba[j, 3] = 0
        if name == 'tool_plate_visual':
            # The simulator's decorative STL is fixed at its legacy mounting
            # angle. The newly exported actual-mounting CAD visual replaces it.
            display.geom_rgba[j, 3] = 0
        if name.startswith('tool_disc') or name in tool_visual['hidden_collision_display_geoms']:
            display.geom_rgba[j, 3] = 0
    for field in ('body_pos', 'body_quat', 'jnt_pos', 'jnt_axis'):
        if not np.allclose(getattr(display, field), getattr(original, field), atol=1e-8, rtol=1e-8):
            raise ValueError('Visual construction changed robot kinematics: '+field)
    display.vis.global_.offwidth = max(WIDTH, display.vis.global_.offwidth)
    display.vis.global_.offheight = max(HEIGHT, display.vis.global_.offheight)
    return display, {'source': str(source), 'sha256': visual['sha256'],
                     'display_mesh': str(obj), 'display_mesh_sha256': sha(obj),
                     'added_contype': 0, 'added_conaffinity': 0,
                     'original_collision_properties_preserved': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--visuals', type=Path, default=ROOT/'outputs/diagnostics/shared-station-20261023/geometry/motor_visuals.json')
    parser.add_argument('--tool-visual', type=Path, default=ROOT/'outputs/diagnostics/shared-station-20261023/geometry/tool_visual.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--design-candidate', action='store_true',
                        help='Render verified common-home cycles with observed scan boxes and explicitly unregistered nominal CAD station')
    a = parser.parse_args()
    if not 1 <= a.fps <= 60:
        parser.error('FPS must be in [1, 60]')
    a.result = a.result.resolve()
    a.visuals = a.visuals.resolve()
    a.tool_visual = a.tool_visual.resolve()
    a.output = a.output.resolve()
    result = json.loads(a.result.read_text())
    c = check_result(result, design_candidate=a.design_candidate)
    lift_label = camera_lift_label(result)
    retract_mm = c['endpoint_retract_mm']
    a.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from solve_shared_motor_station import SharedStationStudy
    import mujoco
    study = SharedStationStudy(study_args(result, a.output/'scenes'))
    if a.design_candidate and getattr(study.geometry, 'SCHEMA_VERSION', None) != 'observed-terminal-geometry/v1':
        raise ValueError('Design candidate presentation requires the independent observed-terminal geometry adapter')
    for path, value in study.loaded_source_hashes.items():
        if result['sources_sha256_at_load'].get(path) != value:
            raise ValueError('Renderer loaded an unvalidated source: '+path)
    study.station = np.asarray(c['world_T_station'])
    boxes = {fid: np.asarray(T) for fid, T in c['world_T_boxes'].items()}
    world_camera = np.asarray(c['world_T_camera'])
    waist = np.radians(c['waist_model_deg'])
    visuals = json.loads(a.visuals.read_text())
    tool_visual = json.loads(a.tool_visual.read_text())
    for path_key, hash_key in (('source', 'source_sha256'),
                               ('legacy_visual_and_hull_builder', 'legacy_builder_sha256'),
                               ('independent_boss_cad_builder', 'boss_builder_sha256')):
        if sha(tool_visual[path_key]) != tool_visual[hash_key]:
            raise ValueError('Tool visual registration source changed: '+tool_visual[path_key])
    if not np.allclose(tool_visual['flange_T_tcp'], study.mount, rtol=0, atol=1e-12):
        raise ValueError('Tool visual has the wrong flange mounting')
    visual_evidence = {}
    fonts = {s: ImageFont.truetype(FONT, s) for s in (23, 26, 29, 38, 48)}
    video = a.output/'三型号同一工位固定腰往返仿真.mp4'
    encoder = None
    if not a.preview_only:
        encoder = subprocess.Popen([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(a.fps), '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt',
            'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    total_frames = 0
    def emit(frame, repeats=1):
        nonlocal total_frames
        if encoder:
            for _ in range(repeats):
                encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                total_frames += 1

    intro_lines = [
        (136, '三型号电机 · 同一固定工位', 48, '#f3f7fb'),
        (240, '160 / 132 / 100   一次上一个电机', 38, '#8cd8e6'),
        (324, '共用底脚定位中心、摆放方向和固定台面', 29, '#d3dce7'),
        (378, '电机轴水平 · 接线盒口朝上、位于远离机器人一侧', 29, '#d3dce7'),
        (432, '共同等待位与固定腰姿；不同型号使用各自接近路线', 29, '#d3dce7'),
        (532, f'终点在名义配合目标前 {retract_mm:g} mm；尚未验证真实插接及电测。', 26, '#edbc79'),
        (586, '使用存档 CAD 的机身外观；播放速度不代表实机执行周期。', 26, '#a8b5c5'),
    ]
    if a.design_candidate:
        intro_lines.extend([
            (480, DESIGN_CANDIDATE_LABEL, 26, '#f1c16e'),
            (636, '机身至盒口位置及输出轴正负方向仍为设计候选，未完成实物配准。', 23, '#a8b5c5'),
        ])
    if lift_label:
        intro_lines.append((688, lift_label, 26, '#8cd8e6'))
    intro_frame = card(intro_lines, fonts)
    if lift_label:
        Image.fromarray(intro_frame).save(a.output/'intro_preview.png')
    emit(intro_frame, 3*a.fps)
    waist_label = '[' + ', '.join(f'{v:g}°' for v in c['waist_model_deg']) + ']'
    station_mm = study.station[:3, 3] * 1000
    table_height_mm = (study.station[2, 3] - study.base['floor_z']) * 1000
    route_manifest = {}
    try:
        for index, fid in enumerate(FIXTURES):
            folder = a.output/'scenes'/fid
            ctx, spec = build_with_spec(study, fid, boxes, world_camera, folder)
            if a.design_candidate and (ctx.cfg.get('physical_motor_registration_confirmed') is not False or ctx.cfg.get('terminals') is not False):
                raise ValueError('Candidate scene must retain independent terminals and unconfirmed physical registration')
            check_geometry(result, study)
            model, visual_evidence[fid] = display_model(mujoco, ctx.robot.m, spec, study.station, fid, visuals[fid], tool_visual, folder,
                                                       observed_box=a.design_candidate)
            data = mujoco.MjData(model)
            view = mujoco.Renderer(model, PANEL_HEIGHT, WIDTH//2)
            opt = mujoco.MjvOption(); opt.sitegroup[:] = 0
            center = study.station[:3, 3]
            wide = camera(mujoco, [center[0]*.55, center[1]*.5, .83], 2.20, 215, -15)
            close = camera(mujoco, boxes[fid][:3, 3] + [0, 0, .035], .77, 225, -25)
            frames = schedule(c, fid, a.fps)
            endpoint_index = next(i for i, row in enumerate(frames) if row[0] == '非接触终点停留')
            wanted = [0, endpoint_index] if a.preview_only else range(len(frames))
            samples = c['common_home_validation']['fixtures'][fid]['samples']
            route_manifest[fid] = {'presentation_frames': len(frames), 'validated_samples': samples,
                                   'endpoint_frame': endpoint_index}
            previous, panels = None, None
            try:
                for frame_index in wanted:
                    phase, q = frames[frame_index]
                    if previous is None or not np.array_equal(previous, q):
                        ctx.robot.set(study.cr.WAIST, waist); ctx.robot.set(study.cr.LEFT, q); ctx.robot.fk()
                        data.qpos[:] = ctx.robot.d.qpos
                        mujoco.mj_forward(model, data)
                        images = []
                        for cam in (wide, close):
                            view.update_scene(data, camera=cam, scene_option=opt)
                            images.append(view.render().copy())
                        panels = np.concatenate(images, axis=1)
                        previous = q.copy()
                    im = Image.new('RGB', (WIDTH, HEIGHT), '#111923')
                    im.paste(Image.fromarray(panels), (0, 112))
                    draw = ImageDraw.Draw(im)
                    draw.text((24, 12), f'{fid}  |  {phase}', font=fonts[38], fill=COLOURS[fid])
                    if a.design_candidate:
                        draw.text((800, 23), DESIGN_CANDIDATE_LABEL, font=fonts[23], fill='#f1c16e')
                    draw.text((25, 65), f'固定腰姿 {waist_label}   |   原工装安装 Rz(-90°)   |   离线仿真',
                              font=fonts[26], fill='#e1e7ef')
                    draw.text((22, 123), '共同工位全景', font=fonts[23], fill='white', stroke_width=1, stroke_fill='#111923')
                    close_label = '实测六柱 / 透明扫描包络' if a.design_candidate else '水平电机 / 工装 / 接线盒'
                    draw.text((743, 123), close_label, font=fonts[23], fill='white', stroke_width=1, stroke_fill='#111923')
                    draw.text((24, 762), f'共用固定底座与台面；终点在名义配合目标前 {retract_mm:g} mm，尚非真实插接 / 电测。',
                              font=fonts[26], fill='#edbc79')
                    draw.text((24, 802), f'本型往返 {samples} 点复核通过  |  腰部变化 0°  |  固定台面距地面 {table_height_mm:.1f} mm',
                              font=fonts[23], fill='#b8c5d5')
                    if lift_label:
                        draw.rectangle((0, 711, WIDTH, 751), fill='#111923')
                        draw.text((24, 716), lift_label, font=fonts[26], fill='#8cd8e6')
                    progress = (index+frame_index/max(1, len(frames)-1))/3
                    draw.rectangle((0, HEIGHT-5, WIDTH*progress, HEIGHT), fill=COLOURS[fid])
                    if frame_index in (0, endpoint_index):
                        im.save(a.output/f'{fid}_{"home" if frame_index == 0 else "endpoint"}_preview.png')
                    emit(np.asarray(im))
            finally:
                view.close()
            print(f'Rendered {fid}: {len(wanted)} frames', flush=True)
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('ffmpeg encoding failed')
    manifest = {
        'source_result': str(a.result.resolve()), 'source_result_sha256': sha(a.result),
        'renderer_sha256': sha(__file__), 'sources_sha256_at_render': study.loaded_source_hashes,
        'geometry_sha256_at_build': study.geometry_hashes,
        'motor_visual_manifest_sha256': sha(a.visuals), 'motor_visuals': visual_evidence,
        'tool_visual_manifest_sha256': sha(a.tool_visual), 'tool_visual_registration': tool_visual,
        'fps': a.fps, 'frames': total_frames, 'duration_seconds': total_frames/a.fps,
        'resolution': [WIDTH, HEIGHT], 'waist_model_deg': c['waist_model_deg'],
        'world_T_station': c['world_T_station'], 'station_origin_world_mm': station_mm.tolist(),
        'table_height_above_floor_mm': table_height_mm, 'routes': route_manifest,
        'controller_execution_timing': False, 'physical_or_electrical_acceptance': False,
        'design_candidate': a.design_candidate,
        'station_geometry_confirmed': result['station_geometry_confirmed'],
        'physical_motor_registration_confirmed': result.get('physical_motor_registration_confirmed', False),
        'station_geometry_scope': result.get('station_geometry_scope'),
        'every_frame_candidate_label': DESIGN_CANDIDATE_LABEL if a.design_candidate else None,
        'placement': 'Same foot-hole rectangle centre, shaft direction and level tabletop; one motor at a time.',
        'route': 'Saved validated per-model edges and shared-home connectors, fixed common waist.',
        'endpoint_retract_mm': retract_mm,
        'endpoint': f'{retract_mm:g}mm before the saved nominal mating target, not that distance from measured rim or actual contact.',
        'candidate_limitations': ([
            'Observed local box surfaces and terminals are displayed with archived CAD motor bodies.',
            'Motor-foot-to-box translation and positive output-shaft sign remain nominal design assumptions, not physical registration.',
            'All saved common-home model cycles are checked; physical insertion and electrical tests remain unaccepted.',
        ] if a.design_candidate else []),
    }
    if lift_label:
        manifest.update(every_frame_camera_lift_label=lift_label,
                        camera_installation_metadata=result['frame_metadata'])
    if encoder:
        manifest.update(video_sha256=sha(video), video_bytes=video.stat().st_size)
    (a.output/'video_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'video': str(video) if encoder else None, 'frames': total_frames,
                      'duration_seconds': total_frames/a.fps}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
