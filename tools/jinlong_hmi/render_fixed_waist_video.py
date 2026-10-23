#!/usr/bin/env python3
"""Render the saved, validated noncontact joint route without solving a new one.

Presentation timing is chosen for readability, not controller execution. Frames
interpolate the same joint-space edges checked in the source result. No camera,
robot, networking, or hardware client is used by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/diagnostics/full-solve-20261023'
WIDTH, HEIGHT, PANEL_HEIGHT = 1440, 848, 640
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
COLOURS = {'JM160': '#edaa65', 'JL132': '#6dd2e2', 'JL100': '#c4a0ed'}


def study_args(output, result):
    return SimpleNamespace(
        bundle=ROOT/'outputs/diagnostics/front-fixed-waist-20261023/input.json',
        scene=ROOT/'outputs/diagnostics/adapter-geometry-20261023/scene_used.json',
        sim_root=Path('/home/wjr/.local/share/jinlong-gpu-planner/ipc/sim-repo'),
        control_root=Path('/home/wjr/.local/share/jinlong-gpu-planner/ipc/control'),
        vision_root=Path('/home/wjr/code/jinlong-motor-arm-control/vision/app'),
        design_config=ROOT/'configs/jinlong/front_fixed_waist_20261023.json',
        geometry_module=ROOT/'tools/jinlong_hmi/build_measured_box_geometry.py',
        tool_geometry_module=ROOT/'tools/jinlong_hmi/build_tool_geometry.py',
        validation_module=ROOT/'tools/jinlong_hmi/cycle_validation.py',
        geometry_fixtures=['JM160', 'JL132', 'JL100'], qr_layout=None,
        contact_offsets=None, target_corrections=None, fixture_revision=None,
        seed=20261023, seed_results=[], seconds=120, output=output,
        bounded_contact_horizon=True, camera_height_mm=550,
        endpoint_retract_mm=result['endpoint_retract_mm'])


def sample_chain(q, positions, fraction):
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(distances)]
    s = float(np.clip(fraction, 0, 1)) * cumulative[-1]
    edge = min(int(np.searchsorted(cumulative, s, side='right'))-1, len(q)-2)
    edge = max(0, edge)
    u = (s-cumulative[edge])/distances[edge] if distances[edge] > 0 else 0.
    return q[edge] + np.clip(u, 0, 1)*(q[edge+1]-q[edge])


def schedule(candidate, fps):
    q = np.asarray(candidate['path']['outward_arm_rad'])
    poses = np.asarray(candidate['path']['outward_flange_poses'])
    xyz = poses[:, :3, 3]
    # The vertical lift ends at the first change in XY. Keep every saved edge.
    away = np.flatnonzero(np.linalg.norm(xyz[:, :2]-xyz[0, :2], axis=1) > 1e-7)
    split = int(away[0]-1) if len(away) else len(q)-1
    if split < 1 or split >= len(q)-1:
        raise ValueError('Expected a vertical approach plus lateral corridor')
    phases = [
        ('等待 / 拍照位置', 1., q[-1:], xyz[-1:]),
        ('水平接近', 3.5, q[split:][::-1], xyz[split:][::-1]),
        ('垂直接近', 1.5, q[:split+1][::-1], xyz[:split+1][::-1]),
        ('非接触终点停留', 2., q[:1], xyz[:1]),
        ('垂直撤离', 1.5, q[:split+1], xyz[:split+1]),
        ('原路返回', 3.5, q[split:], xyz[split:]),
        ('返回同一等待位', 1., q[-1:], xyz[-1:]),
    ]
    frames = []
    for label, seconds, joints, points in phases:
        count = round(seconds*fps)
        for i in range(count):
            value = joints[0] if len(joints) == 1 else sample_chain(joints, points, i/max(1, count-1))
            frames.append((label, value))
    if not np.array_equal(frames[0][1], frames[-1][1]):
        raise ValueError('Presentation route must close at the same home joints')
    return frames


def camera(mujoco, lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
    return c


def card(texts, fonts):
    im = Image.new('RGB', (WIDTH, HEIGHT), '#111923')
    draw = ImageDraw.Draw(im)
    for y, text, size, colour in texts:
        draw.text((65, y), text, font=fonts[size], fill=colour)
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, default=BASE/'root-final-reproduce/result.json')
    p.add_argument('--output', type=Path, default=BASE/'video')
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--preview-only', action='store_true')
    a = p.parse_args()
    if not 1 <= a.fps <= 60:
        p.error('FPS must be in [1, 60]')
    a.output.mkdir(parents=True, exist_ok=True)
    result = json.loads(a.result.read_text())
    if not result.get('full_noncontact_cycle_verified') or result.get('endpoint_retract_mm') != 30:
        raise ValueError('Expected the validated 30mm noncontact cycle')
    for key, source in result.get('sources_sha256_at_load', {}).items():
        finish = result['sources_sha256_at_finish'].get(key)
        if finish is not None and finish != source:
            raise ValueError('Original run changed source during validation: '+key)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from solve_fixed_waist_cycle import Study
    import mujoco
    study = Study(study_args(a.output/'scenes', result))
    c = result['candidate']
    boxes = {f: np.array(t) for f, t in c['world_T_boxes'].items()}
    cam_pose = np.asarray(c['world_T_camera'])
    waist = np.radians(c['waist_model_deg'])
    frames = schedule(c, a.fps)
    fonts = {s: ImageFont.truetype(FONT, s) for s in (23, 26, 29, 38, 48)}
    video_path = a.output/'三型号固定腰非接触往返仿真.mp4'
    encoder = None
    if not a.preview_only:
        encoder = subprocess.Popen([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}',
            '-r', str(a.fps), '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(video_path)], stdin=subprocess.PIPE)

    total_frames = 0
    def emit(frame, repeats=1):
        nonlocal total_frames
        if encoder:
            for _ in range(repeats):
                encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                total_frames += 1

    intro = card([
        (160, '三型号电机 · 固定腰部往返仿真', 48, '#f3f7fb'),
        (262, '160 / 132 / 100   分别定位，一次上一个电机', 38, '#8cd8e6'),
        (350, '同一腰姿、同一手臂路线；实际法兰安装 Rz(-90°)', 29, '#d3dce7'),
        (406, '等待拍照 → 接近 → 非接触停留 → 原路返回', 29, '#d3dce7'),
        (518, '终点在旧名义配合目标前 30 mm；尚未验证真实插接及电测。', 26, '#edbc79'),
        (574, '演示速度便于观看，不代表实机执行周期。', 26, '#a8b5c5'),
    ], fonts)
    emit(intro, a.fps*2)
    try:
        for index, fid in enumerate(('JM160', 'JL132', 'JL100')):
            ctx = study.build(fid, boxes, cam_pose, a.output/'scenes'/fid)
            # Reject changed solid inputs. The same joint interpolation is used,
            # with source result and geometry hashes retained in the manifest.
            for path, value in study.geometry_hashes.items():
                expected = result['geometry_sha256_at_build'].get(path)
                if expected is None and Path(path).name == 'rotated_tool_hulls.json':
                    # configure_geometry writes an identical hull file into the
                    # current output directory; identify its model, not root.
                    suffix = '/' + '/'.join(Path(path).parts[-2:])
                    matches = [v for k, v in result['geometry_sha256_at_build'].items() if k.endswith(suffix)]
                    expected = matches[0] if len(matches) == 1 else None
                if expected != value:
                    raise ValueError('Geometry differs from the validated result: '+path)
            r = ctx.robot
            r.m.vis.global_.offwidth = max(1440, r.m.vis.global_.offwidth)
            r.m.vis.global_.offheight = max(848, r.m.vis.global_.offheight)
            for gid in range(r.m.ngeom):
                if r.m.geom(gid).name.startswith('tool_disc'):
                    r.m.geom_rgba[gid, 3] = .8
                    r.m.geom_group[gid] = 0
            view = mujoco.Renderer(r.m, PANEL_HEIGHT, WIDTH//2)
            opt = mujoco.MjvOption(); opt.sitegroup[:] = 0
            wide = camera(mujoco, [.23, .13, .83], 2.05, 215, -15)
            focus = (boxes[fid][:3, 3] + np.array([.42, .04, .97]))/2
            close = camera(mujoco, focus, .65, 225, -23)
            previous, panels = None, None
            wanted = [0, round(7*a.fps)] if a.preview_only else range(len(frames))
            for frame_index in wanted:
                phase, q = frames[frame_index]
                if previous is None or not np.array_equal(previous, q):
                    r.set(study.cr.WAIST, waist); r.set(study.cr.LEFT, q); r.fk()
                    images = []
                    for cam in (wide, close):
                        view.update_scene(r.d, camera=cam, scene_option=opt)
                        images.append(view.render().copy())
                    panels = np.concatenate(images, axis=1)
                    previous = q.copy()
                image = Image.new('RGB', (WIDTH, HEIGHT), '#111923')
                image.paste(Image.fromarray(panels), (0, 112))
                draw = ImageDraw.Draw(image)
                draw.text((24, 12), f'{fid}  |  {phase}', font=fonts[38], fill=COLOURS[fid])
                draw.text((25, 65), '腰姿固定 [-15°, 5°, 5°, -60°]   |   实际安装 Rz(-90°)   |   离线仿真',
                          font=fonts[26], fill='#e1e7ef')
                draw.text((22, 123), '全局视角', font=fonts[23], fill='#ffffff', stroke_width=1, stroke_fill='#111923')
                draw.text((743, 123), '工装 / 接线盒近景', font=fonts[23], fill='#ffffff', stroke_width=1, stroke_fill='#111923')
                draw.text((24, 762), '非接触终点：旧名义配合目标前 30 mm；未演示真实插接 / 电测。',
                          font=fonts[26], fill='#edbc79')
                draw.text((24, 802), '每型完整路线 752 点复核通过  |  腰部变化 0°  |  演示速度，非实机周期',
                          font=fonts[23], fill='#b8c5d5')
                progress = (index + frame_index/max(1, len(frames)-1))/3
                draw.rectangle((0, HEIGHT-5, WIDTH*progress, HEIGHT), fill=COLOURS[fid])
                frame = np.asarray(image)
                if frame_index in (0, round(7*a.fps)):
                    suffix = 'home' if frame_index == 0 else 'endpoint'
                    image.save(a.output/f'{fid}_{suffix}_preview.png')
                emit(frame)
            view.close()
            print(f'Rendered {fid}: {len(wanted)} frames', flush=True)
    finally:
        if encoder:
            encoder.stdin.close()
            returncode = encoder.wait()
            if returncode:
                raise RuntimeError(f'ffmpeg failed with exit status {returncode}')
    manifest = {
        'source_result': str(a.result.resolve()),
        'source_result_sha256': hashlib.sha256(a.result.read_bytes()).hexdigest(),
        'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'geometry_sha256_at_build': study.geometry_hashes,
        'sources_sha256_at_render': study.loaded_source_hashes,
        'fps': a.fps, 'frames': total_frames, 'duration_seconds': total_frames/a.fps,
        'resolution': [WIDTH, HEIGHT], 'waist_model_deg': c['waist_model_deg'],
        'controller_execution_timing': False, 'physical_or_electrical_acceptance': False,
        'route': 'Saved validated joint edges, interpolated and reversed for approach; same fixed waist.',
        'endpoint': '30mm before saved nominal target, not 30mm from measured box rim/contact.',
    }
    if encoder:
        manifest['video_sha256'] = hashlib.sha256(video_path.read_bytes()).hexdigest()
        manifest['video_bytes'] = video_path.stat().st_size
    (a.output/'video_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'video': str(video_path) if encoder else None, 'frames': total_frames,
                      'duration_seconds': total_frames/a.fps}), flush=True)


if __name__ == '__main__':
    main()
