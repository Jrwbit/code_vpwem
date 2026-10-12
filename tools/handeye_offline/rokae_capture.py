#!/usr/bin/env python3
"""Capture Hikrobot images paired with ROKAE feedback; never issues motion commands."""
import argparse
from ctypes import pointer
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

import calibrate


def checked(robot, method, *args):
    ec = {}
    value = getattr(robot, method)(*args, ec)
    if ec.get('ec') != 0:
        raise RuntimeError(f'{method}: {ec}')
    return value


def snapshot(robot, sdk):
    started = time.time()
    posture = checked(robot, 'posture', sdk.CoordinateType.flangeInBase)
    joints = checked(robot, 'jointPos')
    velocity = checked(robot, 'jointVel')
    ended = time.time()
    transform = calibrate.matrix(np.asarray(sdk.utility.postureToTransArray(posture)).reshape(4, 4))
    return {'timestamp_s': (started + ended) / 2, 'read_duration_s': ended - started,
            'base_T_flange': transform.tolist(), 'joints_rad': list(joints),
            'joint_velocity_rad_s': list(velocity), 'posture_m_rad': list(posture)}


def stationary(before, after):
    values = [np.asarray(s['joint_velocity_rad_s']) for s in (before, after)]
    difference = calibrate.delta(calibrate.matrix(before['base_T_flange']), calibrate.matrix(after['base_T_flange']))
    return all(np.isfinite(v).all() and np.max(np.abs(v)) < .002 for v in values) and difference['translation_mm'] < .2 and difference['rotation_deg'] < .1


def acquire(camera, definitions, robot, sdk, convert):
    # Clear old queued frames before pairing robot feedback around a fresh fetch.
    if hasattr(camera, 'MV3D_RGBD_ClearDataBuffer'):
        result = camera.MV3D_RGBD_ClearDataBuffer()
        if result != 0:
            raise RuntimeError(f'ClearDataBuffer: {result}')
    before = snapshot(robot, sdk)
    frame = definitions.MV3D_RGBD_FRAME_DATA()
    begin = time.time()
    result = camera.MV3D_RGBD_FetchFrame(pointer(frame), 3000)
    end = time.time()
    if result != 0 or frame.nValidInfo != 0:
        raise RuntimeError(f'FetchFrame failed: {result}, validInfo={frame.nValidInfo}')
    image = None
    for i in range(int(frame.nImageCount)):
        converted = convert(frame.stImageData[i], definitions)
        if converted is not None:
            image = converted.copy()
    if image is None:
        raise RuntimeError('No RGB image in package')
    after = snapshot(robot, sdk)
    return image, before, after, {'fetch_begin_s': begin, 'fetch_end_s': end}


def save_sample(folder, data, image, before, after, timing):
    if not stationary(before, after):
        raise ValueError('Robot moving or feedback changed during image capture')
    # Host intervals bracket acquisition, not the sensor exposure timestamp.
    interval = after['timestamp_s'] - before['timestamp_s'] + .5 * (before['read_duration_s'] + after['read_duration_s'])
    if interval < 0 or interval > data['limits']['max_time_delta_s']:
        raise ValueError(f'Capture uncertainty {interval:.3f}s exceeds configured limit')
    ident = f"{len(data['samples']) + 1:04d}"
    image_path = folder / 'images' / f'{ident}.png'
    if image_path.exists():
        raise ValueError('Image already exists; inspect dataset before resuming')
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), image):
        raise OSError('Cannot save image')
    sample = {'id': ident, 'image': f'images/{ident}.png',
              'image_timestamp_s': (timing['fetch_begin_s'] + timing['fetch_end_s']) / 2,
              'pose_timestamp_s': before['timestamp_s'], 'robot_stationary': True,
              'base_T_tcp': before['base_T_flange'],
              'timestamp_source': 'host bracket of acquisition; not hardware exposure timestamp',
              'capture_uncertainty_s': interval, 'robot_before': before, 'robot_after': after, **timing}
    candidate = dict(data, samples=data['samples'] + [sample])
    try:
        calibrate.write_json(folder / 'dataset.json', candidate)
    except Exception:
        image_path.unlink()
        raise
    return candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-ip', default='192.168.71.161')
    p.add_argument('--robot-class', choices=['xMateErProRobot', 'xMateRobot', 'StandardRobot'], default='xMateErProRobot')
    p.add_argument('--camera-serial', default='00DA8664418')
    p.add_argument('--camera-local-ip', default='192.168.1.67')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('dataset.template.json'))
    p.add_argument('--once', action='store_true', help='Save a single diagnostic frame and feedback; no robot motion')
    args = p.parse_args()
    import xCoreSDK_python as sdk
    from hk_pointcloud_registration.capture_haikang import _load_hk_sdk, _list_devices, _color_to_bgr
    api, definitions = _load_hk_sdk()
    devices, count = _list_devices(api, definitions)
    selected = None
    for i in range(count):
        serial = bytes(devices.DeviceInfo[i].chSerialNumber).split(b'\0')[0].decode()
        local_ip = bytes(devices.DeviceInfo[i].SpecialInfo.stNetInfo.chNetExport).split(b'\0')[0].decode()
        if serial == args.camera_serial and local_ip == args.camera_local_ip:
            selected = i
            break
    if selected is None:
        raise RuntimeError('Specified camera/local interface not found')
    robot = getattr(sdk, args.robot_class)(args.robot_ip)
    camera = api()
    opened = started = False
    try:
        info = checked(robot, 'robotInfo')
        print('Robot', info.type, 'controller', info.version, 'joints', info.joint_num, flush=True)
        result = camera.MV3D_RGBD_OpenDevice(pointer(devices.DeviceInfo[selected]))
        if result != 0:
            raise RuntimeError(f'OpenDevice 0x{result:08x}; do not stop another owner automatically')
        opened = True
        params = definitions.MV3D_RGBD_CAMERA_PARAM()
        if camera.MV3D_RGBD_GetCameraParam(pointer(params)) != 0:
            raise RuntimeError('Cannot obtain camera intrinsics')
        rgb = params.stRgbCalibInfo
        intrinsics = {'width': int(rgb.nWidth), 'height': int(rgb.nHeight),
                      'camera_matrix': np.asarray(list(rgb.stIntrinsic.fData)).reshape(3, 3).tolist(),
                      'distortion_coefficients': list(rgb.stDistortion.fData)}
        if camera.MV3D_RGBD_Start() != 0:
            raise RuntimeError('Cannot start camera acquisition')
        started = True
        folder = args.session.resolve()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'dataset.json'
        config = calibrate.read_json(args.config)
        identity = {'robot_ip': args.robot_ip, 'robot_type': info.type, 'robot_id': info.id,
                    'camera_serial': args.camera_serial, 'pose_reference': 'flangeInBase'}
        if path.exists():
            data = calibrate.read_json(path)
            if data.get('devices') != identity or data['intrinsics'] != intrinsics or data['board'] != config['board'] or data['mode'] != config['mode']:
                raise ValueError('Dataset configuration differs; use a new session')
        else:
            data = dict(config, intrinsics=intrinsics, samples=[], devices=identity,
                        robot_reference='flange; base_T_tcp is base_T_flange, tcp_T_board is flange_T_board')
            calibrate.write_json(path, data)
        for _ in range(5):
            frame = definitions.MV3D_RGBD_FRAME_DATA()
            camera.MV3D_RGBD_FetchFrame(pointer(frame), 3000)
        while True:
            if not args.once and input('Enter: capture stationary pair; q: finish > ').strip().lower() == 'q':
                break
            image, before, after, timing = acquire(camera, definitions, robot, sdk, _color_to_bgr)
            if args.once:
                if not cv2.imwrite(str(folder / 'diagnostic.png'), image):
                    raise OSError('Cannot save diagnostic image')
                calibrate.write_json(folder / 'diagnostic.json', {'before': before, 'after': after, **timing})
            try:
                candidate = save_sample(folder, data, image, before, after, timing)
                _, accepted, rejected = calibrate.load_samples(path)
                data = candidate
                print('Saved', len(data['samples']), 'valid', len(accepted), 'rejected', rejected, flush=True)
            except ValueError as exc:
                print('Capture rejected:', exc, flush=True)
            if args.once:
                break
    finally:
        if started:
            camera.MV3D_RGBD_Stop()
        if opened:
            camera.MV3D_RGBD_CloseDevice()
        robot.disconnectFromRobot({})


if __name__ == '__main__':
    main()
