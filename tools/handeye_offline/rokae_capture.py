#!/usr/bin/env python3
"""Capture Hikrobot images paired with ROKAE feedback; never issues motion commands."""
import argparse
from ctypes import pointer
from pathlib import Path
import sys
import select
import time
import threading
import queue

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


def configure_rgb(camera, definitions):
    """Apply and read back RGB-only, free-running acquisition before Start."""
    settings = [('CameraWorkingMode', definitions.ParamType_Enum, 0),
                ('ImageMode', definitions.ParamType_Enum, 8),
                ('OutputRgbd', definitions.ParamType_Int, 0),
                ('TriggerMode', definitions.ParamType_Enum, 0)]
    for key, kind, value in settings:
        param = definitions.MV3D_RGBD_PARAM()
        param.enParamType = kind
        if kind == definitions.ParamType_Enum:
            param.ParamInfo.stEnumParam.nCurValue = value
        else:
            param.ParamInfo.stIntParam.nCurValue = value
        result = camera.MV3D_RGBD_SetParam(key, pointer(param))
        if result != 0:
            raise RuntimeError(f'SetParam({key}) 0x{result:08x}')
        result = camera.MV3D_RGBD_GetParam(key, pointer(param))
        actual = (param.ParamInfo.stEnumParam.nCurValue if kind == definitions.ParamType_Enum
                  else param.ParamInfo.stIntParam.nCurValue)
        if result != 0 or actual != value:
            raise RuntimeError(f'Cannot verify {key}={value}: result={result}, actual={actual}')
        print(f'{key}={actual}', flush=True)


def fetch_rgb(camera, definitions, convert):
    frame = definitions.MV3D_RGBD_FRAME_DATA()
    begin = time.time()
    result = camera.MV3D_RGBD_FetchFrame(pointer(frame), 3000)
    end = time.time()
    if result != 0 or frame.nValidInfo != 0:
        raise RuntimeError(f'FetchFrame failed: {result}, validInfo={frame.nValidInfo}')
    # Mode 0 may include raw mono auxiliary data; reject every depth/RGBD stream.
    streams = [frame.stImageData[i] for i in range(frame.nImageCount)]
    if any(int(im.enStreamType) in (definitions.StreamType_Depth, definitions.StreamType_Rgbd)
           or int(im.enImageType) in (definitions.ImageType_Depth, definitions.ImageType_Rgbd)
           for im in streams):
        raise RuntimeError('Expected no depth output; camera returned depth/RGBD')
    colors = [im for im in streams if im.enStreamType == definitions.StreamType_Color]
    if len(colors) != 1:
        raise RuntimeError('Expected one RGB image in package')
    image = convert(colors[0], definitions)
    if image is None:
        raise RuntimeError('No RGB image in package')
    return image.copy(), {'fetch_begin_s': begin, 'fetch_end_s': end}


class RGBStream:
    """One SDK owner consumes frames even while the UI or sample writer is busy."""
    def __init__(self, camera, definitions, robot, sdk, convert):
        self.camera, self.definitions = camera, definitions
        self.robot, self.sdk, self.convert = robot, sdk, convert
        self.done = threading.Event()
        self.condition = threading.Condition()
        self.requests = queue.Queue(maxsize=1)
        self.responses = queue.Queue(maxsize=1)
        self.latest = None
        self.sequence = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, name='handeye-rgb', daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            while not self.done.is_set():
                try:
                    self.requests.get_nowait()
                    requested = True
                except queue.Empty:
                    requested = False
                if requested:
                    packet = acquire(self.camera, self.definitions, self.robot, self.sdk, self.convert)
                    image = packet[0]
                    self.responses.put(packet)
                else:
                    image, _ = fetch_rgb(self.camera, self.definitions, self.convert)
                with self.condition:
                    self.latest = image
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def current(self):
        with self.condition:
            if self.error:
                raise self.error
            return self.latest

    def warmup(self):
        with self.condition:
            if not self.condition.wait_for(lambda: self.sequence >= 5 or self.error, timeout=20):
                raise RuntimeError('RGB warmup timed out')
            if self.error:
                raise self.error

    def capture(self):
        self.current()  # Surface a failed reader before submitting a request.
        self.requests.put_nowait(True)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                return self.responses.get(timeout=.05)
            except queue.Empty:
                self.current()
        raise RuntimeError('Paired capture timed out')

    def stop(self):
        self.done.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('RGB reader did not stop; do not close its active SDK handle')


def preview_canvas(image, status):
    canvas = np.zeros((700, 960, 3), dtype=np.uint8)
    height = min(583, round(image.shape[0] * 960 / image.shape[1]))
    width = round(image.shape[1] * height / image.shape[0])
    canvas[:height, :width] = cv2.resize(image, (width, height))
    cv2.rectangle(canvas, (10, 590), (300, 635), (40, 130, 40), -1)
    cv2.rectangle(canvas, (740, 590), (950, 635), (65, 65, 150), -1)
    cv2.putText(canvas, 'SAVE SAMPLE (S / Enter)', (20, 620), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    cv2.putText(canvas, 'EXIT (Q)', (775, 620), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
    for i in range(2):
        cv2.putText(canvas, status[i * 105:(i + 1) * 105], (10, 660 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    return canvas


def wait_for_capture(stream, preview=False, status='Ready. Stop the robot, then click SAVE SAMPLE.'):
    """Poll input independently of the blocking SDK FetchFrame call."""
    print('RGB live: Enter/s capture stationary pair; q finish > ', flush=True)
    action = {'value': None}
    window = 'Hand-eye RGB (s: capture, q: quit)'
    if preview:
        cv2.namedWindow(window)
        cv2.moveWindow(window, 80, 60)
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP and 590 <= y <= 635:
                if 10 <= x <= 300:
                    action['value'] = True
                elif 740 <= x <= 950:
                    action['value'] = False
        cv2.setMouseCallback(window, on_mouse)
    while True:
        image = stream.current()
        if preview and image is not None:
            cv2.imshow(window, preview_canvas(image, status))
            key = cv2.waitKey(20) & 0xff
            if key in (ord('s'), ord('S'), 13, 10, 32):
                action['value'] = True
            if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                action['value'] = False
        else:
            time.sleep(.02)
        if action['value'] is not None:
            return action['value']
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line or line.strip().lower() == 'q':
                return False
            if line.strip().lower() in ('', 's'):
                return True


def acquire(camera, definitions, robot, sdk, convert):
    # Clear old queued frames before pairing robot feedback around a fresh fetch.
    if hasattr(camera, 'MV3D_RGBD_ClearDataBuffer'):
        result = camera.MV3D_RGBD_ClearDataBuffer()
        if result != 0:
            raise RuntimeError(f'ClearDataBuffer: {result}')
    before = snapshot(robot, sdk)
    image, timing = fetch_rgb(camera, definitions, convert)
    after = snapshot(robot, sdk)
    return image, before, after, timing


def save_sample(folder, data, image, before, after, timing):
    if not stationary(before, after):
        raise ValueError('Robot moving or feedback changed: '
                         f"max velocity={max(np.max(np.abs(v['joint_velocity_rad_s'])) for v in (before, after)):.6f} rad/s; "
                         'wait until stationary and retry')
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
    p.add_argument('--camera-local-ip', default='192.168.1.68')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('dataset.template.json'))
    p.add_argument('--once', action='store_true', help='Save a single diagnostic frame and feedback; no robot motion')
    p.add_argument('--preview', action='store_true', help='Show live RGB on the IPC desktop; s saves, q exits')
    p.add_argument('--stream-check-seconds', type=float, default=0, help='Check continuous RGB-only frames for this duration, then exit without samples')
    args = p.parse_args()
    if args.stream_check_seconds < 0 or not np.isfinite(args.stream_check_seconds):
        p.error('--stream-check-seconds must be finite and nonnegative')
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
    stream = None
    try:
        info = checked(robot, 'robotInfo')
        print('Robot', info.type, 'controller', info.version, 'joints', info.joint_num, flush=True)
        result = camera.MV3D_RGBD_OpenDevice(pointer(devices.DeviceInfo[selected]))
        if result != 0:
            raise RuntimeError(f'OpenDevice 0x{result:08x}; do not stop another owner automatically')
        opened = True
        configure_rgb(camera, definitions)
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
        if args.stream_check_seconds:
            begin = time.monotonic()
            frames = 0
            while time.monotonic() - begin < args.stream_check_seconds:
                image, timing = fetch_rgb(camera, definitions, _color_to_bgr)
                frames += 1
            print(f'RGB-only stream check: {frames} frames / {time.monotonic() - begin:.2f}s, '
                  f'{image.shape[1]}x{image.shape[0]}, no depth output', flush=True)
            return
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
        stream = RGBStream(camera, definitions, robot, sdk, _color_to_bgr)
        stream.start()
        stream.warmup()
        status = f"Ready. Saved {len(data['samples'])} samples. Stop the robot before saving."
        while True:
            if not args.once and not wait_for_capture(stream, args.preview, status):
                break
            if args.preview:
                image = stream.current()
                cv2.imshow('Hand-eye RGB (s: capture, q: quit)', preview_canvas(image, 'Capturing... keep the robot stationary.'))
                cv2.waitKey(1)
            image, before, after, timing = stream.capture()
            if args.once:
                if not cv2.imwrite(str(folder / 'diagnostic.png'), image):
                    raise OSError('Cannot save diagnostic image')
                calibrate.write_json(folder / 'diagnostic.json', {'before': before, 'after': after, **timing})
            try:
                candidate = save_sample(folder, data, image, before, after, timing)
                data = candidate
                _, accepted, rejected = calibrate.load_samples(path)
                status = (f"SAVED #{data['samples'][-1]['id']} | total {len(data['samples'])} | valid {len(accepted)}"
                          + (f' | board rejected: {rejected[-1]}' if rejected else ' | board check OK'))
                print(status, flush=True)
            except ValueError as exc:
                status = f'NOT SAVED: {exc}'
                print(status, flush=True)
                if args.once:
                    return 2
            if args.once:
                break
    finally:
        try:
            if stream is not None:
                stream.stop()
            if started:
                result = camera.MV3D_RGBD_Stop()
                print(f'Stop 0x{result:08x}', flush=True)
                if result != 0:
                    raise RuntimeError(f'Cannot stop camera: 0x{result:08x}')
        finally:
            try:
                if opened and (stream is None or not stream.thread.is_alive()):
                    result = camera.MV3D_RGBD_CloseDevice()
                    print(f'CloseDevice 0x{result:08x}', flush=True)
                    if result != 0:
                        raise RuntimeError(f'Cannot close camera: 0x{result:08x}')
            finally:
                if stream is None or not stream.thread.is_alive():
                    robot.disconnectFromRobot({})
                if args.preview:
                    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main())
