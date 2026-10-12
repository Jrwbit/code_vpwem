import tempfile
import ctypes
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import unittest
import numpy as np
import calibrate
import rokae_capture as capture


class CaptureTests(unittest.TestCase):
    def fake_camera(self, streams):
        class Image(ctypes.Structure):
            _fields_ = [('enStreamType', ctypes.c_int), ('enImageType', ctypes.c_int)]
        class Frame(ctypes.Structure):
            _fields_ = [('nImageCount', ctypes.c_uint), ('nValidInfo', ctypes.c_uint),
                        ('stImageData', Image * 3)]
        class Camera:
            calls = 0
            def MV3D_RGBD_FetchFrame(self, ptr, timeout):
                self.calls += 1
                ptr.contents.nImageCount = len(streams)
                for i, stream in enumerate(streams):
                    ptr.contents.stImageData[i].enStreamType = stream
                return 0
        return Camera(), SimpleNamespace(MV3D_RGBD_FRAME_DATA=Frame, StreamType_Color=2,
                                             StreamType_Depth=1, StreamType_Rgbd=10,
                                             ImageType_Depth=100, ImageType_Rgbd=101)

    def test_depth_output_rejected_before_conversion(self):
        camera, definitions = self.fake_camera([2, 1])
        with self.assertRaisesRegex(RuntimeError, 'no depth'):
            capture.fetch_rgb(camera, definitions, lambda *_: self.fail('Converted depth package'))

    def test_rgb_buffer_is_copied(self):
        camera, definitions = self.fake_camera([2])
        sdk_buffer = np.zeros((3, 4, 3), dtype=np.uint8)
        image, timing = capture.fetch_rgb(camera, definitions, lambda *_: sdk_buffer)
        sdk_buffer[:] = 255
        self.assertTrue((image == 0).all())
        self.assertLessEqual(timing['fetch_begin_s'], timing['fetch_end_s'])

    def test_wait_handles_input_without_fetching(self):
        stream = SimpleNamespace(current=lambda: np.zeros((3, 4, 3), dtype=np.uint8))
        stdin = SimpleNamespace(readline=lambda: '\n')
        with patch.object(capture.sys, 'stdin', stdin), patch.object(capture.time, 'sleep'), \
             patch.object(capture.select, 'select', side_effect=[([], [], []), ([stdin], [], [])]):
            self.assertTrue(capture.wait_for_capture(stream))

    def test_preview_save_button(self):
        stream = SimpleNamespace(current=lambda: np.zeros((3, 4, 3), dtype=np.uint8))
        def callback(window, fn):
            fn(capture.cv2.EVENT_LBUTTONUP, 100, 610, 0, None)
        with patch.object(capture.cv2, 'namedWindow'), patch.object(capture.cv2, 'moveWindow'), patch.object(capture.cv2, 'setMouseCallback', side_effect=callback), \
             patch.object(capture.cv2, 'imshow'), patch.object(capture.cv2, 'waitKey', return_value=-1), \
             patch.object(capture.cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(capture.wait_for_capture(stream, preview=True))

    def test_stream_pairs_on_reader_thread_and_propagates_errors(self):
        import threading
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        identities = []
        def fetch(*args):
            identities.append(threading.get_ident())
            capture.time.sleep(.005)
            return image, {}
        packet = (image, {}, {}, {})
        stream = capture.RGBStream(None, None, None, None, None)
        with patch.object(capture, 'fetch_rgb', side_effect=fetch), \
             patch.object(capture, 'acquire', return_value=packet) as acquire:
            stream.start()
            try:
                stream.warmup()
                self.assertIs(stream.capture(), packet)
                acquire.assert_called_once()
                self.assertTrue(all(ident != threading.get_ident() for ident in identities))
            finally:
                stream.stop()
        self.assertFalse(stream.thread.is_alive())
        failure = capture.RGBStream(None, None, None, None, None)
        with patch.object(capture, 'fetch_rgb', side_effect=RuntimeError('camera lost')):
            failure.start()
            try:
                with self.assertRaisesRegex(RuntimeError, 'camera lost'):
                    failure.warmup()
            finally:
                failure.stop()

    def states(self):
        before = dict(timestamp_s=100., read_duration_s=.002, base_T_flange=np.eye(4).tolist(), joint_velocity_rad_s=[0.] * 7)
        after = dict(before, timestamp_s=100.01)
        return before, after

    def test_saves_pair_compatible_with_offline_input(self):
        before, after = self.states()
        config = calibrate.read_json(Path(__file__).with_name('dataset.template.json'))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = capture.save_sample(folder, config, np.zeros((30, 40, 3), dtype=np.uint8), before, after, dict(fetch_begin_s=100.003, fetch_end_s=100.005))
            self.assertEqual(len(data['samples']), 1)
            self.assertTrue((folder / 'images/0001.png').exists())
            self.assertEqual(calibrate.read_json(folder / 'dataset.json'), data)
            self.assertEqual(data['samples'][0]['base_T_tcp'], before['base_T_flange'])

    def test_motion_rejected_without_writes(self):
        before, after = self.states()
        after['joint_velocity_rad_s'] = [.1] * 7
        config = calibrate.read_json(Path(__file__).with_name('dataset.template.json'))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'moving'):
                capture.save_sample(Path(tmp), config, np.zeros((10, 10, 3), dtype=np.uint8), before, after, {})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_slow_capture_rejected(self):
        before, after = self.states()
        after['timestamp_s'] = 102.
        config = calibrate.read_json(Path(__file__).with_name('dataset.template.json'))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'uncertainty'):
                capture.save_sample(Path(tmp), config, np.zeros((10, 10, 3), dtype=np.uint8), before, after, {})


if __name__ == '__main__':
    unittest.main()
