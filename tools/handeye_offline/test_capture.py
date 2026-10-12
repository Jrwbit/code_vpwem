import tempfile
from pathlib import Path
import unittest
import numpy as np
import calibrate
import rokae_capture as capture


class CaptureTests(unittest.TestCase):
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
        after['timestamp_s'] = 101.
        config = calibrate.read_json(Path(__file__).with_name('dataset.template.json'))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'uncertainty'):
                capture.save_sample(Path(tmp), config, np.zeros((10, 10, 3), dtype=np.uint8), before, after, {})


if __name__ == '__main__':
    unittest.main()
