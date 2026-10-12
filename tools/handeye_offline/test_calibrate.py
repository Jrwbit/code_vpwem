import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import calibrate as c


def transform(rv, t):
    return c.rt(cv2.Rodrigues(np.asarray(rv, dtype=float))[0], t)


def observations(mode):
    rng = np.random.default_rng(8)
    x = transform([.1, -.2, .3], [.03, .01, .1])
    z = transform([-.3, .2, .1], [.4, .1, .6])
    samples = []
    for i in range(15):
        a = transform(rng.normal(0, .5, 3), rng.uniform(-.3, .3, 3))
        aa = a if mode == 'eye-to-hand' else np.linalg.inv(a)
        camera = np.linalg.inv(z) @ aa @ x
        samples.append({'id': str(i), 'base_T_tcp': a.tolist(), 'camera_T_board': camera.tolist(),
                        'image_timestamp_s': i + 100., 'pose_timestamp_s': i + 100.01,
                        'robot_stationary': True, 'reprojection_rms_px': .1})
    return x, z, samples


class OfflineTests(unittest.TestCase):
    def config(self, mode):
        config = json.loads((Path(__file__).parent / 'dataset.template.json').read_text())
        config['mode'] = mode
        config['intrinsics'].update(width=1920, height=1080, camera_matrix=[[1200,0,960],[0,1200,540],[0,0,1]])
        config['samples'] = observations(mode)[2]
        return config

    def test_both_mount_modes_recover_known_transforms(self):
        for mode in ['eye-to-hand', 'eye-in-hand']:
            x, z, samples = observations(mode)
            for method in [cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH, cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI]:
                fit = c.solve(samples, mode, method)
                names = ('tcp_T_board', 'base_T_camera') if mode == 'eye-to-hand' else ('base_T_board', 'tcp_T_camera')
                np.testing.assert_allclose(fit[names[0]], x, atol=1e-9)
                np.testing.assert_allclose(fit[names[1]], z, atol=1e-9)

    def test_full_report_and_holdout(self):
        for mode in ['eye-to-hand', 'eye-in-hand']:
            with tempfile.TemporaryDirectory() as tmp:
                source, report = Path(tmp) / 'input.json', Path(tmp) / 'report.json'
                c.write_json(source, self.config(mode))
                self.assertEqual(c.run(source, report), 0)
                result = c.read_json(report)
                self.assertTrue(result['ready'])
                self.assertLess(result['leave_one_out']['translation_mm']['max'], 1e-6)

    def test_sync_motion_and_bad_rotation_rejected(self):
        config = self.config('eye-to-hand')
        config['samples'][0]['image_timestamp_s'] += config['limits']['max_time_delta_s'] + 0.1
        config['samples'][1]['robot_stationary'] = False
        config['samples'][2]['base_T_tcp'][0][0] = 5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.json'
            c.write_json(path, config)
            _, accepted, rejected = c.load_samples(path)
            self.assertEqual(len(accepted), 12)
            self.assertEqual(len(rejected), 3)

    def test_degenerate_motion_rejected(self):
        _, _, samples = observations('eye-to-hand')
        for sample in samples:
            sample['base_T_tcp'] = np.eye(4).tolist()
        with self.assertRaisesRegex(ValueError, 'rotation diversity'):
            c.solve(samples, 'eye-to-hand', cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)

    def test_placeholder_intrinsics_rejected(self):
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            c.load_samples(Path(__file__).parent / 'dataset.template.json')

    def test_duplicate_ids_rejected(self):
        config = self.config('eye-to-hand')
        config['samples'][1]['id'] = config['samples'][0]['id']
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.json'
            c.write_json(source, config)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                c.load_samples(source)

    def test_saved_image_corner_detection(self):
        config = self.config('eye-to-hand')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            b = config['board']
            board = cv2.aruco.CharucoBoard((b['squares_x'], b['squares_y']), b['square_size_m'], b['marker_size_m'], cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
            image = board.generateImage((960, 720), marginSize=50)
            cv2.imwrite(str(folder / 'board.png'), image)
            config['intrinsics'].update(width=960, height=720, camera_matrix=[[1000,0,480],[0,1000,360],[0,0,1]])
            sample = config['samples'][0]
            sample.pop('camera_T_board')
            sample.pop('reprojection_rms_px')
            sample['image'] = 'board.png'
            config['samples'] = [sample]
            c.write_json(folder / 'data.json', config)
            _, accepted, rejected = c.load_samples(folder / 'data.json')
            self.assertEqual(rejected, [])
            self.assertEqual(len(accepted), 1)


if __name__ == '__main__':
    unittest.main()
