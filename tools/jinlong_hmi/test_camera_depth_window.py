"""Depth-window regressions use real projection of box, bottom and QR outlines."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from check_layout_camera import DISTORTION_KEYS
from screen_cycle_camera import SIZES, evaluate, validate_depth_window
from screen_shared_station_camera import evaluate_measured_depths
from screen_camera_vertical_lift import main as lift_main, projection_summary


class CameraDepthWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.intr = {'width': 1920, 'height': 1080, 'fx': 500., 'fy': 500.,
                     'cx': 960., 'cy': 540., 'dist_coeffs': dict.fromkeys(DISTORTION_KEYS, 0.)}
        self.geometry = {}
        for fid in SIZES:
            path = self.folder / (fid + '_collision.json')
            path.write_text(json.dumps({'bounds_box_mm': [[-80, -80, 0], [80, 80, 40]]}))
            self.geometry[fid] = path

    def candidate(self, depth_mm):
        camera = np.diag([1., -1., -1., 1.])
        camera[2, 3] = 1.
        camera_box = np.eye(4)
        camera_box[2, 3] = depth_mm / 1000
        return {'world_T_camera': camera.tolist(),
                'world_T_boxes': {fid: (camera @ camera_box).tolist() for fid in SIZES},
                'world_T_station': np.eye(4).tolist(), 'waist_model_deg': [0, 0, 0, 0],
                'frame_metadata': {'lab_T_robot': np.eye(4).tolist()}}

    def planes(self, result):
        for row in result['models'].values():
            yield row['rim']
            yield row.get('conservative_geometry_depth_envelope',
                          row.get('conservative_90mm_depth_envelope'))
            yield row['proposed_QR_patch']['screen']

    def test_400_lower_bound_rejects_near_rim_bottom_and_qr(self):
        candidate = self.candidate(350)
        old = evaluate_measured_depths(candidate, self.intr, self.geometry)
        new = evaluate_measured_depths(candidate, self.intr, self.geometry, [400, 800])
        self.assertTrue(old['all_projection_gates_pass'])
        self.assertEqual(old['software_depth_window_mm'], [300, 700])
        self.assertFalse(new['all_projection_gates_pass'])
        self.assertEqual(new['software_depth_window_mm'], [400, 800])
        for plane in self.planes(new):
            self.assertTrue(plane['all_samples_inside_image'])
            self.assertFalse(plane['all_samples_inside_software_depth_window'])
        self.assertEqual(evaluate(candidate, self.intr), evaluate(candidate, self.intr, [300, 700]))

    def test_800_upper_bound_accepts_750_rim_and_790_bottom(self):
        candidate = self.candidate(750)
        old = evaluate_measured_depths(candidate, self.intr, self.geometry)
        new = evaluate_measured_depths(candidate, self.intr, self.geometry, [400, 800])
        for plane in self.planes(old):
            self.assertFalse(plane['all_samples_inside_software_depth_window'])
        for plane in self.planes(new):
            self.assertTrue(plane['projection_sample_gates_pass'])
        self.assertTrue(new['all_projection_gates_pass'])
        for row in projection_summary(new).values():
            self.assertAlmostEqual(row['remaining_upper_depth_window_mm'], 10)
            self.assertAlmostEqual(row['remaining_lower_depth_window_mm'], 350)

    def test_conservative_90mm_bottom_uses_same_selected_window(self):
        candidate = self.candidate(660)
        old = evaluate(candidate, self.intr)
        new = evaluate(candidate, self.intr, [400, 800])
        self.assertFalse(old['all_projection_gates_pass'])
        self.assertTrue(new['all_projection_gates_pass'])
        for row in old['models'].values():
            self.assertTrue(row['rim']['projection_sample_gates_pass'])
            self.assertTrue(row['proposed_QR_patch']['screen']['projection_sample_gates_pass'])
            self.assertFalse(row['conservative_90mm_depth_envelope']['projection_sample_gates_pass'])

    def test_invalid_windows_are_rejected_at_both_projection_entrypoints(self):
        for window in ([400], [400, 800, 900], [0, 800], [-1, 800], [800, 400],
                       [400, 400], [float('nan'), 800], [400, float('inf')]):
            with self.subTest(window=window):
                with self.assertRaises(ValueError):
                    validate_depth_window(window)
                with self.assertRaises(ValueError):
                    evaluate(self.candidate(500), self.intr, window)
                with self.assertRaises(ValueError):
                    evaluate_measured_depths(self.candidate(500), self.intr, self.geometry, window)

    def test_lift_cli_uses_800_for_saved_roi_results_and_theoretical_limit(self):
        source = self.folder / 'result.json'
        source.write_text(json.dumps({'candidate': self.candidate(750),
            'geometry_sha256_at_build': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in self.geometry.values()}}))
        intrinsics = self.folder / 'intrinsics.json'
        intrinsics.write_text(json.dumps(self.intr))
        output = self.folder / 'screen'
        argv = ['screen_camera_vertical_lift.py', '--result', str(source),
                '--intrinsics', str(intrinsics), '--output', str(output),
                '--lifts-mm', '0', '20', '--depth-window-mm', '400', '800']
        with patch('sys.argv', argv), patch('screen_camera_vertical_lift.plot'), contextlib.redirect_stdout(io.StringIO()):
            lift_main()
        result = json.loads((output / 'camera_vertical_lift_optics.json').read_text())
        self.assertEqual(result['software_depth_window_mm'], [400, 800])
        self.assertAlmostEqual(result['depth_lift_limit']['maximum_theoretical_lift_mm'], 10)
        self.assertEqual([step['all_projection_gates_pass'] for step in result['evaluated_lifts']], [True, False])
        checks = result['depth_lift_limit']['numerical_boundary_checks']
        self.assertLess(checks[0]['maximum_depth_mm'], 800)
        self.assertGreater(checks[1]['maximum_depth_mm'], 800)
        for step in result['evaluated_lifts']:
            saved = json.loads((output / step['full_projection_result']).read_text())
            self.assertEqual(saved['software_depth_window_mm'], [400, 800])


if __name__ == '__main__':
    unittest.main()
