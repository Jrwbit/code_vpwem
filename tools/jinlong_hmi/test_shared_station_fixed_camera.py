"""Fixed external camera and visible waiting-pose regressions, without hardware."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from solve_shared_motor_station import FIXTURES, SharedStationStudy


class FixedCameraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.calibration_path = self.folder / 'camera.json'
        self.calibration = {'base_frame': 'rokae_left_sdk_base', 'translation_unit': 'm',
                            'waist_model_rad': [.1, .2, .3, .4],
                            'base_T_camera': np.eye(4).tolist()}
        self.calibration['base_T_camera'][1][3] = -.42
        self.calibration_path.write_text(json.dumps(self.calibration))
        self.fk_calls = []
        module = ModuleType('helios_hmi.urdf_kin')
        module.__file__ = str(self.folder / 'urdf_kin.py')
        Path(module.__file__).write_text('# fake kinematics source\n')
        urdf = self.folder / 'robot.urdf'
        urdf.write_text('<robot/>\n')
        module.default_urdf_path = lambda: urdf
        module.WAIST_JOINTS = ('J1', 'J2', 'J3', 'J4')
        calls = self.fk_calls

        class Kinematics:
            def __init__(self, path):
                assert path == urdf

            def link_pose(self, link, joints):
                calls.append((link, joints.copy()))
                T = np.eye(4)
                T[:3, 3] = [joints['J1'], joints['J2'], 1 + joints['J4']]
                return T

        module.UrdfKinematics = Kinematics
        package = ModuleType('helios_hmi')
        package.urdf_kin = module
        modules = patch.dict('sys.modules', {'helios_hmi': package,
                                            'helios_hmi.urdf_kin': module})
        modules.start()
        self.addCleanup(modules.stop)
        self.study = SharedStationStudy.__new__(SharedStationStudy)
        self.study.station_design = {}
        self.study.loaded_source_hashes = {}
        self.study.fixed_world_T_camera = None

    def test_reference_waist_freezes_camera_across_station_and_working_waist(self):
        self.study.load_fixed_camera_calibration(self.calibration_path)
        expected = np.eye(4)
        expected[:3, 3] = [.1, -.22, 1.4]
        for xyz, waist in [([.37, -.04, .5], [-25, 20, 10, -100]),
                           ([.5, .2, .4], [0, 0, 0, -60])]:
            self.study.station = np.eye(4)
            self.study.station[:3, 3] = xyz
            self.study.args = SimpleNamespace(waist_deg=[waist], camera_height_mm=900,
                                               camera_offset_xy_mm=[250, 250])
            boxes = {f: self.study.station.copy() for f in FIXTURES}
            camera = self.study.camera_pose(boxes)
            np.testing.assert_allclose(camera, expected)
            camera[0, 3] += 9  # Callers cannot change the stored fixed pose.
        self.assertEqual(len(self.fk_calls), 1)
        self.assertEqual(self.fk_calls[0], ('AR5-5_07L_base',
                                          {'J1': .1, 'J2': .2, 'J3': .3, 'J4': .4}))
        self.assertEqual(len(self.study.loaded_source_hashes), 3)
        for filename, digest in self.study.loaded_source_hashes.items():
            self.assertEqual(digest, hashlib.sha256(Path(filename).read_bytes()).hexdigest())

    def test_conflicting_design_camera_is_rejected(self):
        self.study.station_design = {'world_T_camera': np.eye(4).tolist()}
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.study.load_fixed_camera_calibration(self.calibration_path)

    def test_matching_design_camera_is_accepted(self):
        self.study.load_fixed_camera_calibration(self.calibration_path)
        self.study.station_design = {'world_T_camera': self.study.camera_pose({}).tolist()}
        self.study.load_fixed_camera_calibration(self.calibration_path)

    def test_wrong_frame_or_units_and_invalid_reference_angles_are_rejected(self):
        for key, value in [('base_frame', 'rokae_right_sdk_base'),
                           ('translation_unit', 'mm'), ('waist_model_rad', [0, 0, 0]),
                           ('waist_model_rad', [0, 0, 0, float('nan')])]:
            with self.subTest(key=key, value=value):
                changed = {**self.calibration, key: value}
                self.calibration_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    self.study.load_fixed_camera_calibration(self.calibration_path)


class VisibleHomeTests(unittest.TestCase):
    def setup_study(self, visibility_results):
        study = SharedStationStudy.__new__(SharedStationStudy)
        study.station_design = {}
        study.args = SimpleNamespace(require_camera_visible_home=True)
        study.validator = SimpleNamespace(check_segment=lambda *a, **k: {'ok': True},
                                          validate_cycle=lambda *a, **k: {'ok': True})
        camera = np.eye(4)
        camera[:3, 3] = [.375, -.002, 1.273]
        contexts = {f: SimpleNamespace(cfg={'camera_T_base_link': camera.tolist(),
                                           'box_T_base_link': np.eye(4).tolist()}) for f in FIXTURES}
        waist = np.array([-.4, .3, .1, -1.7])
        paths = {f: {'home_arm_rad': np.zeros(7),
                     'outward_waist_arm_rad': [np.r_[waist, np.zeros(7)], np.r_[waist, np.zeros(7)]],
                     'outward_flange_poses': [np.eye(4), np.eye(4)]} for f in FIXTURES}
        results = iter(visibility_results)
        checked = []

        def visibility(ctx, w, candidate):
            np.testing.assert_array_equal(candidate['world_T_camera'], camera)
            np.testing.assert_array_equal(w, waist)
            self.assertEqual(set(candidate['world_T_boxes']), set(FIXTURES))
            checked.append(candidate['path']['home_arm_rad'].copy())
            return {'all_sample_rays_clear': next(results), 'fixtures': {}}

        study.home_camera_visibility = visibility
        return study, contexts, waist, paths, checked

    def test_collision_free_but_camera_blocking_homes_cannot_pass(self):
        study, contexts, waist, paths, checked = self.setup_study([False] * 4)
        result = study.unify_homes(contexts, waist, paths)
        self.assertFalse(result['passed'])
        self.assertEqual(len(checked), 4)
        self.assertTrue(all(not t['camera_visibility']['all_sample_rays_clear']
                            and not t['connectors'] for t in result['trials']))

    def test_blocked_first_home_is_skipped_for_a_visible_home(self):
        study, contexts, waist, paths, checked = self.setup_study([False, True])
        result = study.unify_homes(contexts, waist, paths)
        self.assertTrue(result['passed'])
        self.assertEqual(result['home_source'], 'JM160')
        self.assertEqual(len(checked), 2)
        self.assertTrue(result['camera_visibility']['all_sample_rays_clear'])


if __name__ == '__main__':
    unittest.main()
