import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from plan_vision_hover import reproject_calibration


class ReprojectionTests(unittest.TestCase):
    def test_world_camera_is_preserved_and_old_acceptance_never_transfers(self):
        original_base = np.eye(4)
        original_base[:3, :3] = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
        original_base[:3, 3] = [.1, .2, .3]
        work_base = np.eye(4)
        work_base[:3, :3] = Rotation.from_euler("xyz", [-25, 15, -40], degrees=True).as_matrix()
        work_base[:3, 3] = [-.2, .1, .4]
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler("xyz", [40, -20, 5], degrees=True).as_matrix()
        camera[:3, 3] = [.4, -.3, .2]
        calibration = {"base_T_camera": camera.tolist(), "waist_model_rad": [0]*4,
                       "independent_validation": {"accepted": True, "count": 3},
                       "source_session": "original", "sha256": "old"}
        original = copy.deepcopy(calibration)
        candidate = reproject_calibration(calibration, original_base, work_base, [.1]*4)
        np.testing.assert_allclose(work_base @ candidate["base_T_camera"], original_base @ camera, atol=1e-12)
        self.assertEqual(calibration, original)
        self.assertFalse(candidate["independent_validation"]["accepted"])
        self.assertFalse(candidate["derivation"]["physical_application"])
        self.assertEqual(candidate["source_session"], "original")
        self.assertNotIn("sha256", candidate)


if __name__ == "__main__":
    unittest.main()
