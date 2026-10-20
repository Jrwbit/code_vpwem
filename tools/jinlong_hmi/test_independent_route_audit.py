import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from independent_route_audit import DIMENSIONS, error, flange, inverse, link_transform, transform


class IndependentKinematicsTests(unittest.TestCase):
    def test_zero_arm_is_straight_at_sum_of_axial_dimensions(self):
        np.testing.assert_allclose(flange(np.zeros(7)),
                                   transform(translation=[0, 0, .8745]), atol=1e-12)

    def test_pose_error_reports_geodesic_angle_and_metric_translation(self):
        t = transform(Rotation.from_euler('z', 30, degrees=True).as_matrix(), [.003, .004, 0])
        e = error(t, np.eye(4))
        self.assertAlmostEqual(e['position_mm'], 5)
        self.assertAlmostEqual(e['rotation_deg'], 30)

    def test_inverse_recovers_pose_from_different_joint_guess(self):
        q = np.radians([-100, -60, 40, 70, 30, 20, -25])
        bounds = np.radians([[-176, 176], [-118, 118], [-176, 176],
                             [-58, 143], [-176, 176], [-58, 58], [-87, 87]])
        solved = inverse(flange(q), q + .04, bounds)
        e = error(flange(solved), flange(q))
        self.assertLess(e['position_mm'], .02)
        self.assertLess(e['rotation_deg'], .005)
        self.assertTrue(np.all((solved >= bounds[:, 0]) & (solved <= bounds[:, 1])))

    def test_urdf_mount_rotates_after_origin_translation(self):
        xml = '''<robot name="test"><joint name="waist" type="revolute">
        <parent link="world"/><child link="body"/><origin xyz="1 0 0"/>
        <axis xyz="0 0 1"/></joint><joint name="mount" type="fixed">
        <parent link="body"/><child link="arm"/><origin xyz="2 0 0"/>
        </joint></robot>'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'robot.urdf'
            path.write_text(xml)
            t = link_transform(path, 'arm', {'waist': np.pi / 2})
        np.testing.assert_allclose(t[:3, 3], [1, 2, 0], atol=1e-12)
        np.testing.assert_allclose(t[:3, :3], Rotation.from_euler('z', 90, degrees=True).as_matrix())

    def test_inverse_uses_supplied_effective_dimensions(self):
        rows = np.array(DIMENSIONS, float)
        rows[2, 2] += 5
        q = np.radians([-100, -60, 40, 70, 30, 20, -25])
        bounds = np.radians([[-176,176],[-118,118],[-176,176],[-58,143],[-176,176],[-58,58],[-87,87]])
        target = flange(q, rows)
        solved = inverse(target, q + .04, bounds, dimensions=rows)
        self.assertLess(error(flange(solved, rows), target)['position_mm'], .02)
        self.assertGreater(error(flange(solved), target)['position_mm'], 1)


if __name__ == '__main__':
    unittest.main()
