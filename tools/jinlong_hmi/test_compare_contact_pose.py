import unittest

import numpy as np

from compare_contact_pose import compare, difference


class ContactComparisonTests(unittest.TestCase):
    def test_error_axes_follow_rotated_box_instead_of_camera(self):
        reference = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1.]])
        estimated = reference.copy()
        estimated[:3, 3] += [.01, .02, -.03]
        result = difference(estimated, reference)
        np.testing.assert_allclose(result['delta_camera_mm'], [10, 20, -30])
        np.testing.assert_allclose(result['delta_box_mm'], [20, -10, -30])
        self.assertAlmostEqual(result['rotation_error_deg'], 0)

    def test_independent_contact_inverts_production_target_with_all_offsets(self):
        box = np.array([[0, -1, 0, .1], [1, 0, 0, -.2], [0, 0, 1, .5], [0, 0, 0, 1.]])
        tool = np.array([[0, 1, 0, .02], [-1, 0, 0, .03], [0, 0, 1, .04], [0, 0, 0, 1.]])
        offset = np.eye(4); offset[:3, 3] = [.05, .06, .07]
        site = {'selected_fixture': 'JL132', 'flange_T_tcp': tool.tolist(),
                'tcp_z_adjustment_mm': 5, 'fixture_z_adjustments_mm': {'JL132': -22},
                'box_center_adjustments_mm': {'JL132': [10, -20, 30]},
                'tcp_T_objects': {'JL132': offset.tolist()}}
        z = np.eye(4); z[2, 3] = .005
        adjusted_offset = offset.copy(); adjusted_offset[2, 3] -= .022
        shift = np.eye(4); shift[:3, 3] = [.01, -.02, .03]
        flange = box @ shift @ np.linalg.inv(adjusted_offset) @ np.linalg.inv(tool @ z)
        flange[:3, 3] -= box[:3, 2] * .011
        nominal = flange.copy(); nominal[0, 3] += .001
        frames = {'current': {'waist_model_rad': [0, np.pi/2, 0, 0], 'left_rad': [0]*7,
                  'frames': {'camera_T_left_flange_feedback': flange.tolist(),
                             'camera_T_left_flange': nominal.tolist()}}}
        vision = {'success': True, 'center_pose_ready': True, 'model_type': '132', 'camera_T_box': box.tolist()}
        result = compare(frames, site, vision, gap_mm=11)
        self.assertLess(result['sdk_contact_vs_vision']['position_error_mm'], 1e-9)
        self.assertAlmostEqual(result['nominal_vs_sdk_flange']['position_error_mm'], 1)
        np.testing.assert_allclose(result['waist_controller_deg'], [0]*4)
        self.assertGreater(compare(frames, site, vision)['sdk_contact_vs_vision']['position_error_mm'], 10)
        vision['model_type'] = '160'
        with self.assertRaises(ValueError): compare(frames, site, vision)

    def test_invalid_rotations_and_nonfinite_inputs_are_rejected(self):
        for invalid in (np.zeros((4, 4)), np.eye(4) * float('nan')):
            with self.assertRaises(ValueError): difference(invalid, np.eye(4))


if __name__ == '__main__':
    unittest.main()
