import unittest

import numpy as np

from .annotate_sim_gt import _segmentation_mask


class SimulatorAnnotationTests(unittest.TestCase):
    def test_renderer_class_id_is_resized_to_policy_resolution(self):
        seg = np.zeros((4, 8, 4), np.float32)
        seg[1:3, 2:6, 0] = 50 / 255.
        mask = _segmentation_mask(seg, 50, (2, 4))
        self.assertEqual(mask.shape, (2, 4))
        self.assertEqual(int(mask.sum()), 2)


if __name__ == "__main__":
    unittest.main()
