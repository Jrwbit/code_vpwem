import unittest

import numpy as np

from .validate_sidecars import quality_gate


class QualityGateTests(unittest.TestCase):
    def test_rejects_ambiguous_low_score_and_bad_geometry(self):
        boxes = np.array([
            [.1, .1, .2, .2],
            [.1, .1, .2, .2],
            [.1, .1, .2, .2],
            [.8, .2, .1, .3],
        ], np.float32)
        accepted = quality_gate(
            boxes, [True, True, True, True], [False, True, False, False],
            [.8, .9, .2, .8], .35)
        np.testing.assert_array_equal(accepted, [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
