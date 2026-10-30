import unittest
import numpy as np
from .detector import select_detection, red_support_filter

class DetectionTests(unittest.TestCase):
    def test_color_support_rejects_background(self):
        image=np.full((20,20,3),100,dtype=np.uint8)
        image[4:8,4:8]=[100,10,10]
        keep=red_support_filter(image,[[4,4,8,8],[10,10,18,18],[0,0,20,20]])
        np.testing.assert_array_equal(keep,[True,False,False])

    def test_brown_table_is_not_red_support(self):
        image=np.full((20,20,3),[120,60,25],dtype=np.uint8)
        self.assertFalse(red_support_filter(image,[[0,0,20,20]])[0])

    def test_missing_is_invalid_not_full_image(self):
        d = select_detection([], [], 120, 90)
        self.assertFalse(d.valid)
        np.testing.assert_array_equal(d.box, np.zeros(4))

    def test_clipping_coordinates_and_bad_candidates(self):
        d = select_detection([[-5, 10, 130, 80], [5, 4, 2, 9],
                              [0, 0, float("nan"), 4]], [.8, .9, .99], 120, 90)
        self.assertTrue(d.valid)
        np.testing.assert_allclose(d.box, [0, 10/90, 1, 80/90])
        self.assertEqual(d.candidates, 1)

    def test_duplicate_and_distinct_candidates(self):
        d = select_detection([[0,0,20,20], [1,1,20,20], [60,40,80,60]],
                             [.8,.7,.6], 120,90)
        self.assertTrue(d.ambiguous)
        self.assertEqual(d.candidates,2)

    def test_threshold_and_empty_after_filter(self):
        d = select_detection([[0,0,20,20]], [.2], 120,90,.25)
        self.assertFalse(d.valid)

if __name__ == "__main__":
    unittest.main()
